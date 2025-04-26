import os
import math
import wandb
import torch.distributed as dist
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers.models.gpt2 import GPT2TokenizerFast
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.pipelines import pipeline

from config import Config
from metrics import (
    PerplexityCallback,
    ManualThroughputCallback,
    EvaluateCallback,
    TorchProfilerCallback,
    log_generation_metrics,
    log_generation_table,
    log_generation_length_histogram,
    log_generation_wordcloud,
)
from torch.utils.tensorboard import SummaryWriter


def main():
    accelerator = Accelerator()
    cfg = Config()
    is_main = accelerator.is_main_process

    # Only rank 0 does W&B
    if is_main:
        wandb.init(project=cfg.wandb_project, config=cfg.model_dump())
        tb_writer = SummaryWriter(log_dir=f"{cfg.hub_model_id}")
    else:
        os.environ["WANDB_MODE"] = "disabled"

    # ─── Dataset loading ────────────────────────────────────────────────
    if cfg.is_baseline:
        # 1) Rank 0 streams exactly baseline_num_samples examples
        examples = None
        if is_main:
            stream = load_dataset(
                cfg.dataset_path,
                cfg.dataset_config,
                split="train",
                streaming=True,
            )
            examples = []
            for i, ex in enumerate(stream):
                examples.append(ex)
                if i + 1 >= cfg.baseline_num_samples:
                    break
        else:
            examples = []  # placeholder for non‐main

        # 2) Broadcast that one list from rank 0 → all ranks
        obj_list = [examples]
        dist.broadcast_object_list(obj_list, src=0)
        examples = obj_list[0]

        # 3) Build an in-memory Dataset
        ds = Dataset.from_list(examples)

        # 4) Tokenize
        tokenizer = GPT2TokenizerFast.from_pretrained(cfg.model_name)
        tokenizer.pad_token = tokenizer.eos_token

        def tok_fn(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=cfg.max_seq_length,
                padding="max_length",
            )

        ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    else:
        # Dedup mode: your existing loader
        from data import load_and_tokenize

        ds, tokenizer = load_and_tokenize(
            dataset_name=cfg.dataset_path,
            subfolder=cfg.dataset_config,
            n_shards=cfg.dataset_n_shards,
            seq_len=cfg.max_seq_length,
            model_name=cfg.model_name,
        )

    # ─── Train/val split ────────────────────────────────────────────────
    train_ds, val_ds = ds.train_test_split(test_size=0.01).values()
    if is_main:
        wandb.config.update({"train_size": len(train_ds), "val_size": len(val_ds)})

    # ─── Model & collator ──────────────────────────────────────────────
    model = GPT2LMHeadModel.from_pretrained(cfg.model_name)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ─── TrainingArguments ──────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,
        report_to="wandb",
        save_strategy="epoch",
        push_to_hub=True,
        hub_model_id=cfg.hub_model_id,
        hub_strategy="end",
        eval_strategy="epoch",
        fp16=cfg.is_fp16,
    )

    # ─── Trainer & Callbacks ───────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        callbacks=[
            PerplexityCallback(),
            ManualThroughputCallback(cfg.max_seq_length),
            EvaluateCallback(),
            TorchProfilerCallback(output_dir=f"{cfg.hub_model_id}"),
        ],
    )

    # ─── Train ──────────────────────────────────────────────────────────
    trainer.train()

    # ─── Push & Generation Metrics (main only) ──────────────────────────
    if is_main:
        artifact = wandb.Artifact(
            f"tensorboard-logs-{cfg.hub_model_id}".replace("/", "_"), type="tensorboard"
        )
        artifact.add_dir(f"{cfg.hub_model_id}")
        wandb.log_artifact(artifact)

        trainer.push_to_hub(
            commit_message=(
                f"Dataset={cfg.dataset_path}/{cfg.dataset_config}, "
                f"baseline={cfg.is_baseline}, samples={cfg.baseline_num_samples}, "
                f"bs={cfg.batch_size}, epochs={cfg.epochs}, lr={cfg.learning_rate}"
            )
        )

        prompts = ["The meaning of life is", "In a world where AI"]
        gen = pipeline(
            "text-generation",
            model=trainer.model,
            tokenizer=tokenizer,
            device=trainer.args.device,
        )
        outputs = [o[0]["generated_text"] for o in gen(prompts, max_length=50)]

        log_generation_metrics(outputs)
        log_generation_table(prompts, outputs)
        log_generation_length_histogram(outputs)
        log_generation_wordcloud(outputs)

        tb_writer.close()

        wandb.finish()


if __name__ == "__main__":
    main()
