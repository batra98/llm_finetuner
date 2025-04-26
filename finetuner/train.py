import os
import wandb
from accelerate import Accelerator
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.pipelines import pipeline

from config import Config
from data import load_and_tokenize
from metrics import (
    EvaluateCallback,
    PerplexityCallback,
    ManualThroughputCallback,
    TorchProfilerCallback,
    log_generation_metrics,
    log_generation_table,
    log_generation_length_histogram,
    log_generation_wordcloud,
)
from torch.utils.tensorboard import SummaryWriter


def main():
    # 1) Initialize Accelerate
    accelerator = Accelerator()

    # 2) Load config
    cfg = Config()

    # 3) Only the main process should start a real W&B run
    if accelerator.is_main_process:
        run = wandb.init(project=cfg.wandb_project, config=cfg.model_dump())
        tb_writer = SummaryWriter(log_dir=f"{cfg.hub_model_id}")
    else:
        # Disable W&B on other processes
        os.environ["WANDB_MODE"] = "disabled"

    # 4) Load & tokenize dataset
    ds, tokenizer = load_and_tokenize(
        dataset_name=cfg.dataset_path,
        subfolder=cfg.dataset_config,
        n_shards=cfg.dataset_n_shards,
        seq_len=cfg.max_seq_length,
        model_name=cfg.model_name,
        baseline=cfg.is_baseline,
        baseline_num_samples=cfg.baseline_num_samples,
    )
    train_ds, val_ds = ds.train_test_split(test_size=0.01).values()

    # 5) Log dataset sizes (main process only)
    if accelerator.is_main_process:
        wandb.config.update(
            {
                "train_size": len(train_ds),
                "val_size": len(val_ds),
            }
        )

    # 6) Prepare model & data collator
    model = GPT2LMHeadModel.from_pretrained(cfg.model_name)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 7) Training arguments
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
        hub_model_id=f"{cfg.hub_model_id}",
        hub_strategy="end",
        eval_strategy="epoch",
        fp16=cfg.is_fp16,
    )

    # 8) Trainer with our callbacks & eval‐time perplexity
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
        # compute_metrics=lambda eval_pred: {"eval_perplexity": math.exp(eval_pred.loss)},
    )

    # 9) Train
    trainer.train()

    # 10) Push to HF Hub from main process only
    if accelerator.is_main_process:
        artifact = wandb.Artifact(
            f"tensorboard-logs-{cfg.hub_model_id}".replace("/", "_"), type="tensorboard"
        )
        artifact.add_dir(f"{cfg.hub_model_id}")
        wandb.log_artifact(artifact)

        trainer.push_to_hub(
            commit_message=(
                f"Run: {run.name}, Dataset: {cfg.dataset_path}/{cfg.dataset_config}, "
                f"shards={cfg.dataset_n_shards}, bs={cfg.batch_size}, "
                f"epochs={cfg.epochs}, lr={cfg.learning_rate}"
            ),
        )

        # 11) Generation‐quality metrics on a fixed prompt set
        prompts = ["The meaning of life is", "In a world where AI"]
        gen = pipeline(
            "text-generation",
            model=trainer.model,
            tokenizer=tokenizer,
            device=trainer.args.device,
        )
        outputs = [o[0]["generated_text"] for o in gen(prompts, max_length=50)]

        # 12) Log diversity & visualization metrics
        log_generation_metrics(outputs)
        log_generation_table(prompts, outputs)
        log_generation_length_histogram(outputs)
        log_generation_wordcloud(outputs)

        tb_writer.close()

        wandb.finish()


if __name__ == "__main__":
    main()
