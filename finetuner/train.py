import math
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
    PerplexityCallback,
    ManualThroughputCallback,
    log_generation_metrics,
    log_generation_table,
    log_generation_length_histogram,
    log_generation_wordcloud,
)


def main():
    # 1) Load config & start W&B
    cfg = Config()
    run = wandb.init(project=cfg.wandb_project, config=cfg.model_dump())

    # 2) Load + tokenize your full dataset
    ds, tokenizer = load_and_tokenize(
        dataset_name=cfg.dataset_path,
        subfolder=cfg.dataset_config,
        n_shards=cfg.dataset_n_shards,
        seq_len=cfg.max_seq_length,
        model_name=cfg.model_name,
    )
    train_ds, val_ds = ds.train_test_split(test_size=0.01).values()

    # 3) Optionally log dataset stats
    wandb.config.update(
        {
            "train_size": len(train_ds),
            "val_size": len(val_ds),
        }
    )

    # 4) Model & data collator
    model = GPT2LMHeadModel.from_pretrained(cfg.model_name)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5) Training arguments (log every logging_steps steps, eval each epoch)
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

    # 6) Trainer with our callbacks & eval‐time perplexity
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        callbacks=[
            PerplexityCallback(),
            ManualThroughputCallback(cfg.max_seq_length),
        ],
        compute_metrics=lambda eval_pred: {"eval_perplexity": math.exp(eval_pred.loss)},
    )

    # 7) Train & push to HF Hub
    trainer.train()
    trainer.push_to_hub(
        commit_message=f"Run: {run.name}, Dataset_name: {cfg.dataset_path}, subfolder: {cfg.dataset_config}, n_shards: {cfg.dataset_n_shards}, batch_size: {cfg.batch_size}, epochs: {cfg.epochs}, learning_rate: {cfg.learning_rate}",
        branch=f"{run.name}-{cfg.dataset_path}",
    )

    # 8) Generation‐quality metrics on a fixed prompt set
    prompts = ["The meaning of life is", "In a world where AI"]
    gen = pipeline(
        "text-generation",
        model=trainer.model,
        tokenizer=tokenizer,
        device=trainer.args.device,
    )
    outputs = [o[0]["generated_text"] for o in gen(prompts, max_length=50)]

    # 9) Log diversity metrics
    log_generation_metrics(outputs)

    # 10) Log prompt⇾generation table
    log_generation_table(prompts, outputs)

    # 11) Log histogram of generation lengths
    log_generation_length_histogram(outputs)

    # 12) Log word‑cloud image of generated text
    log_generation_wordcloud(outputs)

    wandb.finish()


if __name__ == "__main__":
    main()
