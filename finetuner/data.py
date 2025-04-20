from datasets import load_dataset
from transformers.models.gpt2 import GPT2TokenizerFast
from typing import Tuple
from huggingface_hub import hf_hub_download


def _download_shards(
    repo_id: str,
    subfolder: str,
    n_shards: int,
) -> list[str]:
    """
    Download chunk_1…chunk_n.parquet from a HF dataset repo and return their local paths.
    """
    paths = []
    for i in range(1, n_shards + 1):
        filename = f"{subfolder}/chunk_{i}-00000-of-00001.parquet"
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
        )
        paths.append(local_path)
    return paths


def load_and_tokenize(
    dataset_name: str,
    seq_len: int,
    n_shards: int,
    subfolder: str,
    model_name: str,
) -> Tuple:
    parquet_files = _download_shards(
        repo_id=dataset_name, subfolder=subfolder, n_shards=n_shards
    )

    ds = load_dataset("parquet", data_files={"train": parquet_files}, split="train")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=seq_len, padding="max_length"
        )

    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return ds, tokenizer
