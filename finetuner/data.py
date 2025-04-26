from typing import Tuple, List
from datasets import load_dataset, Dataset
from transformers import GPT2TokenizerFast
from huggingface_hub import hf_hub_download


def _download_shards(
    repo_id: str,
    subfolder: str,
    n_shards: int,
) -> List[str]:
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
    subfolder: str,
    model_name: str,
    seq_len: int,
    *,
    # Dedup mode
    n_shards: int = 0,
    # Baseline mode
    baseline: bool = False,
    baseline_num_samples: int = 0,
) -> Tuple[Dataset, GPT2TokenizerFast]:
    """
    Load & tokenize either:
      - the deduped parquet shards (n_shards > 0), or
      - a baseline HF dataset split (streaming baseline_num_samples examples).

    Args:
      dataset_name: HF repo, e.g. "batragaurav2616/deduped-c4-mini"
      subfolder:    for dedup: the config folder name; for baseline: the config name
      model_name:   e.g. "gpt2"
      seq_len:      max token length
      n_shards:     >0 to use dedup shards mode
      baseline:     True to use streaming baseline mode instead
      baseline_num_samples: how many examples to pull in baseline mode
    """
    if baseline:
        # ── Baseline mode: stream exactly N examples ───────────────────────────
        stream = load_dataset(
            dataset_name,
            subfolder,
            split="train",
            streaming=True,
        )
        examples = []
        for i, ex in enumerate(stream):
            examples.append(ex)
            if baseline_num_samples and i + 1 >= baseline_num_samples:
                break
        ds = Dataset.from_list(examples)

    else:
        # ── Dedup mode: download & concat parquet shards ───────────────────────
        parquet_files = _download_shards(
            repo_id=dataset_name,
            subfolder=subfolder,
            n_shards=n_shards,
        )
        ds = load_dataset(
            "parquet",
            data_files={"train": parquet_files},
            split="train",
        )

    # ── Tokenize ─────────────────────────────────────────────────────────────
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        )

    ds = ds.map(_tokenize, batched=True, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return ds, tokenizer
