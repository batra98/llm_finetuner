import re
from typing import Tuple, List, Dict
from collections import defaultdict

from datasets import load_dataset, Dataset
from transformers import GPT2TokenizerFast
from huggingface_hub import hf_hub_download, HfApi


def _download_shards(
    repo_id: str,
    subfolder: str,
    n_shards: int,
) -> List[str]:
    """
    Auto-discover and download all parts of chunk_1…chunk_n shards,
    regardless of how many parts each has.

    Returns a list of local file paths, sorted by (chunk_idx, part_idx).
    """
    api = HfApi()
    # 1) List all files in the dataset repo
    all_files = api.list_repo_files(repo_id, repo_type="dataset")

    # 2) Filter for parquet parts under the given subfolder
    pattern = re.compile(
        rf"^{re.escape(subfolder)}/chunk_(\d+)-(\d+)-of-(\d+)\.parquet$"
    )
    # Map chunk_idx -> list of (part_idx, filename)
    file_map: Dict[int, List[tuple[int, str]]] = defaultdict(list)
    for f in all_files:
        m = pattern.match(f)
        if not m:
            continue
        chunk_idx = int(m.group(1))
        part_idx = int(m.group(2))
        file_map[chunk_idx].append((part_idx, f))

    # 3) For each shard 1..n_shards, sort by part_idx and download
    paths: List[str] = []
    for i in range(1, n_shards + 1):
        parts = file_map.get(i)
        if not parts:
            raise ValueError(
                f"No parquet files found for chunk {i} under {repo_id}/{subfolder}"
            )
        parts.sort(key=lambda x: x[0])
        for _, filename in parts:
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
      - the deduped parquet shards (n_shards > 0), auto-discovering parts, or
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
        # ── Baseline: stream exactly N examples ───────────────────────────
        stream = load_dataset(
            dataset_name,
            subfolder,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        examples = []
        for idx, ex in enumerate(stream):
            examples.append(ex)
            if baseline_num_samples and idx + 1 >= baseline_num_samples:
                break
        ds = Dataset.from_list(examples)

    else:
        # ── Dedup: download all parts, concat via parquet loader ───────────
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

    # ── Tokenize ─────────────────────────────────────────────────────────
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        )

    ds = ds.map(
        _tokenize,
        batched=True,
        remove_columns=ds.column_names,
        keep_in_memory=True,  # avoids extra disk use
    )
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return ds, tokenizer
