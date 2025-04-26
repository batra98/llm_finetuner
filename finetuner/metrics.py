import math
import time
import wandb
from transformers.trainer_callback import TrainerCallback
from collections import Counter
from typing import Sequence
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from torch.profiler import (
    profile,
    ProfilerActivity,
    tensorboard_trace_handler,
)
import torch


def _wandb_log_safe(*args, **kwargs):
    """
    Wrapper around wandb.log that only logs if a run is active.
    """
    if wandb.run is not None:
        wandb.log(*args, **kwargs)


def log_generation_table(prompts: Sequence[str], outputs: Sequence[str]):
    """
    Log a W&B table with columns [prompt, generation].
    """
    if wandb.run is None:
        return
    table = wandb.Table(columns=["prompt", "generation"])
    for prompt, gen in zip(prompts, outputs):
        table.add_data(prompt, gen)
    _wandb_log_safe({"generation_table": table})


def log_generation_length_histogram(outputs: Sequence[str]):
    """
    Log a histogram of generation lengths (in tokens).
    """
    if wandb.run is None:
        return
    lengths = [len(gen.split()) for gen in outputs]
    _wandb_log_safe({"generation_length": wandb.Histogram(lengths)})


def log_generation_wordcloud(
    outputs: Sequence[str], title: str = "generation_wordcloud"
):
    """
    Generate a word‑cloud from the list of generated strings and log it to W&B.
    """
    if wandb.run is None:
        return

    # 1) build the cloud
    combined = " ".join(outputs)
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=set(STOPWORDS),
        collocations=False,
    ).generate(combined)

    # 2) render to buffer
    buf = io.BytesIO()
    plt.figure(figsize=(8, 4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    # 3) load into PIL and log
    img = Image.open(buf)
    _wandb_log_safe({title: wandb.Image(img)})


def compute_distinct_n(texts, n):
    """
    Compute Distinct-n: (# unique n‑grams) / (total n‑grams across all texts).
    """
    gram_counts = Counter()
    total_ngrams = 0
    for t in texts:
        toks = t.split()
        for i in range(len(toks) - n + 1):
            gram = " ".join(toks[i : i + n])
            gram_counts[gram] += 1
            total_ngrams += 1
    return len(gram_counts) / total_ngrams if total_ngrams > 0 else 0.0


def repeat_rate(texts):
    """
    Fraction of immediate token repeats across all generated texts.
    """
    total = 0
    repeats = 0
    for t in texts:
        toks = t.split()
        for i in range(1, len(toks)):
            total += 1
            if toks[i] == toks[i - 1]:
                repeats += 1
    return repeats / total if total > 0 else 0.0


class PerplexityCallback(TrainerCallback):
    """
    Log exp(train_loss) as train_perplexity at each logging event.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if wandb.run is None or logs is None or "loss" not in logs:
            return
        _wandb_log_safe({"train_perplexity": math.exp(logs["loss"])}, commit=False)


class ManualThroughputCallback(TrainerCallback):
    """
    Compute and log tokens/sec and steps/sec at each logging event.
    """

    def __init__(self, seq_length: int):
        self.seq_length = seq_length

    def on_train_begin(self, args, state, control, **kwargs):
        self._t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if wandb.run is None:
            return
        elapsed = time.time() - self._t0
        samples = state.global_step * args.per_device_train_batch_size * args.world_size
        tokens_per_sec = samples * self.seq_length / elapsed
        _wandb_log_safe({"tokens_per_second": tokens_per_sec}, commit=False)

        if logs and "train_steps_per_second" in logs:
            _wandb_log_safe(
                {"steps_per_second": logs["train_steps_per_second"]},
                commit=False,
            )


class EvaluateCallback(TrainerCallback):
    """
    After each evaluation, log validation_perplexity and its delta.
    Also at the very end of training, log the final_validation_perplexity.
    """

    def __init__(self):
        self.last_val_ppl = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if wandb.run is None or metrics is None or "eval_loss" not in metrics:
            return
        val_loss = metrics["eval_loss"]
        val_ppl = math.exp(val_loss)

        _wandb_log_safe({"validation_perplexity": val_ppl})

        if self.last_val_ppl is not None:
            _wandb_log_safe({"delta_val_perplexity": self.last_val_ppl - val_ppl})
        self.last_val_ppl = val_ppl

    def on_train_end(self, args, state, control, **kwargs):
        # Log the final validation perplexity once at the end
        if wandb.run is not None and self.last_val_ppl is not None:
            _wandb_log_safe({"final_validation_perplexity": self.last_val_ppl})


def log_generation_metrics(texts):
    """
    After generation, log distinct-1, distinct-2, and repeat-rate.
    """
    if wandb.run is None:
        return
    d1 = compute_distinct_n(texts, 1)
    d2 = compute_distinct_n(texts, 2)
    rr = repeat_rate(texts)
    _wandb_log_safe(
        {
            "distinct_1": d1,
            "distinct_2": d2,
            "repeat_rate": rr,
        }
    )


class TorchProfilerCallback(TrainerCallback):
    """
    Runs a PyTorch profiler over the training loop, dumps
    a Chrome trace via tensorboard_trace_handler, and then
    logs total runtime & peak GPU memory to W&B.
    """

    def __init__(self, output_dir: str = "./tb_logs"):
        self.output_dir = output_dir
        self.prof = None

    def on_train_begin(self, args, state, control, **kwargs):
        # mark start time
        self._t0 = time.time()
        # enter profiler context
        self.prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=tensorboard_trace_handler(self.output_dir),
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        )
        self.prof.__enter__()

    def on_step_end(self, args, state, control, **kwargs):
        if self.prof:
            self.prof.step()

    def on_train_end(self, args, state, control, **kwargs):
        # exit and write trace
        if self.prof:
            self.prof.__exit__(None, None, None)
        # compute runtime & peak memory
        elapsed = time.time() - self._t0
        stats = {"train_runtime_sec": elapsed}

        if torch.cuda.is_available():
            # bytes → GiB
            max_alloc = torch.cuda.max_memory_allocated() / (1024**3)
            max_reserved = torch.cuda.max_memory_reserved() / (1024**3)
            stats.update(
                {
                    "peak_memory_allocated_GiB": max_alloc,
                    "peak_memory_reserved_GiB": max_reserved,
                }
            )

        _wandb_log_safe(stats)
