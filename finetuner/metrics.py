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


def log_generation_table(prompts: Sequence[str], outputs: Sequence[str]):
    """
    Log a W&B table with columns [prompt, generation].
    """
    table = wandb.Table(columns=["prompt", "generation"])
    for prompt, gen in zip(prompts, outputs):
        table.add_data(prompt, gen)
    wandb.log({"generation_table": table})


def log_generation_length_histogram(outputs: Sequence[str]):
    """
    Log a histogram of generation lengths (in tokens).
    """
    lengths = [len(gen.split()) for gen in outputs]
    wandb.log({"generation_length": wandb.Histogram(lengths)})


def log_generation_wordcloud(
    outputs: Sequence[str], title: str = "generation_wordcloud"
):
    """
    Generate a word‑cloud from the list of generated strings and log it to W&B.
    """
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
    wandb.log({title: wandb.Image(img)})


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
        if logs and "loss" in logs:
            wandb.log({"train_perplexity": math.exp(logs["loss"])}, commit=False)


class ManualThroughputCallback(TrainerCallback):
    """
    Compute and log tokens/sec and steps/sec at each logging event.
    """

    def __init__(self, seq_length: int):
        self.seq_length = seq_length

    def on_train_begin(self, args, state, control, **kwargs):
        self._t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        elapsed = time.time() - self._t0
        samples = state.global_step * args.per_device_train_batch_size * args.world_size
        tokens_per_sec = samples * self.seq_length / elapsed
        wandb.log({"tokens_per_second": tokens_per_sec}, commit=False)

        if logs and "train_steps_per_second" in logs:
            wandb.log(
                {"steps_per_second": logs["train_steps_per_second"]},
                commit=False,
            )


class EvaluateCallback(TrainerCallback):
    """
    After each evaluation, log validation_perplexity and its delta
    relative to the previous epoch.
    """

    def __init__(self):
        self.last_val_ppl = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # HuggingFace returns 'eval_loss'; convert to perplexity
        val_loss = metrics.get("eval_loss")
        if val_loss is None:
            return
        val_ppl = math.exp(val_loss)

        # log current val PPL
        wandb.log({"validation_perplexity": val_ppl})

        # log change since last eval
        if self.last_val_ppl is not None:
            wandb.log({"delta_val_perplexity": self.last_val_ppl - val_ppl})
        self.last_val_ppl = val_ppl


def log_generation_metrics(texts):
    """
    After generation, log distinct-1, distinct-2, and repeat-rate.
    """
    d1 = compute_distinct_n(texts, 1)
    d2 = compute_distinct_n(texts, 2)
    rr = repeat_rate(texts)
    wandb.log(
        {
            "distinct_1": d1,
            "distinct_2": d2,
            "repeat_rate": rr,
        }
    )
