from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    # Hugging Face & W&B credentials
    hf_token: SecretStr | None = Field(
        None, description="Hugging Face API token for authentication"
    )
    hub_model_id: str = Field(description="HuggingFace model repository name")
    wandb_api_key: SecretStr | None = Field(
        None, description="Weights & Biases API key"
    )
    wandb_project: str = Field(
        "gpt2-dedup-experiments", description="Weights & Biases project name"
    )

    # Dataset & model
    dataset_path: str = Field(
        "batragaurav2616/deduped-c4-mini",
        description="Hugging Face dataset repo path, e.g. 'username/dataset-name'",
    )
    dataset_config: str = Field(description="Dataset config, e.g., 'en'")
    dataset_n_shards: int = Field(description="number of shards in the dataset")
    model_name: str = Field("gpt2", description="Model name or path for fine‑tuning")
    output_dir: str = Field(
        "outputs/gpt2-dedup", description="Directory to save outputs and checkpoints"
    )

    # Training hyperparameters
    batch_size: int = Field(8, description="Per‑device training batch size")
    epochs: int = Field(3, description="Number of training epochs")
    learning_rate: float = Field(5e-5, description="Learning rate")
    max_seq_length: int = Field(512, description="Maximum sequence length")
    logging_steps: int = Field(100, description="Number of logging steps")
    eval_steps: int = Field(100, description="Number of evaluation steps")
    is_fp16: bool = Field(True, description="toggle fp16 quantization")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
