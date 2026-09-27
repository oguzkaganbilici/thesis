from dataclasses import dataclass, field


@dataclass
class Config:

    # path'ler
    model: str = "triplesumm"
    dataset: str = "trt"
    data_dir: str = "/Users/oguzkaganbilici/Desktop/pipelines"
    exp_name: str = "smoke_test"
    output_dir: str = "/Users/oguzkaganbilici/Desktop/triplesumm/thesis/Phase 3/triplesumm/outputs/smoke_test"
    pretrained_ckpt: str = "/Users/oguzkaganbilici/Desktop/triplesumm/thesis/Phase 3/triplesumm/checkpoint/best_model_ckpt_mrhisum.pth"

    # modelin boyutu - mimari ayarları
    visual_dim: int = 2048
    text_dim: int = 768
    audio_dim: int = 768
    d_model: int = 128
    hidden_dim: int = 192
    num_heads: int = 4
    num_layers: int = 4
    window_size: list = field(default_factory=lambda: [5, 15, 45, 0])
    dropout: float = 0.1
    max_seq_len: int = 10_000

    # egitim ayarları
    device: str = "mps"
    num_epochs: int = 100
    patience: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    get_attn_weights: bool = False
    model_ckpt: str = ""
    wandb: bool = False
    batch_size: int = 1