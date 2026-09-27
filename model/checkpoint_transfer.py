import torch
from models.model import TripleSumm


def build_model_with_transfer(cfg):
    model = TripleSumm(
        visual_dim=cfg.visual_dim,
        text_dim=cfg.text_dim,
        audio_dim=cfg.audio_dim,
        d_model=cfg.d_model,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        window_size=cfg.window_size,
        dropout=cfg.dropout,
        max_seq_len=cfg.max_seq_len
    )

    ckpt = torch.load(cfg.pretrained_ckpt, map_location="cpu", weights_only=False)
    ckpt_filtered = dict(ckpt)

    # boyut uyusmuyor, sıfırdan egitecegiz.
    del ckpt_filtered['visual_proj.weight']
    del ckpt_filtered['visual_proj.bias']

    result = model.load_state_dict(ckpt_filtered, strict=False)
    print(f"missing: {result.missing_keys}, unexpected: {result.unexpected_keys}")
    return model
