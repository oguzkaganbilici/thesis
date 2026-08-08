import torch
from models.model import TripleSumm   # orijinalin kendi models'i

sd = torch.load("../checkpoint/best_model_ckpt_mrhisum.pth", map_location="cpu")

# parametreleri config'ten al — mrhisum.yaml'a bakmamız lazım
model = TripleSumm(
    visual_dim=1024, text_dim=768, audio_dim=768, input_dim=128, hidden_dim=192,
    num_model_layers=2, num_mst_layers=2, num_cmf_layers=2,
    num_heads=4, dropout=0.1, window_size=[5,15,45,0], max_seq_len=10000,
    get_attn_weights=False
)
model.load_state_dict(sd, strict=False)
model.eval()

torch.manual_seed(0)
v = torch.randn(2, 12, 1024); t = torch.randn(2, 12, 768); a = torch.randn(2, 12, 768)
with torch.no_grad():
    out, _ = model(v, t, a)

torch.save({'input': (v,t,a), 'output': out}, "../orig_result.pt")
print("orijinal çıktı kaydedildi:", out.shape, out[0,:3])