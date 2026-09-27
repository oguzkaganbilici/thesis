import torch
from models.model import TripleSumm

# checkpoint key sayısı:
sd = torch.load("./checkpoint/best_model_ckpt_mrhisum.pth", map_location="cpu")
model = TripleSumm()
model.load_state_dict(sd, strict=False)
model.eval()
v = torch.randn(2, 12, 1024); t = torch.randn(2, 12, 768); a = torch.randn(2, 12, 768)
with torch.no_grad():
    out, attn = model(v, t, a)
print(out.shape, out.min().item(), out.max().item())   # [2,12], 0-1 arası