import torch
from models.model import TripleSumm   # senin models'in

saved = torch.load("orig_result.pt")
v, t, a = saved['input']
orig_out = saved['output']

sd = torch.load("./checkpoint/best_model_ckpt_mrhisum.pth", map_location="cpu")
my = TripleSumm()   # senin default parametrelerin
my.load_state_dict(sd, strict=False)
my.eval()

with torch.no_grad():
    my_out, _ = my(v, t, a)

print("aynı mı:", torch.allclose(my_out, orig_out, atol=1e-5))
print("max fark:", (my_out - orig_out).abs().max().item())
print("orijinal:", orig_out[0,:3])
print("seninki :", my_out[0,:3])