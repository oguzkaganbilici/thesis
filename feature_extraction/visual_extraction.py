import sys, os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import models, transforms
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import maclar, AKTIF_MAC, VISUAL_FEAT_PATH
import h5py

full_match_path = maclar[AKTIF_MAC]["full"]
npy_path = f"feats_visual_{AKTIF_MAC}.npy"

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    ]
)

def load_encoder():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = models.inception_v3(
        weights=models.Inception_V3_Weights.IMAGENET1K_V1,
        aux_logits = True
    )
    
    # weights = eğitilmiş halini indiriyoruz. boş bırakırsak random ağırlıklar geliyor
    # aux_logits = bunu true yapmalıyız yoksa hata verecek. biraz garip bi durum. sonradan bakarız.

    model.fc = nn.Identity()

    """
    fc -> fully connected

    InceptionV3 normalde en sonda bir fc (fully-connected) katmanıyla 2048-d temsili 1000 ImageNet sınıfına indirger. 
    Biz sınıf tahminini istemiyoruz — bir önceki durağı, o 2048-d ham temsili (GAP çıktısı) istiyoruz; parmak izi orada.
    fc'yi Identity ("hiçbir şey yapma, girdiyi aynen geçir") ile değiştirince, model artık 1000 yerine 2048 döndürür.
    Sınıflandırma kafasını söküp temsili açığa çıkarıyoruz. 
    GAP (Global Average Pooling) fc'den hemen önce olduğu için, fc'yi kaldırınca elimizde kalan tam olarak o [N,2048] GAP vektörü.
    """

    model.eval() # train modundan çıkartalım
    model = model.to(device)
    return model, device

def frame_to_feature(frame, model, device, transform):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resize = cv2.resize(rgb, (299, 299))
    image = transform(resize)
    image = image.unsqueeze(0)

    image = image.to(device)
  

    with torch.no_grad():
        out = model(image)

    return out.cpu()

def extract_visual(video_path, model, device, transform):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
    N = int(total_frame / fps)

    outs = []

    for i in range(int(total_frame)):
        # if i % 10000 == 0: print(i)
        ret, frame = video.read()
        if not ret: break
        if i % int(fps) == 0:
            feature = frame_to_feature(frame, model, device, transform)

            outs.append(feature)

        if len(outs) == N: break

    video.release()

    out = torch.cat(outs, dim=0)
    out = out.numpy()
    
    return out

if __name__ == "__main__":
    model, device = load_encoder()
    if os.path.exists(npy_path):
        feats = np.load(npy_path)
    else:
        feats = extract_visual(full_match_path, model, device, transform)
        np.save(npy_path, feats)


    with h5py.File(VISUAL_FEAT_PATH, "a") as f:
        if AKTIF_MAC in f: del f[AKTIF_MAC]
        group = f.create_group(AKTIF_MAC)
        group.create_dataset("features", data=feats)
        print("writing is done!")

    with h5py.File(VISUAL_FEAT_PATH, "r") as f:
        print(f[AKTIF_MAC]["features"].shape)
        print(list(f.keys()))






    
