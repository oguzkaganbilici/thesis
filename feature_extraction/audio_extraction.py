import sys, os
import torch
import numpy as np
import librosa
import cv2
import h5py
from transformers import ASTFeatureExtractor, ASTModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import maclar, AKTIF_MAC, AUDIO_FEAT_PATH

FULL_MATCH_PATH = maclar[AKTIF_MAC]["full"]
NPY_PATH = f"feats_audio_{AKTIF_MAC}.npy"

def load_encoder():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device: ", device)

    extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    model.eval() # BatchNorm/dropout dondur,
    model.to(device)

    return device, extractor, model

def extract_audio(video_path, extractor, model, device):
    # AST 16kHz şart; sr değişmemeli
    y, sample_rate = librosa.load(video_path, sr=16_000, mono=True)
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
    N = int(total_frame / fps)

    cls_list = []
    for sec in range(N):
        if sec % 500 == 0: print(sec)
        window_start = sec * sample_rate - (5 * sample_rate) # 80.000 = 5 saniye * 16.000 örnek
        window_end = sec * sample_rate + (5 * sample_rate)

        safe_start = max(0, window_start)
        safe_end = min(len(y), window_end)

        window = y[safe_start: safe_end]


        left = max(0, -window_start)
        right = max(0, window_end - len(y))

        window = np.pad(window, (left, right))

        features = extractor(window, sampling_rate = sample_rate, return_tensors = "pt")

        input_values = features["input_values"].to(device)

        with torch.no_grad():
            out = model(input_values)
            cls = out.last_hidden_state[:, 0, :].cpu()
            cls_list.append(cls)

    video.release()

    cls_ = torch.cat(cls_list, dim=0).numpy()
    print(cls_.shape)
    return cls_


if __name__ == "__main__":
    
    device, extractor, model = load_encoder()

    if os.path.exists(NPY_PATH):
        feats = np.load(NPY_PATH)
    else:
        feats = extract_audio(FULL_MATCH_PATH, extractor, model, device).astype(np.float32)
        np.save(NPY_PATH, feats)


    with h5py.File(AUDIO_FEAT_PATH, "a") as f:
        if AKTIF_MAC in f: del f[AKTIF_MAC]
        group = f.create_group(AKTIF_MAC)
        group.create_dataset("features", data=feats)
        print("writing is done!")

    with h5py.File(AUDIO_FEAT_PATH, "r") as f:
        print(f[AKTIF_MAC]["features"].shape)
        print(list(f.keys()))
        
    """
    print("shape: ", feats.shape)
    print("np.isnan(feats).any(): ", np.isnan(feats).any())
    print("feats.min(): ", feats.min())
    print("feats.max(): ", feats.max())
    print("feats.dtype: ", feats.dtype)

    yakin = np.linalg.norm(feats[3000] - feats[3001])
    uzak = np.linalg.norm(feats[3000] - feats[3500])

    print("yakin < uzak: ", yakin < uzak)

    print("feats[0]: ", feats[0])
    print("feats[N-1]: ", feats[-1])
"""