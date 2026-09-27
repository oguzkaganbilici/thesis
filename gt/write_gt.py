import numpy as np
import h5py

from config import maclar, VISUAL_FEAT_PATH, GT_PATH

veri = {}
with h5py.File(VISUAL_FEAT_PATH, "r") as f:
    for mac in maclar:
        visual_N  = f[mac]["features"].shape[0]

        gt_score = np.load(f"gt_score_{mac}.npy")
        gt_summary = np.load(f"gt_summary_{mac}.npy")

        assert gt_score.shape[0] == visual_N, f"{mac}, gt_score ile visual_N aynı boyutta degil"
        assert gt_summary.shape[0] == visual_N, f"{mac}, gt_summary ile visual_N aynı boyutta degil"

        veri[mac] = (gt_score, gt_summary, visual_N)


print("d1 kontrolü GEÇTİ — hepsi visual ile birebir")

with h5py.File(GT_PATH, "w") as f:
    for mac, (gt_score, gt_summary, visual_N) in veri.items():
        group = f.create_group(mac)
        group.create_dataset("gt_score", data=gt_score.astype(np.float32))
        group.create_dataset("gt_summary", data=gt_summary.astype(np.int64))
        group.create_dataset("change_points", data=np.array([[0, visual_N - 1]], dtype=np.int64))


