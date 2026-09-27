import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

from config import FINGERPRINT_LABEL_PATH, OCR_LABEL_PATH, AKTIF_MAC


fp_labels = np.load(FINGERPRINT_LABEL_PATH)
ocr_labels = np.load(OCR_LABEL_PATH)

print("OCR tek başına 1 sayısı:", ocr_labels.sum())
print("Fingerprint tek başına 1 sayısı:", fp_labels.sum())

N = len(ocr_labels) # N_v
d = abs(len(fp_labels) - N)
if d >= 5:
    raise ValueError("aradaki fark çok büyük")

if len(fp_labels) < N:
    print("padding: ", N - len(fp_labels))
    fp_labels = np.pad(fp_labels, (0, N - len(fp_labels)))
    

elif len(fp_labels) > N: 
    print("cutting: ", len(fp_labels) - N)
    fp_labels = fp_labels[:N]
    

assert fp_labels.shape == ocr_labels.shape, "Boyutlar uyusmuyor"

final_labels = np.logical_or(fp_labels, ocr_labels).astype(int)

print("Union 1 sayısı:", final_labels.sum())
# print(np.where(final_labels==1)[0])

smooth_final = gaussian_filter1d(final_labels.astype(float), sigma=4)

print("min:", smooth_final.min())      # ~0 olmalı
print("max:", smooth_final.max())      # tepe değeri, 1'e yakın mı?
print("mean:", smooth_final.mean())    # ortalama, çoğu 0 civarı olmalı (seyrek önemli an)
print("0'dan büyük kaç nokta:", (smooth_final > 0.01).sum())  # kaç nokta anlamlı
plt.plot(smooth_final)
plt.savefig("smooth_test.png")

gt_score_path   = f"/Users/oguzkaganbilici/Desktop/pipelines/gt_score_{AKTIF_MAC}.npy"
gt_summary_path = f"/Users/oguzkaganbilici/Desktop/pipelines/gt_summary_{AKTIF_MAC}.npy"

np.save(gt_score_path, smooth_final)          # gaussian [0,1] → modelin eğitim hedefi
np.save(gt_summary_path, final_labels)        # binary 0/1 → değerlendirme etiketi

print("kaydedildi:", gt_score_path)
print("kaydedildi:", gt_summary_path)
