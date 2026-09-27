import numpy as np
from scipy.ndimage import gaussian_filter1d

def segments2label(segments, fullMatch_length):

    araliklar = []

    fullMatch_length_sn = int(fullMatch_length * 128/22050)
    labels = np.zeros(fullMatch_length_sn) # 1-0 dolduracagiz


    for seg in segments:
        fullMatch_start = int(seg[:, 1].min() * 128 / 22050) # np.float dönüyor, sıkıntı yaratabilir.
        fullMatch_end = int(seg[:, 1].max() * 128 / 22050) # frame -> saniye

        labels[fullMatch_start: fullMatch_end] = 1

        
        araliklar.append((fullMatch_start, fullMatch_end))

    print("Fingerprint Etiket boyutu: ", labels.shape)

    return labels, araliklar



