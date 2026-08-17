import numpy as np
from scipy.ndimage import gaussian_filter1d

def segments2label(segments, fullMatch_length):

    araliklar = []
    labels = np.zeros(fullMatch_length) # 1-0 dolduracagiz
    smooth_labels = None

    for seg in segments:
        fullMatch_start = int(seg[:, 1].min()) # np.float dönüyor, sıkıntı yaratabilir.
        fullMatch_end = int(seg[:, 1].max())

        labels[fullMatch_start: fullMatch_end] = 1

        
        araliklar.append((fullMatch_start, fullMatch_end))

    smooth_labels = gaussian_filter1d(labels, sigma=300)

    return smooth_labels, araliklar



