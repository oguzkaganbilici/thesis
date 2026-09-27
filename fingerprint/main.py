from audio_to_peaks import audio2peaks
from fingerprints import fingerprints,create_hash_table
from find_offsets import find_offsett
from visualize import visualize
from ransac import sequential_ransac
from segments_to_label import segments2label
import numpy as np
import json


import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import maclar, AKTIF_MAC, SEGMENTS_PATH, FINGERPRINT_LABEL_PATH

np.random.seed(100)

match= maclar[AKTIF_MAC]
FULL_PATH = match["full"]
HL_PATH = match["ozet"]

print(f"-------FINGERPRINT İŞLENEN MAÇ: {AKTIF_MAC}------")

fm_peaks, fullMatch_length = audio2peaks(FULL_PATH, n_fft=512, 
                       hop_length=128, size = 40, th= -25)

hl_peaks, _ = audio2peaks(HL_PATH, n_fft=512, 
                       hop_length=128, size = 40, th= -25)

fm_pairs = fingerprints(fm_peaks, 1, 100, 10)
hl_pairs = fingerprints(hl_peaks, 1, 100, 10)

fm_hash_table = create_hash_table(fm_pairs)

offsets, matches = find_offsett(fm_hash_table, hl_pairs)

segments = sequential_ransac(matches=matches)
print("len segments: ", len(segments))

segment_list = []
for i, seg in enumerate(segments):
    baslangic_frame = seg[:, 1].min()
    saniye = baslangic_frame * 128 / 22050
    segment_list.append(
        {
            "video_sn": float(saniye),
            "frame": int(baslangic_frame)
        }
    ) # json için

    dakika = int(saniye // 60)
    kalan_saniye = int(saniye % 60)
    print(f"Segment {i}: {dakika}:{kalan_saniye:02d}  (frame {int(baslangic_frame)}, {saniye:.1f}s)")

with open(SEGMENTS_PATH, "w") as f: # ../segments.json bir üst dizin
    json.dump(segment_list, f, indent=2)


labels, araliklar = segments2label(segments, fullMatch_length)
np.save(FINGERPRINT_LABEL_PATH, labels)

visualize(best_inliers=labels)
