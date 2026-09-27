from scene_detect import get_scene_frames
from read_clocks import read_all_clocks
from grouping import group_positions
from labelling import generate_labels
from visualize import visualize_labels
from offset_calc import offset_calc
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import maclar, AKTIF_MAC, SEGMENTS_PATH, OCR_LABEL_PATH

match = maclar[AKTIF_MAC]
FULL_PATH = match["full"]
HL_PATH = match["ozet"]

X = match["roi_x"]
Y = match["roi_y"]
print(f"-------OCR İŞLENEN MAÇ: {AKTIF_MAC}------")

frames  = get_scene_frames(HL_PATH)
clocks = read_all_clocks(frames, X, Y)


"""
# debug
for scene in clocks:
    start, end, clock = scene
    ozet_start = start / 25   # özet fps
    ozet_end = end / 25
    print(f"{ozet_start:.0f}s - {ozet_end:.0f}s : {clock}")
"""


offset = offset_calc(fullMatch_path=FULL_PATH, segments_path=SEGMENTS_PATH, x=X, y=Y)
print("offset: ", offset)
labels = generate_labels(clocks, offsets=offset, 
                         path=FULL_PATH)

print("OCR etiket boyutu: ", labels.shape)

np.save(OCR_LABEL_PATH, labels)

np.set_printoptions(threshold=np.inf)

print(np.where(labels == 1)[0])

visualize_labels(labels)
