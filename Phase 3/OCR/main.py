from scene_detect import get_scene_frames
from read_clocks import read_all_clocks
from grouping import group_positions
from labelling import generate_labels
import numpy as np

VIDEO_PATH = "/Users/oguzkaganbilici/Desktop/OCR/liverpool-real-madrid-hl.mp4"

X = [0.06, 0.13] # hesaplanmıs bir oran
Y = [0.05, 0.11] # hesaplanmıs bir oran

frames  = get_scene_frames(VIDEO_PATH)
clocks = read_all_clocks(frames, X, Y)
labels = generate_labels(clocks, offsets=260, path="liverpool-real-madrid.mp4")

