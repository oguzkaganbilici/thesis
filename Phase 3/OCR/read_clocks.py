
from image_crop import crop
from binarization import binarization
from regex import regex

def read_all_clocks(frames, x, y):
    seconds = []

    for scene in frames:
        start, end, frame_img = scene
        clock = crop(frame_img, x=x, y=y)
        binary, ret = binarization(clock)
        saniye = regex(binary=binary)

        seconds.append((start, end, saniye))    
        # print("saniye: ", saniye)

    return seconds