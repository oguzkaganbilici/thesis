
from image_crop import crop
from binarization import binarization
from regex import regex
import cv2
def read_all_clocks(frames, x, y):
    seconds = []

    for scene in frames:
        start, end, frame_img = scene
        clock = crop(frame_img, x=x, y=y)
        # binary, ret = binarization(clock)
        saniye = regex(crop=clock)


        if saniye is None:
            cv2.imwrite(f"debug/debug_none_{start}.png", clock)
            # cv2.imwrite(f"debug/none_{start}_binary.png", binary)

        
        seconds.append((start, end, saniye))    
        # print("saniye: ", saniye)

    return seconds