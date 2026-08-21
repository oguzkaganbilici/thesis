from image_crop import crop
from binarization import binarization
from regex import regex

X = [0.06, 0.13] # hesaplanmıs bir oran
Y = [0.05, 0.11] # hesaplanmıs bir oran


for i in range(0, 9):
    image_path = f"frames/full/fp_seg{i}.png"


    clock = crop(image_path, x=X, y=Y)

    binary, ret = binarization(clock)

    saniye = regex(binary=binary)
    print(f"{i}. png - saniye: ", saniye)