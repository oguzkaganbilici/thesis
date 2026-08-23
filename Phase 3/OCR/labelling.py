import cv2
import numpy as np

def generate_labels(clocks, path, offsets=260):

    video = cv2.VideoCapture(path)
    FPS = video.get(cv2.CAP_PROP_FPS)
    toplam_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
    N = int(toplam_frame / FPS)

    arr = np.zeros(N)

    for clc in clocks:
        clock = clc[2]
        start_frame = clc[0]
        end_frame = clc[1]

        if clock is None:
            continue

        if clock >= 2700:
            clock += offsets

        sure_sn = (end_frame - start_frame) / FPS

        baslangic = int(clock)
        bitis = int(clock + sure_sn)
        
        arr[baslangic: bitis] = 1

    print(np.where(arr == 1)[0])

    

    return arr


        





