import json
import cv2
import numpy as np
from image_crop import crop
from regex import regex



IKINCI_YARI_MIN = 3000   # ilk yarı+devre arası kesin bitişi
UZATMA_MIN = 5600        # 90:00+offset civarı, uzatma başı


def offset_calc(fullMatch_path, x, y, segments_path="../segments.json"):
    with open(segments_path, "r") as file:
        data = json.load(file)

    second_half = []
    offset_list = [] 


    for i in data:
        video_sn = i["video_sn"]

        if IKINCI_YARI_MIN < video_sn and video_sn < UZATMA_MIN:
            second_half.append(i)


    video = cv2.VideoCapture(fullMatch_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    for i in second_half:
        video_sn = i["video_sn"]
        frame_no = int(fps*video_sn)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, img = video.read()
        cropped_img = crop(img, x, y)
        reg = regex(cropped_img)
        # print(f"video_sn={video_sn:.0f}, reg={reg}")   # debug
        if reg is not None:
            offset = video_sn - reg 
            if 0 < offset < 1250:
                offset_list.append(offset)
        else:
            continue


    print(offset_list)


    if not offset_list:
        raise ValueError("Hiç segment okunamadı, offset hesaplanamadı")
    real_ofsett = np.median(offset_list)

    video.release() # bellek bosalt.
    return real_ofsett