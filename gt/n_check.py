import cv2
# import librosa
import numpy as np
from config import maclar, AKTIF_MAC, OCR_LABEL_PATH, FINGERPRINT_LABEL_PATH

video_path = maclar[AKTIF_MAC]["full"]
print(f"------- N-CHECK: {AKTIF_MAC} -------")

video = cv2.VideoCapture(video_path)

fps = video.get(cv2.CAP_PROP_FPS)
total_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)

video.release()

duration_video = total_frame / fps
N_v = int(duration_video)

print("fps: ", fps) # 25.0
print("total_frame: ", total_frame) # 147943.0
print("duration video: ", duration_video) # 5917.72
print("N_v: ", N_v) # 5917

# N_v == N_gt olacak
"""
# audio için check

waveform, sample_rate = librosa.load(video_path, sr=22050, mono=True)

duration_audio = len(waveform) / sample_rate

print("len(v): ", len(waveform))
print("duration_audio: ", duration_audio)
print("int(duration_audio): ", int(duration_audio))

"""
ocr = np.load(OCR_LABEL_PATH)
fp = np.load(FINGERPRINT_LABEL_PATH)
print("ocr shape: ", ocr.shape[0])
print("fp shape: ", fp.shape[0])
print("N_v: ", N_v)

