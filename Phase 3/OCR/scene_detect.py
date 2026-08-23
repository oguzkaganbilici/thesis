import cv2
from scenedetect import detect, AdaptiveDetector


def get_scene_frames(video_path):

    scene_list = detect(video_path, AdaptiveDetector(adaptive_threshold=3.0, min_scene_len=15))
    print(f"toplam sahne sayisi {len(scene_list)}")

    frames = []
    crop_frames = []
    for i, scene in enumerate(scene_list):
        start, end = scene
        start_frame = start.get_frames()
        end_frame = end.get_frames()

        orta_frame = (start_frame + end_frame) // 2

        frames.append((start_frame, end_frame, orta_frame))
        

  
    video = cv2.VideoCapture("liverpool-real-madrid-hl.mp4")
  
    for frame_no in frames:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_no[2])

        ret, frame = video.read()

        if not ret:
            # print(f"Frame {frame_no} okunamadı")
            continue

        # print(f"Frame {frame_no} okundu. Shape:", frame.shape)

        crop_frames.append((frame_no[0], frame_no[1], frame))

    return crop_frames





 


