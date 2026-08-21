import numpy
import cv2

def crop(img_path, x: list, y: list):
    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError(f"Görüntü bulunamadı: {img_path}")

    H, W = img.shape[:2]

    x1 = int(W * x[0])
    x2 = int(W * x[1])

    y1 = int(H * y[0])
    y2 = int(H * y[1])

    clock = img[y1:y2, x1:x2]

    cv2.imwrite("clock_crope.png", clock)
    return clock