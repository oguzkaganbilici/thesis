import cv2

def binarization(img, upscale_x=4, upscale_y=4):

    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # upscale
    upscaled = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # threshold
    ret, binary = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imwrite("clock_binary.png", binary)

    return binary, ret