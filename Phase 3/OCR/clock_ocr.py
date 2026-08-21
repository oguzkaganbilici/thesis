import cv2
import numpy as np
import pytesseract
import re

image = "frames/summary/sum_grid_15.png"

config = "--oem 3 --psm 7"

img = cv2.imread(image) # png'yi okur ve bir np array döndürür
print("shape of image:" ,img.shape) 
# (1080, 1920, 3) -> yükseklik(satır sayısı), genislik(sutun sayısı), renk


# height and weight
H, W = img.shape[:2]

# orandan -> piksele 
x1 = int(W * x[0])
x2 = int(W * x[1])

y1 = int(H * y[0])
y2 = int(H * y[1])

# kırpma , burda önce satırlar, sonra sütunlar
clock = img[y1:y2, x1:x2]

cv2.imwrite("clock_crope.png", clock)

print("clock shape: ", clock.shape)

# binarization

# 1.grayscale

gray = cv2.cvtColor(clock, cv2.COLOR_BGR2GRAY)

print("gray.shape: ", gray.shape)

# 2. upscale
upscaled = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

print("upscaled: ", upscaled.shape)

# threshold black or white - no gray

ret, binary = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print("otsu: ", ret)
cv2.imwrite("clock_binary.png", binary)

# tesseract

tesseract = pytesseract.image_to_string(binary, config=config).strip()

print(f"OCR sonucu: '{tesseract}' ")

# regex - kalıbımız bu olacak \d{1,2}:\d{2}
# /d --> herhangi bir rakam 0-9
# bir öncekinden(yani rakamdan) 1 veya 2 tane olacak
# : direkt bu gelecek
# /d{2} tam 2 rakam gelmesi bekleniyor

# 2 method var search ve match 
# match sadece stringin başında arar, search stringing her yerinde
pattern = r"\d{1,2}:\d{2}"
sonuc = re.search(pattern = pattern, string=tesseract).group()
print("sonuc: ", sonuc)
sn = sonuc.split(":")
saniye = int(sn[0]) * 60 + int(sn[1])

print("saniye: ", saniye)