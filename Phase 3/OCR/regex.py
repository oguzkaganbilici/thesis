import re
import pytesseract

def regex(binary, pattern=r"(\d{1,2})[:.](\d{2})"):

    config = "--oem 3 --psm 7"

    tesseract = pytesseract.image_to_string(
        binary,
        config=config
    ).strip()


    sonuc = re.search(pattern, tesseract)

    if sonuc is None:
        return None
    
    dakika = sonuc.group(1)
    saniye = sonuc.group(2)

    total_saniye = int(dakika) * 60 + int(saniye)

    return total_saniye