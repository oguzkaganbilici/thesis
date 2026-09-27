import re
import pytesseract

def regex(binary, pattern=r"(\d{1,2})[:.](\d{2})"):

    # -- psm 7 bazı clockları kacırdı, 8 daha iyi.
    config = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:"

    tesseract = pytesseract.image_to_string(
        binary,
        config=config
    ).strip()

    print(f"HAM TESSERACT: '{tesseract}'")  # DEBUG

    sonuc = re.search(pattern, tesseract)

    if sonuc is None:
        return None
    
    dakika = sonuc.group(1)
    saniye = sonuc.group(2)

    total_saniye = int(dakika) * 60 + int(saniye)

    return total_saniye


