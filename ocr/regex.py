import re
from paddleocr import PaddleOCR

ocr_engine = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation = False, 
    lang='en')


def regex(crop, pattern=r"(\d{1,2})[:.](\d{2})"):

    result = ocr_engine.predict(crop)
    rec_text = result[0]["rec_texts"]
    # print(f"HAM: {rec_text}") # debug
    if not rec_text:
        return None
    
    sonuc = re.search(pattern, rec_text[0])

    if sonuc is None:
        return None
    
    dakika = sonuc.group(1)
    saniye = sonuc.group(2)

    total_saniye = int(dakika) * 60 + int(saniye)

    return total_saniye


