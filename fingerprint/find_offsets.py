from collections import Counter

def find_offsett(hash_table: dict, hl_pairs: list):

    offset_list = []
    matches = [] # visualize için

    for (hash, ozet_zamani) in hl_pairs:
        if hash in hash_table:
            for full_zaman in hash_table[hash]:
                offset = full_zaman - ozet_zamani
                offset_list.append(offset)
                matches.append((ozet_zamani, full_zaman))

    return offset_list, matches


"""
yukarıda ki methodla, full maç görüntülerinden oluşturulan hash tablosunda eşleşen
özet görüntülerinin hash değerlerini yakaladık. Her eşleşme frame düzeyinde 
offsett = full_zaman - özet_zaman üretti. 

Fakat bu eşleşmelerin bir çoğu sahte eşleşme ve oldukça gürültülü bir veri elde ettik.
çünkü aynı hash değerleri tam maç videosunda birden fazla yerde eşleşti. (collision)
bu sahteler rastgele offset'ler ürettiği için ortaya gürültülü bir dağılım çıktı.

Bu sahte eşlemelerden kurtulmak için RANSAC kullandık. Temel işi şu: 
RANSAC, gürültünün baskın olduğu veride "uzlaşan azınlığı" bulup gürültüyü eleyen, model-uydurma yöntemidir.
"""

"""
# HISTOGRAM EŞİK FİLTRESİ - baska bir method. calısıyor fakat dayanıklı değil.
def offset_filter(offset, matches):
    offsett_sayilari = Counter(offset)
    guclu_offsetts = {off for off, sayi in offsett_sayilari.items() if sayi > 300}

    real_matches = [
        (t_hl, t_full) for (t_hl, t_full) in matches if (t_full - t_hl) in guclu_offsetts
    ]

    return guclu_offsetts, real_matches

"""

