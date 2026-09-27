maclar = {
    "liverpool-realmadrid":
    {
        "full": "/Users/oguzkaganbilici/Desktop/pipelines/videos/liverpool-real-madrid.mp4",
        "ozet": "/Users/oguzkaganbilici/Desktop/pipelines/videos/liverpool-real-madrid-hl.mp4",
        "roi_x": [0.06, 0.13],
        "roi_y": [0.05, 0.11],
    },

    "fcsb-fenerbahce":
    {
        "full": "/Users/oguzkaganbilici/Desktop/pipelines/videos/fcsb-fenerbahce-full.mp4",
        "ozet": "/Users/oguzkaganbilici/Desktop/pipelines/videos/fcsb-fenerbahce-hl.mp4",
        "roi_x": [0.06, 0.13],
        "roi_y": [0.05, 0.11],
    },

    "galatasaray-juventus":
    {
        "full": "/Users/oguzkaganbilici/Desktop/pipelines/videos/galatasaray-juventus-full.mp4",
        "ozet": "/Users/oguzkaganbilici/Desktop/pipelines/videos/galatasaray-juventus-hl.mp4",
        "roi_x": [0.06, 0.13],
        "roi_y": [0.05, 0.11],
    },

    "shkendija-samsunspor":
    {
        "full": "/Users/oguzkaganbilici/Desktop/pipelines/videos/shkendija-samsunspor-full.mp4",
        "ozet": "/Users/oguzkaganbilici/Desktop/pipelines/videos/shkendija-samsunspor-hl.mp4",
        "roi_x": [0.06, 0.13],
        "roi_y": [0.05, 0.11],
    },

}


AKTIF_MAC = "shkendija-samsunspor"

SEGMENTS_PATH = "/Users/oguzkaganbilici/Desktop/pipelines/segments.json"

FINGERPRINT_LABEL_PATH = "/Users/oguzkaganbilici/Desktop/pipelines/fingerprint_label.npy"
OCR_LABEL_PATH = "/Users/oguzkaganbilici/Desktop/pipelines/ocr_label.npy"

VISUAL_FEAT_PATH = "/Users/oguzkaganbilici/Desktop/pipelines/trt/trt_feat_visual_inceptionv3.h5"
AUDIO_FEAT_PATH  = "/Users/oguzkaganbilici/Desktop/pipelines/trt/trt_feat_audio_ast.h5"
TEXT_FEAT_PATH   = "/Users/oguzkaganbilici/Desktop/pipelines/trt/trt_feat_text_berturk.h5"

GT_PATH = "/Users/oguzkaganbilici/Desktop/pipelines/trt/trt_gt.h5"

SPLIT_PATH = "/Users/oguzkaganbilici/Desktop/pipelines/trt/trt_split.json"
