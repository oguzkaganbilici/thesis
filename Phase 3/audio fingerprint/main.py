from audios import download_videos
from audio_to_peaks import audio2peaks
from fingerprints import fingerprints,create_hash_table
from find_offsets import find_offsett
from visualize import visualize
from ransac import sequential_ransac
from segments_to_label import segments2label
import numpy as np

np.random.seed(100)

DOWNLOADS = [
    {
        "url": "https://www.youtube.com/watch?v=oic1W5ZriQE",
        "outputname": "full_match"
    },
    {
        "url": "https://www.youtube.com/watch?v=LnKrnoMjqVw",
        "outputname": "highlights"
    }
]
# download_videos(DOWNLOADS)

fm_peaks, fullMatch_length = audio2peaks("full_match.wav", n_fft=512, 
                       hop_length=128, size = 40, th= -25)

hl_peaks, _ = audio2peaks("highlights.wav", n_fft=512, 
                       hop_length=128, size = 40, th= -25)

fm_pairs = fingerprints(fm_peaks, 1, 100, 10)
hl_pairs = fingerprints(hl_peaks, 1, 100, 10)

fm_hash_table = create_hash_table(fm_pairs)

offsets, matches = find_offsett(fm_hash_table, hl_pairs)

segments = sequential_ransac(matches=matches)


labels, araliklar = segments2label(segments, fullMatch_length)

visualize(best_inliers=labels)

