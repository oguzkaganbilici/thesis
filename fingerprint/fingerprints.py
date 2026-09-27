import numpy as np
def fingerprints(peaks, t_min, t_max, fan_out):
    pairs = []
    times = peaks[:, 1]

    for anchor in peaks:
        t_anchor = anchor[1]
        f1 = anchor[0]

        start = np.searchsorted(times, t_anchor + t_min, side="left")
        end = np.searchsorted(times,t_anchor + t_max, side="right")
        target_zone = peaks[start:end]

        target_zone = target_zone[:fan_out] # ilk fan_out kadar peaklerle eşleşsin.

        for target in target_zone:
            f2 = target[0]
            del_t = target[1] - t_anchor

            hash_value = (((f1 << 15) | (f2 << 6) | del_t))
            pairs.append((hash_value, t_anchor))

    return pairs

def create_hash_table(pairs: list):
    hash_table = {}

    for (hash, zaman) in pairs:
        hash_table.setdefault(hash, []).append(zaman)

    return hash_table

