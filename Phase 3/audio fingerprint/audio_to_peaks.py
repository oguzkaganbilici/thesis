from audios import download_videos
import librosa
import numpy as np
import scipy 


def audio2peaks(path: str, n_fft=512, hop_length=128, size=20, th=-40):

    # waveform, sample rate
    y, sr = librosa.load(path)

    # STFT
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # magnitude
    amplitude = np.abs(stft)

    # amplitude -> db
    db = librosa.amplitude_to_db(amplitude, ref=np.max(amplitude))

    # maximum filter
    constellation_matrix = scipy.ndimage.maximum_filter(db, size=size)

    # peaks
    mask = (db == constellation_matrix) & (db > th)
    peaks = np.argwhere(mask)

    sorted_peaks = np.argsort(peaks[:, 1]) # zamana göre sıralı [frekans, zaman]
    peaks = peaks[sorted_peaks]

    return peaks, stft.shape[1]

