import librosa
import numpy as np
import librosa.display

SR = 44100
N_MFCC = 13
N_MELS = 128


def load_wav_file(wav_file_path, sr=22050):
    audio_time_series, sample_rate = librosa.load(wav_file_path, sr=sr)
    orig_wav_vector = np.array(audio_time_series)
    return orig_wav_vector


def extract_sig_mean(wav_vector):
    return np.mean(np.abs(wav_vector))


def extract_sig_std(wav_vector):
    return np.mean(np.std(wav_vector))


def extract_rmse(wav_vector):
    return librosa.feature.rms(wav_vector)[0]
def extract_rmse_mean(rmse):
    return np.mean(rmse)


def extract_rmse_std(rmse):
    return np.std(rmse)


def extract_slience(wav_vector):
    rmse = extract_rmse(wav_vector)
    silence = np.sum(rmse <= 0.4 * np.mean(rmse)) / len(rmse)
    return silence


def extract_harmonic(wav_vector):
    harmonic = librosa.effects.hpss(wav_vector)[0]
    return np.mean(harmonic) * 1000


def extract_auto_cor(wav_vector):
    sig_mean = extract_sig_mean(wav_vector)
    cl = 0.45 * sig_mean
    center_clipped = np.clip(wav_vector - cl, -cl, cl)
    center_clipped[np.abs(center_clipped) < cl] = 0
    auto_corrs = librosa.core.autocorrelate(center_clipped)
    return auto_corrs


def extract_auto_cor_max(wav_vector):
    auto_corrs = extract_auto_cor(wav_vector)
    return 1000 * np.max(auto_corrs) / len(auto_corrs)


def extract_auto_cor_std(wav_vector):
    auto_corrs = extract_auto_cor(wav_vector)
    return np.std(auto_corrs)


def extract_mfcc(wav_vector):
    mfccs = librosa.feature.mfcc(y=wav_vector, sr=SR, n_mfcc=N_MFCC)
    return mfccs


def extract_mfcc_mean(mfccs):
    mfccs_mean_list = []
    for i in range(N_MFCC):
        mfccs_mean_list.append(np.mean(mfccs[i]))
    return mfccs_mean_list


def extract_mfcc_std(mfccs):
    mfccs_sta_list = []
    for i in range(N_MFCC):
        mfccs_sta_list.append(np.std(mfccs[i]))
    return mfccs_sta_list


def extract_mel(wav_vector):
    mel_spectrogram = librosa.feature.melspectrogram(y=wav_vector, sr=SR, n_mels=N_MELS)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db


def extract_mel_mean(mel_spectrogram_db):
    return np.mean(mel_spectrogram_db)


def extract_mel_std(mel_spectrogram_db):
    return np.std(mel_spectrogram_db)


def generate_sig(wav_vector):
    sig_mean = extract_sig_mean(wav_vector)
    sig_std = extract_sig_std(wav_vector)
    return sig_mean,sig_std

def generate_mfcc(wav_vector):
    mfccs = extract_mfcc(wav_vector)
    mfccs_mean = extract_mfcc_mean(mfccs)
    mfccs_std = extract_mfcc_std(mfccs)
    return mfccs_mean, mfccs_std

def generate_mels(wav_vector):
    mels = extract_mel(wav_vector)
    mels_mean = extract_mel_mean(mels)
    mels_std = extract_mel_std(mels)
    return mels_mean, mels_std

def generate_rmse(wav_vector):
    rmse = extract_rmse(wav_vector)
    rmse_mean = extract_rmse_mean(rmse)
    rmse_std = extract_rmse_mean(rmse)
    return rmse_std,rmse_mean

def generate_auto_correlation(wav_vector):
    auto_corrs = extract_auto_cor(wav_vector)
    auto_cor_max = extract_auto_cor_max(wav_vector)
    auto_cor_std = extract_auto_cor_std(wav_vector)
    return auto_corrs,auto_cor_max,auto_cor_std




