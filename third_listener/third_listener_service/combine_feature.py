from sound_feature import *


def generate_all_features_for_sound(wav_file_path):
    orig_wav_vector = load_wav_file(wav_file_path)
    sig_mean, sig_std = generate_sig(orig_wav_vector)
    rmse_mean, rmse_std = generate_rmse(orig_wav_vector)
    silence = extract_slience(orig_wav_vector)
    harmoic = extract_harmonic(orig_wav_vector)
    auto_corrs, auto_cor_max, auto_cor_std = generate_auto_correlation(orig_wav_vector)
    mfccs_mean, mfccs_std = generate_mfcc(orig_wav_vector)
    mels_mean, mels_std = generate_mels(orig_wav_vector)
    return np.array([sig_mean, sig_std, rmse_mean, rmse_std, silence, harmoic, auto_cor_max, auto_cor_std,*mfccs_mean, *mfccs_std,mels_mean, mels_std])


def generate_all_features_for_text(wav_file_path):
    pass

# def generate_all_features(wav_file_path):
#     sounds_features = generate_all_features_for_sound(wav_file_path)
#     return sounds_features
