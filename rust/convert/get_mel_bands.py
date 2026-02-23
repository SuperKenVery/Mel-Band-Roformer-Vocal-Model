import torch
from librosa import filters

sample_rate = 44100
n_fft = 2048
num_bands = 60
audio_channels = 2

mel_filter_bank_numpy = filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=num_bands)
mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy)
mel_filter_bank[0][0] = 1.0
mel_filter_bank[-1, -1] = 1.0

freqs_per_band = mel_filter_bank > 0
num_freqs_per_band = freqs_per_band.sum(dim=1)

freqs_per_bands_with_complex = [2 * int(f) * audio_channels for f in num_freqs_per_band.tolist()]

print("FREQS_PER_BANDS_WITH_COMPLEX:", freqs_per_bands_with_complex)
print("TOTAL_BANDS:", len(freqs_per_bands_with_complex))
print("SUM:", sum(freqs_per_bands_with_complex))
