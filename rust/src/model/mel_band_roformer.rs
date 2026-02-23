use super::band_split::BandSplit;
use super::config::ModelConfig;
use super::mask_estimator::MaskEstimator;
use super::rotary::RotaryEmbedding;
use super::transformer::Transformer;
use burn::module::{Ignored, Module};
use burn::prelude::*;
use burn::tensor::backend::Backend;
use crate::stft::{FftPlan, istft, stft};
use num_complex::Complex32;

#[derive(Clone, Debug)]
pub struct MelFilterBank {
    pub freq_indices: Vec<usize>,
    pub freqs_per_band: Vec<Vec<bool>>,
    pub num_freqs_per_band: Vec<usize>,
    pub num_bands_per_freq: Vec<usize>,
}

impl MelFilterBank {
    pub fn new(sample_rate: u32, n_fft: usize, num_bands: usize, stereo: bool) -> Self {
        let freq_bins = n_fft / 2 + 1;

        let mel_filter = compute_mel_filter_bank(sample_rate, n_fft, num_bands);

        let mut freqs_per_band: Vec<Vec<bool>> = vec![vec![false; freq_bins]; num_bands];
        for (band_idx, filter_row) in mel_filter.iter().enumerate() {
            for (freq_idx, &val) in filter_row.iter().enumerate() {
                freqs_per_band[band_idx][freq_idx] = val > 0.0;
            }
        }

        freqs_per_band[0][0] = true;
        if let Some(last) = freqs_per_band.last_mut() {
            if let Some(last_freq) = last.last_mut() {
                *last_freq = true;
            }
        }

        let mut freq_indices: Vec<usize> = Vec::new();
        for band_idx in 0..num_bands {
            for freq_idx in 0..freq_bins {
                if freqs_per_band[band_idx][freq_idx] {
                    if stereo {
                        freq_indices.push(freq_idx * 2);
                        freq_indices.push(freq_idx * 2 + 1);
                    } else {
                        freq_indices.push(freq_idx);
                    }
                }
            }
        }

        let num_freqs_per_band: Vec<usize> = freqs_per_band
            .iter()
            .map(|band| band.iter().filter(|&&b| b).count())
            .collect();

        let mut num_bands_per_freq = vec![0usize; freq_bins];
        for band in &freqs_per_band {
            for (freq_idx, &included) in band.iter().enumerate() {
                if included {
                    num_bands_per_freq[freq_idx] += 1;
                }
            }
        }

        Self {
            freq_indices,
            freqs_per_band,
            num_freqs_per_band,
            num_bands_per_freq,
        }
    }

    pub fn freqs_per_bands_with_complex(&self, audio_channels: usize) -> Vec<usize> {
        self.num_freqs_per_band
            .iter()
            .map(|&f| 2 * f * audio_channels)
            .collect()
    }
}

fn compute_mel_filter_bank(sample_rate: u32, n_fft: usize, num_mels: usize) -> Vec<Vec<f32>> {
    let sr = sample_rate as f64;
    let fmin = 0.0;
    let fmax = sr / 2.0;

    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
    const LOGSTEP: f64 = 0.06875177742094912;

    let hz_to_mel = |hz: f64| -> f64 {
        if hz < MIN_LOG_HZ {
            hz / F_SP
        } else {
            MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOGSTEP
        }
    };

    let mel_to_hz = |mel: f64| -> f64 {
        if mel < MIN_LOG_MEL {
            mel * F_SP
        } else {
            MIN_LOG_HZ * (LOGSTEP * (mel - MIN_LOG_MEL)).exp()
        }
    };

    let n_bins = n_fft / 2 + 1;
    let min_mel = hz_to_mel(fmin);
    let max_mel = hz_to_mel(fmax);

    let mel_f: Vec<f64> = (0..num_mels + 2)
        .map(|i| mel_to_hz(min_mel + (max_mel - min_mel) * i as f64 / (num_mels + 1) as f64))
        .collect();

    let fdiff: Vec<f64> = mel_f.windows(2).map(|w| w[1] - w[0]).collect();

    let fft_freqs: Vec<f64> = (0..n_bins)
        .map(|j| j as f64 * sr / n_fft as f64)
        .collect();

    let mut weights = vec![vec![0.0f32; n_bins]; num_mels];

    for i in 0..num_mels {
        let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);

        for (j, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - mel_f[i]) / fdiff[i];
            let upper = (mel_f[i + 2] - freq) / fdiff[i + 1];
            let w = lower.min(upper).max(0.0);
            weights[i][j] = (w * enorm) as f32;
        }
    }

    weights
}

#[derive(Module, Debug)]
pub struct MelBandRoformer<B: Backend> {
    band_split: BandSplit<B>,
    time_transformers: Vec<Transformer<B>>,
    freq_transformers: Vec<Transformer<B>>,
    mask_estimators: Vec<MaskEstimator<B>>,
    config: Ignored<ModelConfig>,
    mel_filter_bank: Ignored<MelFilterBank>,
    time_rotary: Ignored<RotaryEmbedding>,
    freq_rotary: Ignored<RotaryEmbedding>,
}

impl<B: Backend> MelBandRoformer<B> {
    pub fn new(device: &B::Device, config: ModelConfig) -> Self {
        let mel_filter_bank = MelFilterBank::new(
            config.sample_rate,
            config.stft_n_fft,
            config.num_bands,
            config.stereo,
        );

        let freqs_per_bands = mel_filter_bank.freqs_per_bands_with_complex(config.audio_channels());

        let band_split = BandSplit::new(device, config.dim, freqs_per_bands.clone());

        let time_transformers: Vec<Transformer<B>> = (0..config.depth)
            .map(|_| {
                Transformer::new(
                    device,
                    config.dim,
                    config.time_transformer_depth,
                    config.heads,
                    config.dim_head,
                    config.ff_mult,
                    config.attn_dropout,
                    config.ff_dropout,
                )
            })
            .collect();

        let freq_transformers: Vec<Transformer<B>> = (0..config.depth)
            .map(|_| {
                Transformer::new(
                    device,
                    config.dim,
                    config.freq_transformer_depth,
                    config.heads,
                    config.dim_head,
                    config.ff_mult,
                    config.attn_dropout,
                    config.ff_dropout,
                )
            })
            .collect();

        let mask_estimators: Vec<MaskEstimator<B>> = (0..config.num_stems)
            .map(|_| {
                MaskEstimator::new(
                    device,
                    config.dim,
                    freqs_per_bands.clone(),
                    config.mask_estimator_depth,
                )
            })
            .collect();

        let time_rotary = RotaryEmbedding::new(config.dim_head);
        let freq_rotary = RotaryEmbedding::new(config.dim_head);

        Self {
            band_split,
            time_transformers,
            freq_transformers,
            mask_estimators,
            config: Ignored(config),
            mel_filter_bank: Ignored(mel_filter_bank),
            time_rotary: Ignored(time_rotary),
            freq_rotary: Ignored(freq_rotary),
        }
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config.0
    }

    pub fn forward(&self, raw_audio: &[Vec<f32>]) -> Vec<Vec<Vec<f32>>> {
        let plan = FftPlan::new(
            self.config.0.stft_n_fft,
            self.config.0.stft_hop_length,
            self.config.0.stft_win_length,
        );

        let stft_repr = stft(raw_audio, &plan);

        let device = B::Device::default();
        let batch = 1;
        let num_channels = stft_repr.len();
        let freq_bins = stft_repr.get(0).map(|f| f.len()).unwrap_or(0);
        let time_frames = stft_repr
            .get(0)
            .and_then(|f| f.get(0))
            .map(|t| t.len())
            .unwrap_or(0);

        let total_freqs = freq_bins * num_channels;
        let mut stft_flat: Vec<f32> = Vec::with_capacity(time_frames * total_freqs * 2);

        for t in 0..time_frames {
            for f in 0..freq_bins {
                for ch in 0..num_channels {
                    stft_flat.push(stft_repr[ch][f][t].re);
                    stft_flat.push(stft_repr[ch][f][t].im);
                }
            }
        }

        let mut gathered: Vec<f32> = Vec::new();
        for t in 0..time_frames {
            for &freq_idx in &self.mel_filter_bank.0.freq_indices {
                let base = t * total_freqs * 2 + freq_idx * 2;
                gathered.push(stft_flat[base]);
                gathered.push(stft_flat[base + 1]);
            }
        }

        let gathered_len = self.mel_filter_bank.0.freq_indices.len() * 2;
        let x = Tensor::<B, 1>::from_floats(gathered.as_slice(), &device)
            .reshape([batch, time_frames, gathered_len]);

        let x = self.band_split.forward(x);

        let [_, time, num_bands, dim] = x.dims();
        let mut x = x;

        for (time_transformer, freq_transformer) in
            self.time_transformers.iter().zip(self.freq_transformers.iter())
        {
            let x_time = x.clone().swap_dims(1, 2);
            let x_time = x_time.reshape([batch * num_bands, time, dim]);
            let x_time = time_transformer.forward(x_time);
            let x_time = x_time.reshape([batch, num_bands, time, dim]);
            x = x_time.swap_dims(1, 2);

            let x_freq = x.clone().reshape([batch * time, num_bands, dim]);
            let x_freq = freq_transformer.forward(x_freq);
            x = x_freq.reshape([batch, time, num_bands, dim]);
        }

        let num_stems = self.mask_estimators.len();
        let mut all_masks: Vec<Tensor<B, 3>> = Vec::with_capacity(num_stems);

        for mask_estimator in &self.mask_estimators {
            let mask = mask_estimator.forward(x.clone());
            all_masks.push(mask);
        }

        let original_len = raw_audio.get(0).map(|c| c.len()).unwrap_or(0);

        let mut output_stems: Vec<Vec<Vec<f32>>> = Vec::with_capacity(num_stems);

        for stem_mask in &all_masks {
            let mask_data: Vec<f32> = stem_mask.clone().into_data().to_vec().unwrap();

            let mut masked_stft = stft_repr.clone();

            let mut scatter_sum_real = vec![vec![vec![0.0f32; time_frames]; freq_bins]; num_channels];
            let mut scatter_sum_imag = vec![vec![vec![0.0f32; time_frames]; freq_bins]; num_channels];

            let mut mask_idx = 0;
            for t in 0..time_frames {
                for &freq_idx in &self.mel_filter_bank.0.freq_indices {
                    let ch = freq_idx % num_channels;
                    let f = freq_idx / num_channels;

                    let mask_re = mask_data[t * gathered_len + mask_idx * 2];
                    let mask_im = mask_data[t * gathered_len + mask_idx * 2 + 1];

                    let stft_re = stft_repr[ch][f][t].re;
                    let stft_im = stft_repr[ch][f][t].im;

                    let out_re = stft_re * mask_re - stft_im * mask_im;
                    let out_im = stft_re * mask_im + stft_im * mask_re;

                    scatter_sum_real[ch][f][t] += out_re;
                    scatter_sum_imag[ch][f][t] += out_im;

                    mask_idx += 1;
                }
                mask_idx = 0;
            }

            for ch in 0..num_channels {
                for f in 0..freq_bins {
                    let denom = self.mel_filter_bank.0.num_bands_per_freq[f].max(1) as f32;
                    for t in 0..time_frames {
                        masked_stft[ch][f][t] = Complex32::new(
                            scatter_sum_real[ch][f][t] / denom,
                            scatter_sum_imag[ch][f][t] / denom,
                        );
                    }
                }
            }

            let reconstructed = istft(&masked_stft, &plan, Some(original_len));
            output_stems.push(reconstructed);
        }

        output_stems
    }
}
