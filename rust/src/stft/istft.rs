use super::FftPlan;
use num_complex::Complex32;

/// Compute Inverse Short-Time Fourier Transform with overlap-add
/// Input: [channels, freq_bins, time_frames] as complex values
/// Output: [channels, samples]
pub fn istft(
    spectrogram: &[Vec<Vec<Complex32>>],
    plan: &FftPlan,
    output_length: Option<usize>,
) -> Vec<Vec<f32>> {
    let num_channels = spectrogram.len();
    if num_channels == 0 {
        return vec![];
    }

    let num_frames = spectrogram[0].get(0).map(|f| f.len()).unwrap_or(0);
    if num_frames == 0 {
        return vec![vec![]; num_channels];
    }

    let reconstructed_len = plan.n_fft + (num_frames - 1) * plan.hop_length;
    let pad_left = plan.n_fft / 2;

    let mut output = vec![vec![0.0f32; reconstructed_len]; num_channels];
    let mut window_sum = vec![0.0f32; reconstructed_len];

    let window_sq: Vec<f32> = plan.window.iter().map(|w| w * w).collect();

    for (ch, channel_spec) in spectrogram.iter().enumerate() {
        for frame_idx in 0..num_frames {
            let spectrum: Vec<rustfft::num_complex::Complex<f32>> = channel_spec
                .iter()
                .map(|freq_bin| {
                    let c = freq_bin[frame_idx];
                    rustfft::num_complex::Complex::new(c.re, c.im)
                })
                .collect();

            let frame = plan.inverse(&spectrum);

            let start = frame_idx * plan.hop_length;
            for (i, &sample) in frame.iter().enumerate() {
                if start + i < reconstructed_len {
                    output[ch][start + i] += sample;
                    if ch == 0 {
                        window_sum[start + i] += window_sq[i];
                    }
                }
            }
        }
    }

    let eps = 1e-8;
    for ch in 0..num_channels {
        for i in 0..reconstructed_len {
            if window_sum[i] > eps {
                output[ch][i] /= window_sum[i];
            }
        }
    }

    for ch in 0..num_channels {
        let final_len = output_length.unwrap_or(reconstructed_len - plan.n_fft);
        let trimmed: Vec<f32> = output[ch]
            .iter()
            .skip(pad_left)
            .take(final_len)
            .copied()
            .collect();
        output[ch] = trimmed;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stft::stft;

    #[test]
    fn test_stft_istft_roundtrip() {
        let plan = FftPlan::new(2048, 441, 2048);

        let original: Vec<Vec<f32>> = vec![(0..44100)
            .map(|i| (i as f32 * 0.01).sin())
            .collect()];
        let original_len = original[0].len();

        let spectrum = stft(&original, &plan);
        let reconstructed = istft(&spectrum, &plan, Some(original_len));

        assert_eq!(reconstructed.len(), 1);
        assert_eq!(reconstructed[0].len(), original_len);

        let mut max_error = 0.0f32;
        let skip = plan.n_fft;
        let check_len = original_len.saturating_sub(skip * 2);

        for i in skip..(skip + check_len) {
            let error = (reconstructed[0][i] - original[0][i]).abs();
            max_error = max_error.max(error);
        }

        assert!(
            max_error < 0.1,
            "STFT/ISTFT roundtrip error too high: {}",
            max_error
        );
    }
}
