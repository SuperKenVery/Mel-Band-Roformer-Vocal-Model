use super::FftPlan;
use num_complex::Complex32;

/// Compute Short-Time Fourier Transform
/// Input: [channels, samples]
/// Output: [channels, freq_bins, time_frames] as complex values
pub fn stft(signal: &[Vec<f32>], plan: &FftPlan) -> Vec<Vec<Vec<Complex32>>> {
    let num_channels = signal.len();
    let signal_len = signal.get(0).map(|s| s.len()).unwrap_or(0);

    if signal_len == 0 {
        return vec![vec![vec![]; 0]; num_channels];
    }

    let padded_len = signal_len + plan.n_fft;
    let num_frames = 1 + (padded_len - plan.n_fft) / plan.hop_length;
    let freq_bins = plan.freq_bins();

    let mut output = vec![vec![vec![Complex32::new(0.0, 0.0); num_frames]; freq_bins]; num_channels];

    for (ch, channel_signal) in signal.iter().enumerate() {
        let mut padded = vec![0.0f32; padded_len];
        let pad_left = plan.n_fft / 2;
        for (i, &sample) in channel_signal.iter().enumerate() {
            padded[pad_left + i] = sample;
        }

        for frame_idx in 0..num_frames {
            let start = frame_idx * plan.hop_length;
            let end = (start + plan.n_fft).min(padded.len());
            let frame = &padded[start..end];

            let spectrum = plan.forward(frame);

            for (freq_idx, &val) in spectrum.iter().enumerate() {
                output[ch][freq_idx][frame_idx] = Complex32::new(val.re, val.im);
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stft_shape() {
        let plan = FftPlan::new(2048, 441, 2048);
        let signal = vec![vec![0.0f32; 44100]]; // 1 second of silence at 44.1kHz

        let result = stft(&signal, &plan);

        assert_eq!(result.len(), 1); // 1 channel
        assert_eq!(result[0].len(), 1025); // n_fft/2 + 1 freq bins
        assert!(result[0][0].len() > 0); // Some time frames
    }

    #[test]
    fn test_stft_stereo() {
        let plan = FftPlan::new(2048, 441, 2048);
        let signal = vec![vec![0.0f32; 44100], vec![0.0f32; 44100]]; // stereo

        let result = stft(&signal, &plan);

        assert_eq!(result.len(), 2); // 2 channels
    }
}
