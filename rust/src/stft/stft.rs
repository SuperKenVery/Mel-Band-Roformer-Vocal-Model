use super::FftPlan;
use num_complex::Complex32;

fn reflect_pad(signal: &[f32], pad_left: usize, pad_right: usize) -> Vec<f32> {
    let n = signal.len();
    let total_len = pad_left + n + pad_right;
    let mut result = vec![0.0f32; total_len];
    
    for i in 0..n {
        result[pad_left + i] = signal[i];
    }
    
    // Reflect padding on the left: signal[1], signal[2], ... (excluding signal[0])
    for i in 0..pad_left {
        let src_idx = pad_left - i;  // 1, 2, 3, ... for pad_left positions
        if src_idx < n {
            result[i] = signal[src_idx];
        }
    }
    
    // Reflect padding on the right: signal[n-2], signal[n-3], ... (excluding signal[n-1])
    for i in 0..pad_right {
        let src_idx = n.saturating_sub(2 + i);  // n-2, n-3, ... for pad_right positions
        result[pad_left + n + i] = signal[src_idx];
    }
    
    result
}

/// Compute Short-Time Fourier Transform
/// Input: [channels, samples]
/// Output: [channels, freq_bins, time_frames] as complex values
/// 
/// This uses center=True with reflect padding to match PyTorch's default behavior.
pub fn stft(signal: &[Vec<f32>], plan: &FftPlan) -> Vec<Vec<Vec<Complex32>>> {
    let num_channels = signal.len();
    let signal_len = signal.get(0).map(|s| s.len()).unwrap_or(0);

    if signal_len == 0 {
        return vec![vec![vec![]; 0]; num_channels];
    }

    let pad_left = plan.n_fft / 2;
    let pad_right = plan.n_fft / 2;
    let padded_len = signal_len + pad_left + pad_right;
    let num_frames = 1 + (padded_len - plan.n_fft) / plan.hop_length;
    let freq_bins = plan.freq_bins();

    let mut output = vec![vec![vec![Complex32::new(0.0, 0.0); num_frames]; freq_bins]; num_channels];

    for (ch, channel_signal) in signal.iter().enumerate() {
        let padded = reflect_pad(channel_signal, pad_left, pad_right);

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
