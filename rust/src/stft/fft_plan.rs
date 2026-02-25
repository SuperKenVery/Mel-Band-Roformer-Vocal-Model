use realfft::{RealFftPlanner, RealToComplex};
use rustfft::{FftPlanner, num_complex::Complex};
use std::sync::Arc;

pub struct FftPlan {
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub window: Vec<f32>,
    r2c: Arc<dyn RealToComplex<f32>>,
    c2r: Arc<dyn rustfft::Fft<f32>>,
}

impl FftPlan {
    pub fn new(n_fft: usize, hop_length: usize, win_length: usize) -> Self {
        let window = hann_window(win_length);

        let mut real_planner = RealFftPlanner::<f32>::new();
        let r2c = real_planner.plan_fft_forward(n_fft);

        let mut planner = FftPlanner::<f32>::new();
        let c2r = planner.plan_fft_inverse(n_fft);

        Self {
            n_fft,
            hop_length,
            win_length,
            window,
            r2c,
            c2r,
        }
    }

    pub fn forward(&self, frame: &[f32]) -> Vec<Complex<f32>> {
        let mut input: Vec<f32> = vec![0.0; self.n_fft];
        let windowed_len = frame.len().min(self.win_length);

        for i in 0..windowed_len {
            input[i] = frame[i] * self.window[i];
        }

        let mut spectrum = self.r2c.make_output_vec();
        self.r2c
            .process(&mut input, &mut spectrum)
            .expect("FFT forward failed");

        spectrum
    }

    pub fn inverse(&self, spectrum: &[Complex<f32>]) -> Vec<f32> {
        let mut full_spectrum: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); self.n_fft];

        for (i, &val) in spectrum.iter().enumerate() {
            full_spectrum[i] = val;
        }
        for i in 1..(self.n_fft / 2) {
            full_spectrum[self.n_fft - i] = spectrum[i].conj();
        }

        self.c2r.process(&mut full_spectrum);

        let scale = 1.0 / self.n_fft as f32;
        full_spectrum
            .iter()
            .take(self.win_length)
            .enumerate()
            .map(|(i, c)| c.re * scale * self.window[i])
            .collect()
    }

    pub fn num_frames(&self, signal_length: usize) -> usize {
        if signal_length < self.n_fft {
            1
        } else {
            1 + (signal_length - self.n_fft) / self.hop_length
        }
    }

    pub fn freq_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }
}

fn hann_window(length: usize) -> Vec<f32> {
    // Match PyTorch default periodic=True
    // sin^2(pi * n / L)
    (0..length)
        .map(|i| {
            let x = std::f32::consts::PI * i as f32 / length as f32;
            (x.sin()).powi(2)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window() {
        let window = hann_window(4);
        assert!((window[0] - 0.0).abs() < 1e-6);
        assert!((window[1] - 0.5).abs() < 1e-6);
        assert!((window[2] - 1.0).abs() < 1e-6);
        assert!((window[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_fft_roundtrip() {
        let plan = FftPlan::new(8, 4, 8);
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let spectrum = plan.forward(&input);
        let output = plan.inverse(&spectrum);

        for i in 0..input.len() {
            let expected = input[i] * plan.window[i] * plan.window[i];
            assert!(
                (output[i] - expected).abs() < 1e-4,
                "Mismatch at {}: {} vs {}",
                i,
                output[i],
                expected
            );
        }
    }
}
