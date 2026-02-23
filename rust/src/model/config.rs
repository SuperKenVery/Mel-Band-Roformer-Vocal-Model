use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub dim: usize,
    pub depth: usize,
    pub num_stems: usize,
    pub time_transformer_depth: usize,
    pub freq_transformer_depth: usize,
    pub num_bands: usize,
    pub dim_head: usize,
    pub heads: usize,
    pub stereo: bool,
    pub mask_estimator_depth: usize,
    pub stft_n_fft: usize,
    pub stft_hop_length: usize,
    pub stft_win_length: usize,
    pub sample_rate: u32,
    #[serde(default)]
    pub attn_dropout: f32,
    #[serde(default)]
    pub ff_dropout: f32,
    #[serde(default = "default_ff_mult")]
    pub ff_mult: usize,
}

fn default_ff_mult() -> usize {
    4
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            dim: 384,
            depth: 6,
            num_stems: 1,
            time_transformer_depth: 1,
            freq_transformer_depth: 1,
            num_bands: 60,
            dim_head: 64,
            heads: 8,
            stereo: true,
            mask_estimator_depth: 2,
            stft_n_fft: 2048,
            stft_hop_length: 441,
            stft_win_length: 2048,
            sample_rate: 44100,
            attn_dropout: 0.0,
            ff_dropout: 0.0,
            ff_mult: 4,
        }
    }
}

impl ModelConfig {
    pub fn audio_channels(&self) -> usize {
        if self.stereo {
            2
        } else {
            1
        }
    }

    pub fn freq_bins(&self) -> usize {
        self.stft_n_fft / 2 + 1
    }

    pub fn dim_inner(&self) -> usize {
        self.heads * self.dim_head
    }
}
