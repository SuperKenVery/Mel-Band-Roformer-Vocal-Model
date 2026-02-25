use crate::io::weights::{WeightsError, load_config_for_model};
use crate::model::{MelBandRoformer, ModelConfig};
use burn::tensor::backend::Backend;
use burn_store::{BurnpackStore, ModuleSnapshot};
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Failed to load weights: {0}")]
    WeightsError(#[from] WeightsError),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Invalid audio format")]
    InvalidAudioFormat,
    #[error("Failed to load model: {0}")]
    LoadError(String),
}

pub struct InferenceEngine<B: Backend> {
    model: MelBandRoformer<B>,
    config: ModelConfig,
}

impl<B: Backend> InferenceEngine<B> {
    pub fn new(
        model_path: &Path,
        config_path: Option<&Path>,
        device: &B::Device,
    ) -> Result<Self, InferenceError> {
        let mut config = load_config_for_model(model_path, config_path)?;
        
        config.attn_dropout = 0.0;
        config.ff_dropout = 0.0;
        
        let mut model = MelBandRoformer::<B>::new(device, config.clone());

        let mut store = BurnpackStore::from_file(model_path);
        model.load_from(&mut store).map_err(|e| {
            InferenceError::LoadError(format!("Failed to load model: {:?}", e))
        })?;

        Ok(Self { model, config })
    }

    pub fn separate(
        &self,
        audio: &[Vec<f32>],
        _sample_rate: u32,
        chunk_size: usize,
        num_overlap: usize,
    ) -> Result<Vec<Vec<Vec<f32>>>, InferenceError> {
        if audio.is_empty() {
            return Err(InferenceError::InvalidAudioFormat);
        }

        let expected_channels = self.config.audio_channels();
        if audio.len() != expected_channels {
            return Err(InferenceError::InvalidAudioFormat);
        }

        let total_samples = audio[0].len();

        if total_samples <= chunk_size {
            return Ok(self.model.forward(audio));
        }

        let overlap_size = chunk_size / (num_overlap + 1);
        let step_size = chunk_size - overlap_size;

        let num_stems = self.config.num_stems;
        let num_channels = audio.len();
        let mut output_stems: Vec<Vec<Vec<f32>>> =
            vec![vec![vec![0.0; total_samples]; num_channels]; num_stems];
        let mut weight_sum = vec![0.0f32; total_samples];

        let mut start = 0;
        while start < total_samples {
            let end = (start + chunk_size).min(total_samples);
            let chunk_len = end - start;

            let chunk: Vec<Vec<f32>> = audio
                .iter()
                .map(|ch| ch[start..end].to_vec())
                .collect();

            let chunk_stems = self.model.forward(&chunk);

            let fade_in_len = if start == 0 { 0 } else { overlap_size / 2 };
            let fade_out_len = if end >= total_samples {
                0
            } else {
                overlap_size / 2
            };

            for (stem_idx, stem) in chunk_stems.iter().enumerate() {
                for (ch_idx, ch) in stem.iter().enumerate() {
                    for (i, &sample) in ch.iter().enumerate() {
                        let pos = start + i;
                        if pos >= total_samples {
                            break;
                        }

                        let weight = if i < fade_in_len {
                            i as f32 / fade_in_len as f32
                        } else if i >= chunk_len - fade_out_len {
                            (chunk_len - i) as f32 / fade_out_len as f32
                        } else {
                            1.0
                        };

                        output_stems[stem_idx][ch_idx][pos] += sample * weight;
                        if stem_idx == 0 && ch_idx == 0 {
                            weight_sum[pos] += weight;
                        }
                    }
                }
            }

            if end >= total_samples {
                break;
            }
            start += step_size;
        }

        for stem in &mut output_stems {
            for ch in stem {
                for (i, sample) in ch.iter_mut().enumerate() {
                    if weight_sum[i] > 0.0 {
                        *sample /= weight_sum[i];
                    }
                }
            }
        }

        Ok(output_stems)
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
}
