use crate::io::weights::{WeightFormat, WeightsError, load_config_for_model};
use crate::model::{MelBandRoformer, ModelConfig};
use burn::tensor::backend::Backend;
use burn_store::{BurnpackStore, ModuleSnapshot, PytorchStore};
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
    #[error("Failed to save model: {0}")]
    SaveError(String),
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
        let config = load_config_for_model(model_path, config_path)?;
        let mut model = MelBandRoformer::<B>::new(device, config.clone());

        let format = WeightFormat::from_path(model_path)?;

        match format {
            WeightFormat::Burn => {
                let mut store = BurnpackStore::from_file(model_path);
                model.load_from(&mut store).map_err(|e| {
                    InferenceError::LoadError(format!("Failed to load burn model: {:?}", e))
                })?;
                // model.fix_load_weights(); // Weights in Burn format should already be correct
            }
            WeightFormat::Pytorch => {
                let mut store = PytorchStore::from_file(model_path)
                    .with_key_remapping(r"^layers\.(\d+)\.0\.", "time_transformers.$1.")
                    .with_key_remapping(r"^layers\.(\d+)\.1\.", "freq_transformers.$1.")
                    .with_key_remapping(r"\.layers\.(\d+)\.0\.rotary_embed\.freqs", ".layers.$1.rotary_freqs")
                    .with_key_remapping(r"\.layers\.(\d+)\.0\.norm\.", ".layers.$1.attention.norm.")
                    .with_key_remapping(r"\.layers\.(\d+)\.0\.to_qkv\.", ".layers.$1.attention.to_qkv.")
                    .with_key_remapping(r"\.layers\.(\d+)\.0\.to_gates\.", ".layers.$1.attention.to_gates.")
                    .with_key_remapping(r"\.layers\.(\d+)\.0\.to_out\.0\.", ".layers.$1.attention.to_out.")
                    .with_key_remapping(r"\.layers\.(\d+)\.1\.net\.0\.", ".layers.$1.ff.norm.")
                    .with_key_remapping(r"\.layers\.(\d+)\.1\.net\.1\.", ".layers.$1.ff.linear1.")
                    .with_key_remapping(r"\.layers\.(\d+)\.1\.net\.4\.", ".layers.$1.ff.linear2.")
                    .with_key_remapping(r"band_split\.to_features\.(\d+)\.0\.", "band_split.to_features.$1.norm.")
                    .with_key_remapping(r"band_split\.to_features\.(\d+)\.1\.", "band_split.to_features.$1.linear.")
                    .with_key_remapping(r"mask_estimators\.(\d+)\.to_freqs\.(\d+)\.0\.0\.", "mask_estimators.$1.to_freqs.$2.mlp.linear1.")
                    .with_key_remapping(r"mask_estimators\.(\d+)\.to_freqs\.(\d+)\.0\.2\.", "mask_estimators.$1.to_freqs.$2.mlp.linear2.")
                    .with_key_remapping(r"mask_estimators\.(\d+)\.to_freqs\.(\d+)\.0\.4\.", "mask_estimators.$1.to_freqs.$2.mlp.linear3.")
                    .allow_partial(true);
                model.load_from(&mut store).map_err(|e| {
                    InferenceError::LoadError(format!("Failed to load pytorch model: {:?}", e))
                })?;
                model.fix_load_weights();
            }
        }

        Ok(Self { model, config })
    }

    pub fn from_config(config: ModelConfig, device: &B::Device) -> Self {
        let model = MelBandRoformer::<B>::new(device, config.clone());
        Self { model, config }
    }

    pub fn save_burn(&self, output_path: &Path) -> Result<(), InferenceError> {
        let mut store = BurnpackStore::from_file(output_path).overwrite(true);
        self.model.save_into(&mut store).map_err(|e| {
            InferenceError::SaveError(format!("Failed to save burn model: {:?}", e))
        })?;
        Ok(())
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

    pub fn model(&self) -> &MelBandRoformer<B> {
        &self.model
    }
}

pub fn convert_pytorch_to_burn<B: Backend>(
    pytorch_path: &Path,
    output_path: &Path,
    config_path: Option<&Path>,
    device: &B::Device,
) -> Result<(), InferenceError> {
    let engine: InferenceEngine<B> = InferenceEngine::new(pytorch_path, config_path, device)?;
    engine.save_burn(output_path)?;
    Ok(())
}
