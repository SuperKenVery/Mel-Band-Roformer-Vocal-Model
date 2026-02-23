use std::path::Path;
use thiserror::Error;
use serde::Deserialize;

use crate::model::config::ModelConfig;

#[derive(Error, Debug)]
pub enum WeightsError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Unsupported file format: {0}")]
    UnsupportedFormat(String),
    #[error("Config file not found alongside model")]
    ConfigNotFound,
    #[error("YAML parse error: {0}")]
    YamlError(#[from] serde_yaml::Error),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    Burn,
    Pytorch,
}

impl WeightFormat {
    pub fn from_path(path: &Path) -> Result<Self, WeightsError> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        match ext {
            "bpk" | "mpk" | "bin" => Ok(WeightFormat::Burn),
            "pt" | "pth" | "ckpt" => Ok(WeightFormat::Pytorch),
            _ => Err(WeightsError::UnsupportedFormat(ext.to_string())),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ConfigFile {
    model: ModelConfig,
}

pub fn load_config_for_model(model_path: &Path, config_path: Option<&Path>) -> Result<ModelConfig, WeightsError> {
    if let Some(config) = config_path {
        let content = std::fs::read_to_string(config)?;
        if let Ok(config_file) = serde_yaml::from_str::<ConfigFile>(&content) {
            return Ok(config_file.model);
        }
        let config: ModelConfig = serde_yaml::from_str(&content)?;
        return Ok(config);
    }

    let model_dir = model_path.parent().unwrap_or(Path::new("."));
    let config_candidates = [
        model_dir.join("config.yaml"),
        model_dir.join("config.yml"),
        model_dir.join("model_config.yaml"),
        model_path.with_extension("yaml"),
    ];

    for candidate in &config_candidates {
        if candidate.exists() {
            let content = std::fs::read_to_string(candidate)?;
            if let Ok(config_file) = serde_yaml::from_str::<ConfigFile>(&content) {
                return Ok(config_file.model);
            }
            if let Ok(config) = serde_yaml::from_str::<ModelConfig>(&content) {
                return Ok(config);
            }
        }
    }

    Ok(ModelConfig::default())
}
