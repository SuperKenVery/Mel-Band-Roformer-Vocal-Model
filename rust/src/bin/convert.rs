use burn_ndarray::NdArray;
use burn_store::{ModuleSnapshot, PytorchStore, BurnpackStore};
use clap::Parser;
use log::{error, info};
use mel_band_roformer::io::weights::load_config_for_model;
use mel_band_roformer::model::MelBandRoformer;
use std::path::PathBuf;

type B = NdArray<f32>;

#[derive(Parser, Debug)]
#[command(name = "convert")]
#[command(about = "Convert PyTorch MelBandRoformer checkpoint to Burn format")]
struct Args {
    #[arg(short, long, help = "Path to PyTorch checkpoint (.pt or .ckpt)")]
    input: PathBuf,

    #[arg(short, long, help = "Output path for Burn model (.bpk)")]
    output: PathBuf,

    #[arg(short, long, help = "Path to config YAML file (optional)")]
    config: Option<PathBuf>,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    info!("Converting PyTorch checkpoint to Burn format");
    info!("=============================================");
    info!("Input: {:?}", args.input);
    info!("Output: {:?}", args.output);

    if !args.input.exists() {
        error!("Input file not found: {:?}", args.input);
        std::process::exit(1);
    }

    let ext = args.input.extension().and_then(|e| e.to_str()).unwrap_or("");
    if !matches!(ext, "pt" | "pth" | "ckpt") {
        error!("Input must be a PyTorch checkpoint (.pt, .pth, or .ckpt)");
        std::process::exit(1);
    }

    let config = match load_config_for_model(&args.input, args.config.as_deref()) {
        Ok(c) => c,
        Err(e) => {
            error!("Failed to load config: {}", e);
            std::process::exit(1);
        }
    };

    info!("Model config: dim={}, depth={}, num_bands={}", config.dim, config.depth, config.num_bands);

    let device = Default::default();
    let mut model = MelBandRoformer::<B>::new(&device, config);

    info!("Loading PyTorch weights...");
    let mut store = PytorchStore::from_file(&args.input)
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

    if let Err(e) = model.load_from(&mut store) {
        error!("Failed to load PyTorch weights: {:?}", e);
        std::process::exit(1);
    }

    info!("Saving Burn model...");
    let mut out_store = BurnpackStore::from_file(&args.output).overwrite(true);
    if let Err(e) = model.save_into(&mut out_store) {
        error!("Failed to save Burn model: {:?}", e);
        std::process::exit(1);
    }

    info!("Successfully converted to {:?}", args.output);
}
