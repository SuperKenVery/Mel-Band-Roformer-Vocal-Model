use burn::backend::Metal;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn_ndarray::NdArray;
use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use log::{error, info};
use mel_band_roformer::{InferenceEngine, convert_pytorch_to_burn};
use std::path::PathBuf;
use std::time::Instant;

type GpuBackend = Metal;
type CpuBackend = NdArray<f32>;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DeviceType {
    Gpu,
    Cpu,
}

#[derive(Parser, Debug)]
#[command(name = "mel-band-roformer")]
#[command(about = "GPU-accelerated Mel-Band-Roformer for audio source separation")]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Infer {
        #[arg(
            short,
            long,
            help = "Path to the model file (.bpk for Burn, .pt/.ckpt for PyTorch)"
        )]
        model: PathBuf,

        #[arg(short, long, help = "Input audio file (WAV format)")]
        input: PathBuf,

        #[arg(short, long, help = "Output directory for separated stems")]
        output: PathBuf,

        #[arg(short, long, help = "Path to config YAML file (optional)")]
        config: Option<PathBuf>,

        #[arg(
            short,
            long,
            value_enum,
            default_value = "gpu",
            help = "Device to use (gpu or cpu)"
        )]
        device: DeviceType,

        #[arg(long, default_value = "352800", help = "Chunk size for processing")]
        chunk_size: usize,

        #[arg(long, default_value = "2", help = "Number of overlapping chunks")]
        num_overlap: usize,
    },
    Convert {
        #[arg(short, long, help = "Path to PyTorch checkpoint (.pt or .ckpt)")]
        input: PathBuf,

        #[arg(short, long, help = "Output path for Burn model (.bpk)")]
        output: PathBuf,

        #[arg(short, long, help = "Path to config YAML file (optional)")]
        config: Option<PathBuf>,
    },
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    match args.command {
        Commands::Infer {
            model,
            input,
            output,
            config,
            device,
            chunk_size,
            num_overlap,
        } => match device {
            DeviceType::Gpu => {
                run_inference_gpu(
                    &model,
                    &input,
                    &output,
                    config.as_deref(),
                    chunk_size,
                    num_overlap,
                );
            }
            DeviceType::Cpu => {
                run_inference_cpu(
                    &model,
                    &input,
                    &output,
                    config.as_deref(),
                    chunk_size,
                    num_overlap,
                );
            }
        },
        Commands::Convert {
            input,
            output,
            config,
        } => {
            run_convert(&input, &output, config.as_deref());
        }
    }
}

fn run_inference_gpu(
    model_path: &PathBuf,
    input_path: &PathBuf,
    output_path: &PathBuf,
    config_path: Option<&std::path::Path>,
    chunk_size: usize,
    num_overlap: usize,
) {
    info!("Mel-Band-Roformer Audio Source Separation (GPU)");
    info!("================================================");
    info!("Model: {:?}", model_path);
    info!("Input: {:?}", input_path);
    info!("Output: {:?}", output_path);

    if !model_path.exists() {
        error!("Model file not found: {:?}", model_path);
        std::process::exit(1);
    }

    if !input_path.exists() {
        error!("Input file not found: {:?}", input_path);
        std::process::exit(1);
    }

    std::fs::create_dir_all(output_path).unwrap_or_else(|e| {
        error!("Failed to create output directory: {}", e);
        std::process::exit(1);
    });

    let start = Instant::now();

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );

    pb.set_message("Loading model on GPU...");

    let device = WgpuDevice::default();
    info!("Using device: {:?}", device);

    let engine: InferenceEngine<GpuBackend> =
        match InferenceEngine::new(model_path, config_path, &device) {
            Ok(e) => e,
            Err(e) => {
                pb.finish_with_message("Failed!");
                error!("Failed to load model: {}", e);
                std::process::exit(1);
            }
        };

    pb.set_message("Loading audio...");
    let (samples, sample_rate) = match mel_band_roformer::io::wav::read_wav(input_path) {
        Ok(s) => s,
        Err(e) => {
            pb.finish_with_message("Failed!");
            error!("Failed to read audio: {}", e);
            std::process::exit(1);
        }
    };

    info!(
        "Loaded audio: {} samples, {} Hz, {} channels",
        samples[0].len(),
        sample_rate,
        samples.len()
    );

    pb.set_message("Processing on GPU...");
    let stems = match engine.separate(&samples, sample_rate, chunk_size, num_overlap) {
        Ok(s) => s,
        Err(e) => {
            pb.finish_with_message("Failed!");
            error!("Inference failed: {}", e);
            std::process::exit(1);
        }
    };

    pb.set_message("Saving output...");
    if stems.len() >= 1 {
        let instrumental_path = output_path.join("instrumental.wav");
        let input_samples: Vec<Vec<f32>> = samples.clone();
        let vocals = &stems[0];
        let instrumental: Vec<Vec<f32>> = input_samples
            .iter()
            .zip(vocals.iter())
            .map(|(input_ch, vocal_ch)| {
                input_ch
                    .iter()
                    .zip(vocal_ch.iter())
                    .map(|(i, v)| i - v)
                    .collect()
            })
            .collect();
        if let Err(e) = mel_band_roformer::io::wav::write_wav(&instrumental_path, &instrumental, sample_rate) {
            error!("Failed to write output: {}", e);
            std::process::exit(1);
        }
        info!("Saved: {:?}", instrumental_path);
    }

    pb.finish_with_message("Done!");
    info!("Processing completed in {:.2?}", start.elapsed());
}

fn run_inference_cpu(
    model_path: &PathBuf,
    input_path: &PathBuf,
    output_path: &PathBuf,
    config_path: Option<&std::path::Path>,
    chunk_size: usize,
    num_overlap: usize,
) {
    info!("Mel-Band-Roformer Audio Source Separation (CPU)");
    info!("================================================");
    info!("Model: {:?}", model_path);
    info!("Input: {:?}", input_path);
    info!("Output: {:?}", output_path);

    if !model_path.exists() {
        error!("Model file not found: {:?}", model_path);
        std::process::exit(1);
    }

    if !input_path.exists() {
        error!("Input file not found: {:?}", input_path);
        std::process::exit(1);
    }

    std::fs::create_dir_all(output_path).unwrap_or_else(|e| {
        error!("Failed to create output directory: {}", e);
        std::process::exit(1);
    });

    let start = Instant::now();

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );

    pb.set_message("Loading model on CPU...");

    let device = Default::default();

    let engine: InferenceEngine<CpuBackend> =
        match InferenceEngine::new(model_path, config_path, &device) {
            Ok(e) => e,
            Err(e) => {
                pb.finish_with_message("Failed!");
                error!("Failed to load model: {}", e);
                std::process::exit(1);
            }
        };

    pb.set_message("Loading audio...");
    let (samples, sample_rate) = match mel_band_roformer::io::wav::read_wav(input_path) {
        Ok(s) => s,
        Err(e) => {
            pb.finish_with_message("Failed!");
            error!("Failed to read audio: {}", e);
            std::process::exit(1);
        }
    };

    info!(
        "Loaded audio: {} samples, {} Hz, {} channels",
        samples[0].len(),
        sample_rate,
        samples.len()
    );

    pb.set_message("Processing on CPU...");
    let stems = match engine.separate(&samples, sample_rate, chunk_size, num_overlap) {
        Ok(s) => s,
        Err(e) => {
            pb.finish_with_message("Failed!");
            error!("Inference failed: {}", e);
            std::process::exit(1);
        }
    };

    pb.set_message("Saving output...");
    if stems.len() >= 1 {
        let instrumental_path = output_path.join("instrumental.wav");
        let input_samples: Vec<Vec<f32>> = samples.clone();
        let vocals = &stems[0];
        let instrumental: Vec<Vec<f32>> = input_samples
            .iter()
            .zip(vocals.iter())
            .map(|(input_ch, vocal_ch)| {
                input_ch
                    .iter()
                    .zip(vocal_ch.iter())
                    .map(|(i, v)| i - v)
                    .collect()
            })
            .collect();
        if let Err(e) = mel_band_roformer::io::wav::write_wav(&instrumental_path, &instrumental, sample_rate) {
            error!("Failed to write output: {}", e);
            std::process::exit(1);
        }
        info!("Saved: {:?}", instrumental_path);
    }

    pb.finish_with_message("Done!");
    info!("Processing completed in {:.2?}", start.elapsed());
}

fn run_convert(input_path: &PathBuf, output_path: &PathBuf, config_path: Option<&std::path::Path>) {
    info!("Converting PyTorch checkpoint to Burn format");
    info!("=============================================");
    info!("Input: {:?}", input_path);
    info!("Output: {:?}", output_path);

    if !input_path.exists() {
        error!("Input file not found: {:?}", input_path);
        std::process::exit(1);
    }

    let device = Default::default();

    match convert_pytorch_to_burn::<CpuBackend>(input_path, output_path, config_path, &device) {
        Ok(()) => {
            info!("Successfully converted to {:?}", output_path);
        }
        Err(e) => {
            error!("Conversion failed: {}", e);
            std::process::exit(1);
        }
    }
}
