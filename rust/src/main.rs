use burn::backend::Metal;
use burn::backend::wgpu::WgpuDevice;
use burn_ndarray::NdArray;
use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use log::{error, info};
use mel_band_roformer::InferenceEngine;
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
    #[arg(short, long, help = "Path to the Burn model file (.bpk)")]
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
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    match args.device {
        DeviceType::Gpu => run_inference::<GpuBackend>(&args, WgpuDevice::default()),
        DeviceType::Cpu => run_inference::<CpuBackend>(&args, Default::default()),
    }
}

fn run_inference<B: burn::tensor::backend::Backend>(args: &Args, device: B::Device) {
    let device_name = std::any::type_name::<B>();
    let device_short = if device_name.contains("Metal") || device_name.contains("Wgpu") {
        "GPU"
    } else {
        "CPU"
    };

    info!("Mel-Band-Roformer Audio Source Separation ({})", device_short);
    info!("================================================");
    info!("Model: {:?}", args.model);
    info!("Input: {:?}", args.input);
    info!("Output: {:?}", args.output);

    if !args.model.exists() {
        error!("Model file not found: {:?}", args.model);
        std::process::exit(1);
    }

    if !args.input.exists() {
        error!("Input file not found: {:?}", args.input);
        std::process::exit(1);
    }

    std::fs::create_dir_all(&args.output).unwrap_or_else(|e| {
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

    pb.set_message(format!("Loading model on {}...", device_short));

    let engine: InferenceEngine<B> =
        match InferenceEngine::new(&args.model, args.config.as_deref(), &device) {
            Ok(e) => e,
            Err(e) => {
                pb.finish_with_message("Failed!");
                error!("Failed to load model: {}", e);
                std::process::exit(1);
            }
        };

    pb.set_message("Loading audio...");
    let (samples, sample_rate) = match mel_band_roformer::io::wav::read_wav(&args.input) {
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

    pb.set_message(format!("Processing on {}...", device_short));
    let stems = match engine.separate(&samples, sample_rate, args.chunk_size, args.num_overlap) {
        Ok(s) => s,
        Err(e) => {
            pb.finish_with_message("Failed!");
            error!("Inference failed: {}", e);
            std::process::exit(1);
        }
    };

    pb.set_message("Saving output...");
    if !stems.is_empty() {
        let instrumental_path = args.output.join("instrumental.wav");
        let vocals = &stems[0];
        let instrumental: Vec<Vec<f32>> = samples
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
        if let Err(e) =
            mel_band_roformer::io::wav::write_wav(&instrumental_path, &instrumental, sample_rate)
        {
            error!("Failed to write output: {}", e);
            std::process::exit(1);
        }
        info!("Saved: {:?}", instrumental_path);
    }

    pb.finish_with_message("Done!");
    info!("Processing completed in {:.2?}", start.elapsed());
}
