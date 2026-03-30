#![recursion_limit = "512"]

use burn::tensor::backend::Backend;
use burn_wgpu::{Wgpu, WgpuDevice};
use melband_roformer::model::mel_band_roformer::{MelBandRoformer, MelBandRoformerConfig, MelBandConstants};
use std::fs::File;
use safetensors::SafeTensors;
use memmap2::MmapOptions;
use burn::tensor::{Tensor, TensorData, Int};
use burn::module::Module;
use burn::record::{Recorder, FullPrecisionSettings};
use burn::tensor::quantization::{Calibration, QuantScheme, QuantValue, QuantLevel, QuantParam, BlockSize};
use melband_roformer::model::weight_quantizer::WeightQuantizer;
use burn_import::pytorch::PyTorchFileRecorder;
use rubato::{Resampler, FftFixedIn};
use std::env;
use std::path::Path;

fn main() {
    type B = Wgpu;
    let device = WgpuDevice::default();

    let args: Vec<String> = env::args().collect();
    let mut model_path = "model.pt".to_string();
    let mut quantize = false;

    for arg in args.iter().skip(1) {
        if arg == "--quantize" {
            quantize = true;
        } else if !arg.starts_with("--") {
            model_path = arg.clone();
        }
    }

    println!("Model path: {}", model_path);
    println!("Quantize: {}", quantize);

    println!("Loading audio...");
    let audio_path = "/Users/bytedance/Desktop/歲月無聲.mp3";
    let (mut samples, channels, sample_rate) = match melband_roformer::audio::load_audio(audio_path) {
        Ok(res) => res,
        Err(_) => {
            println!("Audio file not found, using silence for testing.");
            (vec![0.0; 44100 * 2 * 10], 2, 44100)
        }
    };

    println!("Loaded {} samples at {} Hz, {} channels", samples.len(), sample_rate, channels);

    if sample_rate != 44100 {
        println!("Resampling from {} to 44100...", sample_rate);
        samples = resample(&samples, sample_rate as usize, 44100, channels as usize);
    }
    let sample_rate = 44100;

    let start_sec = 32;
    let duration_sec = 10;
    let start_sample = sample_rate * start_sec * channels;
    let end_sample = start_sample + sample_rate * duration_sec * channels;

    let samples = if samples.len() > end_sample {
        println!("Slicing audio from {}s to {}s...", start_sec, start_sec + duration_sec);
        samples[start_sample..end_sample].to_vec()
    } else {
        println!("Audio too short, using available samples...");
        samples
    };

    let config = MelBandRoformerConfig::new(384, 6)
        .with_stereo(true)
        .with_num_stems(1)
        .with_time_transformer_depth(1)
        .with_freq_transformer_depth(1)
        .with_num_bands(60)
        .with_dim_head(64)
        .with_heads(8)
        .with_attn_dropout(0.0)
        .with_ff_dropout(0.0)
        .with_dim_freqs_in(1025)
        .with_sample_rate(44100)
        .with_stft_n_fft(2048)
        .with_stft_hop_length(441)
        .with_stft_win_length(2048)
        .with_mask_estimator_depth(2);

    println!("Loading constants...");
    let constants_path = "constants.safetensors";
    if !Path::new(constants_path).exists() {
        panic!("constants.safetensors not found");
    }
    let constants = load_constants::<B>(constants_path, &device);

    println!("Initializing model...");
    let model: MelBandRoformer<B> = MelBandRoformer::new(&config, &device, constants);

    println!("Loading weights from {}...", model_path);
    let record = PyTorchFileRecorder::<FullPrecisionSettings>::new()
        .load(model_path.into(), &device)
        .expect("Failed to load weights from pt file");
    let model = model.load_record(record);

    let model = if quantize {
        println!("Quantizing transformer weights (Int8 symmetric, block-32)...");
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_level(QuantLevel::Block(BlockSize::new([32])))
            .with_param(QuantParam::F32);
        let mut quantizer = WeightQuantizer {
            calibration: Calibration::MinMax,
            scheme,
        };
        model.quantize_transformers(&mut quantizer)
    } else {
        model
    };

    let channels = 2;
    let samples_per_channel = samples.len() / channels;
    let input_shape = [1, channels, samples_per_channel];

    let mut left = Vec::with_capacity(samples_per_channel);
    let mut right = Vec::with_capacity(samples_per_channel);
    for chunk in samples.chunks(2) {
        if chunk.len() == 2 {
            left.push(chunk[0]);
            right.push(chunk[1]);
        }
    }
    let mut planar_samples = left;
    planar_samples.extend(right);

    let input: Tensor<B, 3> = Tensor::from_floats(
        TensorData::new(planar_samples, input_shape),
        &device,
    );

    println!("Running inference...");
    let output = model.forward(input);

    println!("Saving output...");
    let dims = output.dims();
    let time = dims[3];
    let output_data = output.into_data();
    let output_vec: Vec<f32> = output_data.to_vec().unwrap();

    let mut interleaved = Vec::with_capacity(time * 2);
    let left_out = &output_vec[0..time];
    let right_out = &output_vec[time..time * 2];

    for i in 0..time {
        interleaved.push(left_out[i]);
        interleaved.push(right_out[i]);
    }

    melband_roformer::audio::save_audio("output.wav", &interleaved, 44100, 2).unwrap();
    println!("Done! Saved to output.wav");
}

fn load_constants<B: Backend>(path: &str, device: &B::Device) -> MelBandConstants<B> {
    let file = File::open(path).unwrap();
    let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    let tensors = SafeTensors::deserialize(&mmap).unwrap();

    let load_tensor_1d = |name: &str| -> Tensor<B, 1> {
        let view = tensors.tensor(name).unwrap();
        let data = TensorData::new(
            view.data().chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
            [view.shape()[0]],
        );
        Tensor::from_floats(data, device)
    };

    let load_tensor_1d_int = |name: &str| -> Tensor<B, 1, Int> {
        let view = tensors.tensor(name).unwrap();
        let data = TensorData::new(
            view.data().chunks(8).map(|b| i64::from_le_bytes(b.try_into().unwrap()) as i32).collect(),
            [view.shape()[0]],
        );
        Tensor::from_ints(data, device)
    };

    let load_tensor_3d = |name: &str| -> Tensor<B, 3> {
        let view = tensors.tensor(name).unwrap();
        let shape = [view.shape()[0], view.shape()[1], view.shape()[2]];
        let data = TensorData::new(
            view.data().chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
            shape,
        );
        Tensor::from_floats(data, device)
    };

    MelBandConstants {
        freq_indices: load_tensor_1d_int("freq_indices"),
        num_bands_per_freq: load_tensor_1d("num_bands_per_freq"),
        num_freqs_per_band: load_tensor_1d("num_freqs_per_band"),
        stft_kernel_real: load_tensor_3d("stft_kernel_real"),
        stft_kernel_imag: load_tensor_3d("stft_kernel_imag"),
        istft_kernel_real: load_tensor_3d("istft_kernel_real"),
        istft_kernel_imag: load_tensor_3d("istft_kernel_imag"),
    }
}

fn resample(samples: &[f32], from_rate: usize, to_rate: usize, channels: usize) -> Vec<f32> {
    let mut input_planar = vec![Vec::new(); channels];
    for (i, sample) in samples.iter().enumerate() {
        input_planar[i % channels].push(*sample);
    }

    let chunk_size = 1024;
    let mut resampler = FftFixedIn::<f32>::new(from_rate, to_rate, chunk_size, 2, channels).unwrap();

    let mut output_planar = vec![Vec::new(); channels];
    let mut output_buffers = resampler.output_buffer_allocate(true);

    let num_frames = input_planar[0].len();
    let num_chunks = (num_frames + chunk_size - 1) / chunk_size;

    for i in 0..num_chunks {
        let start = i * chunk_size;
        let end = usize::min((i + 1) * chunk_size, num_frames);
        let actual_len = end - start;

        let mut chunk_planar = vec![Vec::new(); channels];
        for c in 0..channels {
            let mut chunk = input_planar[c][start..end].to_vec();
            if chunk.len() < chunk_size {
                chunk.resize(chunk_size, 0.0);
            }
            chunk_planar[c] = chunk;
        }

        let (_, out_frames) = resampler.process_into_buffer(&chunk_planar, &mut output_buffers, None).unwrap();

        let valid_out_frames = if actual_len < chunk_size {
            (actual_len as f64 * to_rate as f64 / from_rate as f64).ceil() as usize
        } else {
            out_frames
        };

        for c in 0..channels {
            output_planar[c].extend_from_slice(&output_buffers[c][0..valid_out_frames]);
        }
    }

    let mut output = Vec::with_capacity(output_planar[0].len() * channels);
    for i in 0..output_planar[0].len() {
        for c in 0..channels {
            output.push(output_planar[c][i]);
        }
    }

    output
}
