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
use std::path::Path;

const CHUNK_SIZE: usize = 352800;
const NUM_OVERLAP: usize = 2;

fn demix_track<B: Backend>(
    model: &MelBandRoformer<B>,
    mix: &[f32],
    num_channels: usize,
    device: &B::Device,
) -> Vec<f32> {
    let samples_per_channel = mix.len() / num_channels;
    let step = CHUNK_SIZE / NUM_OVERLAP;
    let fade_size = CHUNK_SIZE / 10;
    let border = CHUNK_SIZE - step;

    // Deinterleave to planar [ch0_samples..., ch1_samples...]
    let mut planar: Vec<Vec<f32>> = vec![Vec::with_capacity(samples_per_channel); num_channels];
    for (i, &s) in mix.iter().enumerate() {
        planar[i % num_channels].push(s);
    }

    // Reflection-pad both ends by `border` samples
    let padded_len = if samples_per_channel > 2 * border && border > 0 {
        let mut padded: Vec<Vec<f32>> = Vec::with_capacity(num_channels);
        for ch in &planar {
            let mut p = Vec::with_capacity(ch.len() + 2 * border);
            // Left reflect: ch[border], ch[border-1], ..., ch[1]
            for j in (1..=border).rev() {
                p.push(ch[j]);
            }
            p.extend_from_slice(ch);
            // Right reflect: ch[len-2], ch[len-3], ..., ch[len-1-border]
            let n = ch.len();
            for j in (n - 1 - border..n - 1).rev() {
                p.push(ch[j]);
            }
            padded.push(p);
        }
        let plen = padded[0].len();
        planar = padded;
        plen
    } else {
        samples_per_channel
    };

    // Build trapezoidal window
    let mut base_window = vec![1.0f32; CHUNK_SIZE];
    for i in 0..fade_size {
        let t = i as f32 / fade_size as f32;
        base_window[i] *= t;
        base_window[CHUNK_SIZE - 1 - i] *= t;
    }

    // Accumulation buffers (planar, per-stem per-channel)
    // Model output: [1, num_stems, channels, time] → we only have 1 stem
    let mut result = vec![0.0f32; num_channels * padded_len];
    let mut counter = vec![0.0f32; num_channels * padded_len];

    let total_length = padded_len;
    let mut pos = 0;
    let mut chunk_idx = 0;

    while pos < total_length {
        let end = (pos + CHUNK_SIZE).min(total_length);
        let length = end - pos;

        // Extract chunk per channel, pad if needed
        let mut chunk_planar = Vec::with_capacity(num_channels * CHUNK_SIZE);
        for ch in &planar {
            let segment = &ch[pos..pos + length];
            let mut padded_seg = segment.to_vec();
            if length < CHUNK_SIZE {
                // Reflect-pad or zero-pad
                if length > CHUNK_SIZE / 2 + 1 {
                    // Reflect pad
                    let needed = CHUNK_SIZE - length;
                    for j in 0..needed {
                        let idx = length - 2 - (j % (length - 1));
                        padded_seg.push(padded_seg[idx]);
                    }
                } else {
                    padded_seg.resize(CHUNK_SIZE, 0.0);
                }
            }
            chunk_planar.extend_from_slice(&padded_seg);
        }

        // Create input tensor [1, channels, CHUNK_SIZE]
        let input: Tensor<B, 3> = Tensor::from_floats(
            TensorData::new(chunk_planar, [1, num_channels, CHUNK_SIZE]),
            device,
        );

        // Run model: output [1, 1, channels, CHUNK_SIZE]
        let output = model.forward(input);
        let output_data: Vec<f32> = output.into_data().to_vec().unwrap();

        // Apply window
        let mut window = base_window.clone();
        if pos == 0 {
            // First chunk: no fade-in
            for w in window.iter_mut().take(fade_size) {
                *w = 1.0;
            }
        }
        if pos + CHUNK_SIZE >= total_length {
            // Last chunk: no fade-out
            for w in window.iter_mut().skip(CHUNK_SIZE - fade_size) {
                *w = 1.0;
            }
        }

        // Accumulate: output_data is [channels * CHUNK_SIZE] (flattened from [1,1,channels,time])
        for ch in 0..num_channels {
            for j in 0..length {
                let src_idx = ch * CHUNK_SIZE + j;
                let dst_idx = ch * padded_len + pos + j;
                result[dst_idx] += output_data[src_idx] * window[j];
                counter[dst_idx] += window[j];
            }
        }

        chunk_idx += 1;
        if chunk_idx % 2 == 0 || pos + step >= total_length {
            println!("  Processed chunk {}, pos {}/{}", chunk_idx, pos + length, total_length);
        }
        pos += step;
    }

    // Normalize
    for i in 0..result.len() {
        if counter[i] > 0.0 {
            result[i] /= counter[i];
        }
    }

    // Remove border padding, re-interleave
    let start = if samples_per_channel > 2 * border && border > 0 { border } else { 0 };
    let mut interleaved = Vec::with_capacity(samples_per_channel * num_channels);
    for i in 0..samples_per_channel {
        for ch in 0..num_channels {
            interleaved.push(result[ch * padded_len + start + i]);
        }
    }

    interleaved
}

fn main() {
    type B = Wgpu;
    let device = WgpuDevice::default();

    println!("=== Quantized vs Float Model Comparison ===\n");

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

    // Use a short 2-second segment to keep comparison fast
    let start_sec = 30;
    let duration_sec = 30;
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

    // --- Load float model ---
    println!("\n--- Loading float model ---");
    let constants_path = "constants.safetensors";
    if !Path::new(constants_path).exists() {
        panic!("constants.safetensors not found");
    }
    let constants = load_constants::<B>(constants_path, &device);
    let model_float: MelBandRoformer<B> = MelBandRoformer::new(&config, &device, constants);

    let model_path = "model.pt";
    if !Path::new(model_path).exists() {
        panic!("model.pt not found");
    }
    let record = PyTorchFileRecorder::<FullPrecisionSettings>::new()
        .load(model_path.into(), &device)
        .expect("Failed to load float weights");
    let model_float = model_float.load_record(record);
    println!("Float model loaded.");

    // --- Load second copy for quantization ---
    println!("\n--- Loading second copy for quantization ---");
    let constants2 = load_constants::<B>(constants_path, &device);
    let model_q: MelBandRoformer<B> = MelBandRoformer::new(&config, &device, constants2);
    let record2 = PyTorchFileRecorder::<FullPrecisionSettings>::new()
        .load("model.pt".into(), &device)
        .expect("Failed to load float weights");
    let model_q = model_q.load_record(record2);

    // --- Quantize using burn's native quantization ---
    // Block-32 Q8S quantization (per-32-element block scaling).
    // Fusion must be disabled on burn-wgpu due to a reshape bug in the fusion backend.
    println!("\n--- Quantizing transformer weights (Q8S, block-32) ---");
    let scheme = QuantScheme::default()
        .with_value(QuantValue::Q8S)
        .with_level(QuantLevel::Block(BlockSize::new([32])))
        .with_param(QuantParam::F32);
    let mut quantizer = WeightQuantizer {
        calibration: Calibration::MinMax,
        scheme,
    };
    let model_quantized = model_q.quantize_transformers(&mut quantizer);
    println!("Quantization complete.");

    // --- Run float model (chunked) ---
    println!("\n--- Running float model (chunked, {}s chunks, {}% overlap) ---", 
        CHUNK_SIZE as f64 / 44100.0, 100 / NUM_OVERLAP);
    let t0 = std::time::Instant::now();
    let output_float_interleaved = demix_track::<B>(&model_float, &samples, 2, &device);
    let float_elapsed = t0.elapsed();
    let samples_per_channel = samples.len() / 2;
    println!("Float inference time: {:.2?}", float_elapsed);

    // --- Run quantized model (chunked) ---
    println!("\n--- Running quantized model (chunked) ---");
    let t0 = std::time::Instant::now();
    let output_quant_interleaved = demix_track::<B>(&model_quantized, &samples, 2, &device);
    let quant_elapsed = t0.elapsed();
    println!("Quantized inference time: {:.2?}", quant_elapsed);

    let audio_duration_sec = samples_per_channel as f64 / 44100.0;
    println!("\n=== Timing Summary ===");
    println!("Audio duration:       {:.2}s", audio_duration_sec);
    println!("Float inference:      {:.2?} ({:.2}x realtime)", float_elapsed, audio_duration_sec / float_elapsed.as_secs_f64());
    println!("Quantized inference:  {:.2?} ({:.2}x realtime)", quant_elapsed, audio_duration_sec / quant_elapsed.as_secs_f64());
    println!("Speedup:              {:.2}x", float_elapsed.as_secs_f64() / quant_elapsed.as_secs_f64());

    // --- Compare ---
    println!("\n=== Comparison Metrics ===\n");

    let output_float_t: Tensor<B, 1> = Tensor::from_floats(
        TensorData::new(output_float_interleaved.clone(), [output_float_interleaved.len()]),
        &device,
    );
    let output_quant_t: Tensor<B, 1> = Tensor::from_floats(
        TensorData::new(output_quant_interleaved.clone(), [output_quant_interleaved.len()]),
        &device,
    );

    let diff = output_float_t.clone().sub(output_quant_t.clone());

    let mae: f32 = diff.clone().abs().mean().into_scalar();
    println!("Mean Absolute Error (MAE):     {:.6e}", mae);

    let max_ae: f32 = diff.clone().abs().max().into_scalar();
    println!("Max Absolute Error:            {:.6e}", max_ae);

    let mse: f32 = diff.clone().powf_scalar(2.0).mean().into_scalar();
    println!("Mean Squared Error (MSE):      {:.6e}", mse);

    let rmse = mse.sqrt();
    println!("Root Mean Squared Error:       {:.6e}", rmse);

    let signal_power: f32 = output_float_t.clone().powf_scalar(2.0).mean().into_scalar();
    let noise_power = mse;
    let snr = if noise_power > 0.0 {
        10.0 * (signal_power / noise_power).log10()
    } else {
        f32::INFINITY
    };
    println!("Signal-to-Noise Ratio (SNR):   {:.2} dB", snr);

    let float_norm: f32 = output_float_t.clone().powf_scalar(2.0).sum().into_scalar().sqrt();
    let diff_norm: f32 = diff.clone().powf_scalar(2.0).sum().into_scalar().sqrt();
    let rel_error = if float_norm > 0.0 { diff_norm / float_norm } else { 0.0 };
    println!("Relative L2 Error:             {:.6e}", rel_error);

    let n = output_float_t.dims()[0] as f32;
    let mean_f: f32 = output_float_t.clone().mean().into_scalar();
    let mean_q: f32 = output_quant_t.clone().mean().into_scalar();
    let centered_f = output_float_t.clone().sub_scalar(mean_f);
    let centered_q = output_quant_t.clone().sub_scalar(mean_q);
    let cov: f32 = centered_f.clone().mul(centered_q.clone()).sum().into_scalar() / n;
    let std_f: f32 = (centered_f.clone().powf_scalar(2.0).sum().into_scalar() / n).sqrt();
    let std_q: f32 = (centered_q.powf_scalar(2.0).sum().into_scalar() / n).sqrt();
    let correlation = if std_f > 0.0 && std_q > 0.0 {
        cov / (std_f * std_q)
    } else {
        0.0
    };
    println!("Pearson Correlation:           {:.6}", correlation);

    melband_roformer::audio::save_audio("output_float.wav", &output_float_interleaved, 44100, 2).unwrap();
    melband_roformer::audio::save_audio("output_quantized.wav", &output_quant_interleaved, 44100, 2).unwrap();
    println!("\nSaved output_float.wav and output_quantized.wav for auditory comparison.");
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
