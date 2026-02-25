#![recursion_limit = "256"]

use burn::backend::wgpu::WgpuDevice;
use burn::backend::Metal;
use mel_band_roformer::io::wav::read_wav;
use mel_band_roformer::io::weights::load_config_for_model;
use mel_band_roformer::model::{MelBandRoformer, MelFilterBank};
use mel_band_roformer::stft::{FftPlan, stft};
use burn::prelude::*;
use burn::module::Module;
use burn_store::{BurnpackStore, ModuleSnapshot};
use std::path::Path;

type B = Metal;

fn tensor_stats<const D: usize>(name: &str, t: &Tensor<B, D>) {
    let data: Vec<f32> = t.clone().into_data().to_vec().unwrap();
    let n = data.len() as f64;
    let mean: f64 = data.iter().map(|&x| x as f64).sum::<f64>() / n;
    let std: f64 = (data.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
    println!("  {}: shape={:?}, mean={:.6}, std={:.6}", name, t.dims(), mean, std);
}

fn main() {
    std::fs::create_dir_all("/tmp/rust_debug").unwrap();
    
    let model_path = Path::new("./model.bpk");
    let config_path = Path::new("../configs/config_vocals_mel_band_roformer.yaml");
    
    let mut config = load_config_for_model(model_path, Some(config_path)).unwrap();
    // Disable dropout for deterministic comparison
    config.attn_dropout = 0.0;
    config.ff_dropout = 0.0;
    println!("Config: depth={}, dim={}", config.depth, config.dim);
    
    let mel_filter_bank_check = MelFilterBank::new(
        config.sample_rate,
        config.stft_n_fft,
        config.num_bands,
        config.stereo,
    );
    let freqs_per_bands = mel_filter_bank_check.freqs_per_bands_with_complex(config.audio_channels());
    println!("freqs_per_bands first 10: {:?}", &freqs_per_bands[..10]);
    println!("freqs_per_bands total: {}", freqs_per_bands.iter().sum::<usize>());
    
    for (i, &d) in freqs_per_bands.iter().enumerate() {
        if d == config.dim {
             println!("WARNING: Band {} has dim_in == dim == {}. Potential weight loading issue!", i, d);
        }
    }
    
    let device = WgpuDevice::default();
    let mut model = MelBandRoformer::<B>::new(&device, config.clone());
    
    let mut store = BurnpackStore::from_file(model_path);
    model.load_from(&mut store).expect("Failed to load model");
    // model.fix_load_weights();
    
    let (samples, sr) = read_wav(Path::new(
        "/Users/bytedance/Desktop/music-instrumental-extract/inputs/syws.wav"
    )).unwrap();
    
    println!("Loaded audio: {} samples, {} Hz, {} channels", samples[0].len(), sr, samples.len());
    
    let start_sec = 35;
    let chunk_size = 352800;
    let start_sample = start_sec * sr as usize;
    
    let chunk: Vec<Vec<f32>> = samples.iter()
        .map(|ch| ch[start_sample..start_sample + chunk_size].to_vec())
        .collect();
    
    println!("Chunk: {} samples per channel", chunk[0].len());
    
    // Run through model manually to get intermediates
    let plan = FftPlan::new(
        config.stft_n_fft,
        config.stft_hop_length,
        config.stft_win_length,
    );
    
    let stft_repr = stft(&chunk, &plan);
    let num_channels = stft_repr.len();
    let freq_bins = stft_repr[0].len();
    let time_frames = stft_repr[0][0].len();
    
    println!("STFT shape: [{}, {}, {}]", num_channels, freq_bins, time_frames);
    
    let mel_filter_bank = MelFilterBank::new(
        config.sample_rate,
        config.stft_n_fft,
        config.num_bands,
        config.stereo,
    );
    
    // Build x_before_band_split (matching Python's layout)
    let total_freqs = freq_bins * num_channels;
    let mut stft_flat: Vec<f32> = Vec::with_capacity(time_frames * total_freqs * 2);
    for t in 0..time_frames {
        for f in 0..freq_bins {
            for ch in 0..num_channels {
                stft_flat.push(stft_repr[ch][f][t].re);
                stft_flat.push(stft_repr[ch][f][t].im);
            }
        }
    }
    
    let mut gathered: Vec<f32> = Vec::new();
    for t in 0..time_frames {
        for &freq_idx in &mel_filter_bank.freq_indices {
            let base = t * total_freqs * 2 + freq_idx * 2;
            gathered.push(stft_flat[base]);
            gathered.push(stft_flat[base + 1]);
        }
    }
    
    let gathered_len = mel_filter_bank.freq_indices.len() * 2;
    let batch = 1;
    
    let x: Tensor<B, 3> = Tensor::<B, 1>::from_floats(gathered.as_slice(), &device)
        .reshape([batch, time_frames, gathered_len]);
    
    println!("Rust intermediates:");
    tensor_stats("x_before_band_split", &x);
    
    // Run band_split directly
    let mut x = model.band_split.forward(x);
    tensor_stats("x_after_band_split", &x);
    
    let [batch, time, num_bands, dim] = x.dims();
    
    for (i, (time_transformer, freq_transformer)) in
        model.time_transformers.iter().zip(model.freq_transformers.iter()).enumerate()
    {
        let x_time = x.clone().swap_dims(1, 2);
        let x_time = x_time.reshape([batch * num_bands, time, dim]);
        let x_time = time_transformer.forward(x_time);
        let x_time = x_time.reshape([batch, num_bands, time, dim]);
        x = x_time.swap_dims(1, 2);

        let x_freq = x.clone().reshape([batch * time, num_bands, dim]);
        let x_freq = freq_transformer.forward(x_freq);
        x = x_freq.reshape([batch, time, num_bands, dim]);
        
        println!("After layer {}:", i);
        tensor_stats(&format!("x_after_layer_{}", i), &x);
        
        if i == 0 {
             let data: Vec<f32> = x.clone().into_data().to_vec().unwrap();
             // Save as raw f32 bytes
             let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes().to_vec()).collect();
             std::fs::write("/tmp/rust_debug/x_after_layer_0.bin", bytes).unwrap();
        }
    }
    
    tensor_stats("x_before_mask", &x);

    let num_stems = model.mask_estimators.len();
    let mut all_masks: Vec<Tensor<B, 3>> = Vec::with_capacity(num_stems);

    for (i, mask_estimator) in model.mask_estimators.iter().enumerate() {
        let mask = mask_estimator.forward(x.clone());
        tensor_stats(&format!("mask_{}", i), &mask);
        all_masks.push(mask);
    }
    
    // Reconstruct output (simplified for single stem/batch if needed, or full logic)
    // We can just call model.forward(&chunk) for the final output as before, 
    // or reimplement the scattering logic here to verify that part too.
    // For now, let's stick to checking if the divergence happens before masks.
    
    // Run full model forward pass to compare final output
    let output = model.forward(&chunk);
    
    println!("Rust single-chunk output:");
    println!("  shape: [{}, {}, {}]", output.len(), output[0].len(), output[0][0].len());
    let mean: f64 = output.iter().flatten().flatten().map(|&x| x as f64).sum::<f64>() / 
                    (output.len() * output[0].len() * output[0][0].len()) as f64;
    let std: f64 = (output.iter().flatten().flatten()
        .map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / 
        (output.len() * output[0].len() * output[0][0].len()) as f64).sqrt();
    println!("  mean: {:.6}", mean);
    println!("  std: {:.6}", std);
    println!("  first 10: {:?}", &output[0][0][..10]);
    
    // Save for later comparison  
    std::fs::write("/tmp/rust_debug/rust_output_first100.txt", 
        output[0][0][..100].iter().map(|x| format!("{}", x)).collect::<Vec<_>>().join("\n")
    ).unwrap();
}
