#![recursion_limit = "256"]

use burn::backend::wgpu::WgpuDevice;
use burn::backend::Metal;
use burn::prelude::*;
use burn::module::Module;
use burn_store::{BurnpackStore, ModuleSnapshot};
use mel_band_roformer::io::weights::load_config_for_model;
use mel_band_roformer::model::MelBandRoformer;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

type B = Metal;

fn main() {
    let model_path = Path::new("./model.bpk");
    let config_path = Path::new("../configs/config_vocals_mel_band_roformer.yaml");
    
    let config = load_config_for_model(model_path, Some(config_path)).unwrap();
    
    let device = WgpuDevice::default();
    let mut model = MelBandRoformer::<B>::new(&device, config.clone());
    
    let mut store = BurnpackStore::from_file(model_path);
    model.load_from(&mut store).expect("Failed to load model");
    
    let input_data = load_npy("/tmp/test_mlp_input.npy");
    let [rows, cols] = [10usize, 384usize];
    
    println!("Input data len: {}", input_data.len());
    println!("Input first 5: {:?}", &input_data[..5]);
    
    let input: Tensor<B, 2> = Tensor::<B, 1>::from_floats(input_data.as_slice(), &device)
        .reshape([rows, cols]);
    
    println!("Input tensor shape: {:?}", input.dims());
    
    let me = &model.mask_estimators[0];
    let band0 = &me.to_freqs[0];
    
    let output = band0.forward(input);
    
    let output_data: Vec<f32> = output.clone().into_data().to_vec().unwrap();
    let n = output_data.len() as f64;
    let mean: f64 = output_data.iter().map(|&v| v as f64).sum::<f64>() / n;
    let std: f64 = (output_data.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
    
    println!("Rust output:");
    println!("  shape: {:?}", output.dims());
    println!("  mean: {:.6}", mean);
    println!("  std: {:.6}", std);
    println!("  first row first 10: {:?}", &output_data[..10]);
}

fn load_npy(path: &str) -> Vec<f32> {
    use std::io::Read;
    let mut file = File::open(path).expect("Failed to open npy file");
    
    let mut magic = [0u8; 6];
    file.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, b"\x93NUMPY", "Not a valid NPY file");
    
    let mut version = [0u8; 2];
    file.read_exact(&mut version).unwrap();
    
    let header_len = if version[0] == 1 {
        let mut buf = [0u8; 2];
        file.read_exact(&mut buf).unwrap();
        u16::from_le_bytes(buf) as usize
    } else {
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf).unwrap();
        u32::from_le_bytes(buf) as usize
    };
    
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header).unwrap();
    
    let mut data = Vec::new();
    file.read_to_end(&mut data).unwrap();
    
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
