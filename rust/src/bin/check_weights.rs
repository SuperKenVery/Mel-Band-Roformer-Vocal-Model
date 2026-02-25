use burn::backend::wgpu::WgpuDevice;
use burn::backend::Metal;
use mel_band_roformer::io::weights::load_config_for_model;
use mel_band_roformer::model::MelBandRoformer;
use burn::prelude::*;
use burn::module::Module;
use burn_store::{BurnpackStore, ModuleSnapshot};
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
    
    println!("Checking band_split weights:");
    
    // Check first band
    let band0 = &model.band_split.to_features[0];
    
    // Norm Gamma
    let gamma: Vec<f32> = band0.norm.gamma.val().into_data().to_vec().unwrap();
    println!("Band 0 Norm Gamma:");
    println!("  shape: ({},)", band0.norm.gamma.shape().dims[0]);
    println!("  mean: {:.6}", gamma.iter().map(|&x| x as f64).sum::<f64>() / gamma.len() as f64);
    println!("  std: {:.6}", (gamma.iter().map(|&x| (x as f64 - 1.0).powi(2)).sum::<f64>() / gamma.len() as f64).sqrt());
    
    // Linear Weight
    let w: Vec<f32> = band0.linear.weight.val().into_data().to_vec().unwrap();
    println!("Band 0 Linear Weight:");
    println!("  shape: ({}, {})", band0.linear.weight.shape().dims[0], band0.linear.weight.shape().dims[1]);
    println!("  mean: {:.6}", w.iter().map(|&x| x as f64).sum::<f64>() / w.len() as f64);
    println!("  std: {:.6}", (w.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / w.len() as f64).sqrt());
    
    // Linear Bias
    if let Some(bias) = &band0.linear.bias {
        let b: Vec<f32> = bias.val().into_data().to_vec().unwrap();
        println!("Band 0 Linear Bias:");
        println!("  shape: ({},)", bias.shape().dims[0]);
        println!("  mean: {:.6}", b.iter().map(|&x| x as f64).sum::<f64>() / b.len() as f64);
    }
    
    // Check last band
    let band59 = &model.band_split.to_features[59];
    
    // Norm Gamma
    let gamma: Vec<f32> = band59.norm.gamma.val().into_data().to_vec().unwrap();
    println!("Band 59 Norm Gamma:");
    println!("  shape: ({},)", band59.norm.gamma.shape().dims[0]);
    println!("  mean: {:.6}", gamma.iter().map(|&x| x as f64).sum::<f64>() / gamma.len() as f64);
    
    // Linear Weight
    let w: Vec<f32> = band59.linear.weight.val().into_data().to_vec().unwrap();
    println!("Band 59 Linear Weight:");
    println!("  shape: ({}, {})", band59.linear.weight.shape().dims[0], band59.linear.weight.shape().dims[1]);
    println!("  mean: {:.6}", w.iter().map(|&x| x as f64).sum::<f64>() / w.len() as f64);
    
    // Linear Bias
    if let Some(bias) = &band59.linear.bias {
        let b: Vec<f32> = bias.val().into_data().to_vec().unwrap();
        println!("Band 59 Linear Bias:");
        println!("  shape: ({},)", bias.shape().dims[0]);
        println!("  mean: {:.6}", b.iter().map(|&x| x as f64).sum::<f64>() / b.len() as f64);
    }
}
