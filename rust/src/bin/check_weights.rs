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
    
    println!("Checking mask_estimator weights:");
    
    let me = &model.mask_estimators[0];
    let band0 = &me.to_freqs[0];
    
    let w1: Vec<f32> = band0.mlp.linear1.weight.val().into_data().to_vec().unwrap();
    let b1: Vec<f32> = band0.mlp.linear1.bias.as_ref().unwrap().val().into_data().to_vec().unwrap();
    
    println!("linear1:");
    println!("  weight shape: ({}, {})", band0.mlp.linear1.weight.shape().dims[0], band0.mlp.linear1.weight.shape().dims[1]);
    println!("  weight mean: {:.6}", w1.iter().map(|&x| x as f64).sum::<f64>() / w1.len() as f64);
    println!("  weight std: {:.6}", (w1.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / w1.len() as f64).sqrt());
    println!("  bias mean: {:.6}", b1.iter().map(|&x| x as f64).sum::<f64>() / b1.len() as f64);
    
    let w3: Vec<f32> = band0.mlp.linear3.weight.val().into_data().to_vec().unwrap();
    let b3: Vec<f32> = band0.mlp.linear3.bias.as_ref().unwrap().val().into_data().to_vec().unwrap();
    
    println!("linear3:");
    println!("  weight shape: ({}, {})", band0.mlp.linear3.weight.shape().dims[0], band0.mlp.linear3.weight.shape().dims[1]);
    println!("  weight mean: {:.6}", w3.iter().map(|&x| x as f64).sum::<f64>() / w3.len() as f64);
    println!("  weight std: {:.6}", (w3.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / w3.len() as f64).sqrt());
    println!("  bias len: {}", b3.len());
    println!("  bias mean: {:.6}", b3.iter().map(|&x| x as f64).sum::<f64>() / b3.len() as f64);
    println!("  bias first 10: {:?}", &b3[..10.min(b3.len())]);
}
