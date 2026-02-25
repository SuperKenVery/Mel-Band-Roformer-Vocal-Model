
use burn::module::Module;
use burn::record::{Recorder, NamedMpkFileRecorder, FullPrecisionSettings};
use burn_store::BurnpackStore; 
use burn_store::ModuleSnapshot;

use burn_ndarray::NdArray;
use mel_band_roformer::model::mel_band_roformer::MelBandRoformer;
use mel_band_roformer::io::weights::load_config_for_model;
use std::path::Path;

type B = NdArray<f32>;

fn main() {
    let model_path = Path::new("./model.bpk");
    let config_path = Path::new("../configs/config_vocals_mel_band_roformer.yaml");
    
    println!("Loading config from {:?}", config_path);
    let config = load_config_for_model(model_path, Some(config_path)).unwrap();
    let device = Default::default();
    
    println!("Initializing model...");
    let mut model = MelBandRoformer::<B>::new(&device, config);
    
    println!("Loading weights from {:?}", model_path);
    let mut store = BurnpackStore::from_file(model_path);
    model.load_from(&mut store).expect("Failed to load model");
    
    let record = model.into_record();
    
    println!("Saving to model.mpk...");
    NamedMpkFileRecorder::<FullPrecisionSettings>::new()
        .record(record, "model.mpk".into())
        .expect("Failed to save mpk");
        
    println!("Done!");
}
