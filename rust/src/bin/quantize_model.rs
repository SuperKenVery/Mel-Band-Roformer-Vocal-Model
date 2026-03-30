#![recursion_limit = "512"]

use burn::tensor::backend::Backend;
use burn_wgpu::{Wgpu, WgpuDevice};
use melband_roformer::model::mel_band_roformer::{MelBandRoformer, MelBandRoformerConfig, MelBandConstants};
use std::fs::File;
use safetensors::SafeTensors;
use memmap2::MmapOptions;
use burn::tensor::{Tensor, TensorData, Int};
use burn::module::Module;
use burn::record::{Recorder, FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::quantization::{Calibration, QuantScheme, QuantValue, QuantLevel, QuantParam, BlockSize};
use melband_roformer::model::weight_quantizer::WeightQuantizer;
use burn_import::pytorch::PyTorchFileRecorder;
use std::path::Path;

fn main() {
    type B = Wgpu;
    let device = WgpuDevice::default();

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

    println!("Initializing float model...");
    let model_float: MelBandRoformer<B> = MelBandRoformer::new(&config, &device, constants);

    println!("Loading weights from model.pt...");
    let model_path = "model.pt";
    if !Path::new(model_path).exists() {
        panic!("model.pt not found");
    }

    let record = PyTorchFileRecorder::<FullPrecisionSettings>::new()
        .load(model_path.into(), &device)
        .expect("Failed to load weights from model.pt");

    let model_float = model_float.load_record(record);

    println!("Quantizing transformer weights (Int8 symmetric, block-32)...");
    let scheme = QuantScheme::default()
        .with_value(QuantValue::Q8S)
        .with_level(QuantLevel::Block(BlockSize::new([32])))
        .with_param(QuantParam::F32);
    let mut quantizer = WeightQuantizer {
        calibration: Calibration::MinMax,
        scheme,
    };
    let model_quantized = model_float.quantize_transformers(&mut quantizer);

    println!("Saving quantized model to model_quantized.mpk...");
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model_quantized
        .save_file("model_quantized", &recorder)
        .expect("Failed to save quantized model");

    println!("Done!");
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
