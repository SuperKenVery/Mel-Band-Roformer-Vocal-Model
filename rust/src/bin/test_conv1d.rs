use burn::{
    tensor::{Tensor, TensorData, Shape},
    nn::conv::Conv1dConfig,
    module::Param,
};
use burn_wgpu::{Wgpu, WgpuDevice};

type B = Wgpu<f32, i32>;

fn main() {
    let device = WgpuDevice::BestAvailable;
    let n_fft = 2048;
    let win_length = 2048;
    let num_freqs = n_fft / 2 + 1;
    
    // Create dummy kernel (ones)
    let kernel_data = vec![1.0; num_freqs * 1 * win_length];
    let kernel = Tensor::<B, 3>::from_data(TensorData::new(kernel_data, Shape::new([num_freqs, 1, win_length])), &device);
    
    let config = Conv1dConfig::new(1, num_freqs, win_length)
        .with_stride(1)
        .with_bias(false)
        .with_padding(burn::nn::PaddingConfig1d::Valid);
        
    let mut conv = config.init(&device);
    conv.weight = Param::from_tensor(kernel);
    
    // Input: ones
    // [2, 1, 30000] via cat
    let t1 = Tensor::<B, 3>::ones([2, 1, 15000], &device);
    let t2 = Tensor::<B, 3>::ones([2, 1, 15000], &device);
    let input = Tensor::cat(vec![t1, t2], 2);
    
    let output = conv.forward(input);
    
    let d = output.into_data();
    let v = d.to_vec::<f32>().unwrap();
    let max = v.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    
    println!("Test Conv1d gain with ones kernel");
    println!("Output max: {}", max);
    println!("Expected max: 2048.0");
}
