use burn::{
    tensor::{Tensor, TensorData, Shape},
    nn::conv::ConvTranspose1dConfig,
    module::Param,
};
use burn_wgpu::{Wgpu, WgpuDevice};

type B = Wgpu<f32, i32>;

fn main() {
    let device = WgpuDevice::BestAvailable;
    let n_fft = 2048;
    let win_length = 2048;
    let hop_length = 441;
    let num_freqs = n_fft / 2 + 1;
    
    // Create dummy kernel (ones)
    // To see if overlap add works
    let kernel_data = vec![1.0; num_freqs * 1 * win_length];
    let kernel = Tensor::<B, 3>::from_data(TensorData::new(kernel_data, Shape::new([num_freqs, 1, win_length])), &device);
    
    let config = ConvTranspose1dConfig::new([num_freqs, 1], win_length)
        .with_stride(hop_length)
        .with_bias(false)
        .with_padding(0);
    let mut conv = config.init(&device);
    conv.weight = Param::from_tensor(kernel);
    
    // Input: ones
    // [1, num_freqs, 10]
    let time_frames = 10;
    let input = Tensor::<B, 3>::ones([1, num_freqs, time_frames], &device);
    
    let output = conv.forward(input);
    
    // Output should be roughly: num_freqs * overlap_factor
    // overlap_factor = win / hop = 2048 / 441 ~= 4.64
    // num_freqs = 1025
    // expected = 1025 * 4.64 ~= 4756
    
    let d = output.into_data();
    let v = d.to_vec::<f32>().unwrap();
    let max = v.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    
    println!("Test ISTFT gain with ones kernel");
    println!("Input shape: [1, {}, {}]", num_freqs, time_frames);
    println!("Output max: {}", max);
    println!("Expected max (approx): {}", 1025.0 * 2048.0 / 441.0);
}
