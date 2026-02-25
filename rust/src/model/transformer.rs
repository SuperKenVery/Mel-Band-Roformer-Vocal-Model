use super::attention::{Attention, FeedForward};
use super::rms_norm::RmsNorm;
use super::rotary::RotaryEmbedding;
use burn::module::Module;
use burn::prelude::*;
use burn::tensor::backend::Backend;

use burn::module::Param;

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    attention: Attention<B>,
    ff: FeedForward<B>,
    rotary_freqs: Param<Tensor<B, 1>>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(
        device: &B::Device,
        dim: usize,
        heads: usize,
        dim_head: usize,
        ff_mult: usize,
        attn_dropout: f32,
        ff_dropout: f32,
    ) -> Self {
        let rotary = RotaryEmbedding::new(dim_head);
        let freqs_tensor = Tensor::from_floats(rotary.inv_freqs.as_slice(), device);

        Self {
            attention: Attention::new(device, dim, heads, dim_head, attn_dropout),
            ff: FeedForward::new(device, dim, ff_mult, ff_dropout),
            rotary_freqs: Param::from_tensor(freqs_tensor),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let freqs: Vec<f32> = self.rotary_freqs.val().to_data().to_vec().unwrap();
        let rotary = RotaryEmbedding::from_freqs(freqs);

        let x_attn = self.attention.forward(x.clone(), Some(&rotary));
        
        // Debug print
        let data: Vec<f32> = x_attn.clone().into_data().to_vec().unwrap();
        let n = data.len() as f64;
        let mean: f64 = data.iter().map(|&v| v as f64).sum::<f64>() / n;
        let std: f64 = (data.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
        println!("    Attn: mean={:.6}, std={:.6}", mean, std);

        let x = x_attn + x;
        let x_ff = self.ff.forward(x.clone());

        // Debug print
         let data: Vec<f32> = x_ff.clone().into_data().to_vec().unwrap();
         let n = data.len() as f64;
         let mean: f64 = data.iter().map(|&v| v as f64).sum::<f64>() / n;
         let std: f64 = (data.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
         println!("    FF: mean={:.6}, std={:.6}", mean, std);
         
         let [b, s, d] = x_ff.dims();
         if b == 801 && s == 60 {
             if !std::path::Path::new("/tmp/rust_debug/layer0_freq_ff.bin").exists() {
                  let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes().to_vec()).collect();
                  std::fs::write("/tmp/rust_debug/layer0_freq_ff.bin", bytes).unwrap();
             }
         }
         if b == 60 && s == 801 {
             if !std::path::Path::new("/tmp/rust_debug/layer0_time_ff.bin").exists() {
                  let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes().to_vec()).collect();
                  std::fs::write("/tmp/rust_debug/layer0_time_ff.bin", bytes).unwrap();
             }
         }

         let x = x_ff + x;
        x
    }
}

#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    layers: Vec<TransformerBlock<B>>,
    norm: RmsNorm<B>,
}

impl<B: Backend> Transformer<B> {
    pub fn new(
        device: &B::Device,
        dim: usize,
        depth: usize,
        heads: usize,
        dim_head: usize,
        ff_mult: usize,
        attn_dropout: f32,
        ff_dropout: f32,
    ) -> Self {
        let layers = (0..depth)
            .map(|_| {
                TransformerBlock::new(device, dim, heads, dim_head, ff_mult, attn_dropout, ff_dropout)
            })
            .collect();

        Self {
            layers,
            norm: RmsNorm::<B>::new(device, dim),
        }
    }

    pub fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        for layer in &self.layers {
            x = layer.forward(x);
        }
        let x_norm = self.norm.forward(x);
        
        let [b, s, _d] = x_norm.dims();
        // Time Transformer Output
        if b == 60 && s == 801 {
             let data: Vec<f32> = x_norm.clone().into_data().to_vec().unwrap();
             if !std::path::Path::new("/tmp/rust_debug/layer0_time_output.bin").exists() {
                  let bytes: Vec<u8> = data.iter().flat_map(|f: &f32| f.to_le_bytes().to_vec()).collect();
                  std::fs::write("/tmp/rust_debug/layer0_time_output.bin", bytes).unwrap();
             }
             
             // Save Gamma
             let gamma = self.norm.gamma.val();
             let data: Vec<f32> = gamma.clone().into_data().to_vec().unwrap();
             if !std::path::Path::new("/tmp/rust_debug/layer0_time_norm_gamma.bin").exists() {
                  let bytes: Vec<u8> = data.iter().flat_map(|f: &f32| f.to_le_bytes().to_vec()).collect();
                  std::fs::write("/tmp/rust_debug/layer0_time_norm_gamma.bin", bytes).unwrap();
             }
        }
        
        x_norm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_transformer_shape() {
        let device = Default::default();
        let transformer =
            Transformer::<TestBackend>::new(&device, 384, 2, 8, 64, 4, 0.0, 0.0);

        let input = Tensor::zeros([2, 100, 384], &device);
        let output = transformer.forward(input);

        assert_eq!(output.dims(), [2, 100, 384]);
    }
}
