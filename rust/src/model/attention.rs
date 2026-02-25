use super::rms_norm::RmsNorm;
use super::rotary::RotaryEmbedding;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    norm: RmsNorm<B>,
    to_qkv: Linear<B>,
    to_gates: Linear<B>,
    to_out: Linear<B>,
    dropout: Dropout,
    heads: usize,
    dim_head: usize,
    scale: f32,
}

impl<B: Backend> Attention<B> {
    pub fn new(device: &B::Device, dim: usize, heads: usize, dim_head: usize, dropout: f32) -> Self {
        let dim_inner = heads * dim_head;

        let to_qkv = LinearConfig::new(dim, dim_inner * 3)
            .with_bias(false)
            .init(device);
        let to_gates = LinearConfig::new(dim, heads).init(device);
        let to_out = LinearConfig::new(dim_inner, dim)
            .with_bias(false)
            .init(device);
        let dropout = DropoutConfig::new(dropout as f64).init();
        let norm = RmsNorm::<B>::new(device, dim);

        Self {
            norm,
            to_qkv,
            to_gates,
            to_out,
            dropout,
            heads,
            dim_head,
            scale: (dim_head as f32).powf(-0.5),
        }
    }

    /// Forward pass for attention
    /// Input: [batch, seq_len, dim]
    /// Output: [batch, seq_len, dim]
    pub fn forward(&self, x: Tensor<B, 3>, rotary_embed: Option<&RotaryEmbedding>) -> Tensor<B, 3> {
        let [batch, seq_len, _dim] = x.dims();

        // Save Input to Attn (Pre-Norm)
        if batch == 801 && seq_len == 60 {
             if !std::path::Path::new("/tmp/rust_debug/layer0_freq_attn_input.bin").exists() {
                  let data: Vec<f32> = x.clone().into_data().to_vec().unwrap();
                  let bytes: Vec<u8> = data.iter().flat_map(|f: &f32| f.to_le_bytes().to_vec()).collect();
                  std::fs::write("/tmp/rust_debug/layer0_freq_attn_input.bin", bytes).unwrap();
             }
        }

        let x_norm = self.norm.forward(x.clone());
        
        // Save Attn Norm Output
        let [b, s, _d] = x_norm.dims();
        if b == 801 && s == 60 {
             if !std::path::Path::new("/tmp/rust_debug/layer0_freq_attn_norm.bin").exists() {
                  let data: Vec<f32> = x_norm.clone().into_data().to_vec().unwrap();
                  let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes().to_vec()).collect();
                  std::fs::write("/tmp/rust_debug/layer0_freq_attn_norm.bin", bytes).unwrap();
             }
             
             // Save Gamma
             let gamma = self.norm.gamma.val();
             let data: Vec<f32> = gamma.clone().into_data().to_vec().unwrap();
             let n = data.len() as f64;
             let mean: f64 = data.iter().map(|&v| v as f64).sum::<f64>() / n;
             let std: f64 = (data.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
             println!("      Freq Attn Norm Gamma: shape={:?}, mean={:.6}, std={:.6}", gamma.dims(), mean, std);
             
             if !std::path::Path::new("/tmp/rust_debug/layer0_freq_attn_norm_gamma.bin").exists() {
                  let bytes: Vec<u8> = data.iter().flat_map(|f: &f32| f.to_le_bytes().to_vec()).collect();
                  std::fs::write("/tmp/rust_debug/layer0_freq_attn_norm_gamma.bin", bytes).unwrap();
             }
        }

        let qkv = self.to_qkv.forward(x_norm.clone());
         if b == 801 && s == 60 {
              let w = self.to_qkv.weight.val();
              let data: Vec<f32> = w.clone().into_data().to_vec().unwrap();
              let n = data.len() as f64;
              let mean: f64 = data.iter().map(|&v| v as f64).sum::<f64>() / n;
              let std: f64 = (data.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
              println!("      Freq Attn QKV Weight: shape={:?}, mean={:.6}, std={:.6}", w.dims(), mean, std);
              
              if !std::path::Path::new("/tmp/rust_debug/layer0_freq_attn_to_qkv.bin").exists() {
                   let data: Vec<f32> = qkv.clone().into_data().to_vec().unwrap();
                   let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes().to_vec()).collect();
                   std::fs::write("/tmp/rust_debug/layer0_freq_attn_to_qkv.bin", bytes).unwrap();
              }
         }
         let qkv = qkv.reshape([batch, seq_len, 3, self.heads, self.dim_head]);
        let qkv = qkv.swap_dims(2, 3);

        let q: Tensor<B, 4> = qkv
            .clone()
            .slice([0..batch, 0..seq_len, 0..self.heads, 0..1, 0..self.dim_head])
            .reshape([batch, seq_len, self.heads, self.dim_head]);
        let k: Tensor<B, 4> = qkv
            .clone()
            .slice([0..batch, 0..seq_len, 0..self.heads, 1..2, 0..self.dim_head])
            .reshape([batch, seq_len, self.heads, self.dim_head]);
        let v: Tensor<B, 4> = qkv
            .slice([0..batch, 0..seq_len, 0..self.heads, 2..3, 0..self.dim_head])
            .reshape([batch, seq_len, self.heads, self.dim_head]);

        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // Save Q
        if b == 801 && s == 60 {
             if !std::path::Path::new("/tmp/rust_debug/layer0_freq_attn_q.bin").exists() {
                  let data: Vec<f32> = q.clone().into_data().to_vec().unwrap();
                  let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes().to_vec()).collect();
                  std::fs::write("/tmp/rust_debug/layer0_freq_attn_q.bin", bytes).unwrap();
             }
        }

        let (q, k) = if let Some(rope) = rotary_embed {
            (
                rope.rotate_queries_or_keys(q, 0),
                rope.rotate_queries_or_keys(k, 0),
            )
        } else {
            (q, k)
        };

        // Save Q Rotated
        if b == 801 && s == 60 {
             if !std::path::Path::new("/tmp/rust_debug/layer0_freq_attn_q_rotated.bin").exists() {
                  let data: Vec<f32> = q.clone().into_data().to_vec().unwrap();
                  let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes().to_vec()).collect();
                  std::fs::write("/tmp/rust_debug/layer0_freq_attn_q_rotated.bin", bytes).unwrap();
             }
        }

        let attn_weights = q.matmul(k.swap_dims(2, 3)).mul_scalar(self.scale);
        let attn_weights = burn::tensor::activation::softmax(attn_weights, 3);

        // Save Attn Weights
        if b == 801 && s == 60 {
             if !std::path::Path::new("/tmp/rust_debug/layer0_freq_attn_weights.bin").exists() {
                  let data: Vec<f32> = attn_weights.clone().into_data().to_vec().unwrap();
                  let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes().to_vec()).collect();
                  std::fs::write("/tmp/rust_debug/layer0_freq_attn_weights.bin", bytes).unwrap();
             }
        }

        let out = attn_weights.matmul(v);

        let gates = self.to_gates.forward(x_norm);
        let gates = burn::tensor::activation::sigmoid(gates);
        let gates = gates.reshape([batch, seq_len, self.heads, 1]);
        let gates = gates.swap_dims(1, 2);

        let out = out * gates;

        let out = out.swap_dims(1, 2);
        let out = out.reshape([batch, seq_len, self.heads * self.dim_head]);

        let out = self.dropout.forward(self.to_out.forward(out));

        let [b, s, _d] = out.dims();
        if b == 801 && s == 60 {
             if !std::path::Path::new("/tmp/rust_debug/layer0_freq_attn.bin").exists() {
                  let data: Vec<f32> = out.clone().into_data().to_vec().unwrap();
                  let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes().to_vec()).collect();
                  std::fs::write("/tmp/rust_debug/layer0_freq_attn.bin", bytes).unwrap();
             }
        }
        if b == 60 && s == 801 {
             if !std::path::Path::new("/tmp/rust_debug/layer0_time_attn.bin").exists() {
                  let data: Vec<f32> = out.clone().into_data().to_vec().unwrap();
                  let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes().to_vec()).collect();
                  std::fs::write("/tmp/rust_debug/layer0_time_attn.bin", bytes).unwrap();
             }
        }

        out
    }
}

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    norm: RmsNorm<B>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(device: &B::Device, dim: usize, mult: usize, dropout: f32) -> Self {
        let dim_inner = dim * mult;

        Self {
            norm: RmsNorm::new(device, dim),
            linear1: LinearConfig::new(dim, dim_inner).init(device),
            linear2: LinearConfig::new(dim_inner, dim).init(device),
            dropout: DropoutConfig::new(dropout as f64).init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.norm.forward(x);
        // Debug print for Norm
        let data: Vec<f32> = x.clone().into_data().to_vec().unwrap();
        let n = data.len() as f64;
        let mean: f64 = data.iter().map(|&v| v as f64).sum::<f64>() / n;
        let std: f64 = (data.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
        println!("      FF Norm: mean={:.6}, std={:.6}", mean, std);
 
         let [b, s, _d] = x.dims();
         if b == 801 && s == 60 {
             if !std::path::Path::new("/tmp/rust_debug/layer0_freq_ff_norm.bin").exists() {
                  let data: Vec<f32> = x.clone().into_data().to_vec().unwrap();
                  let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes().to_vec()).collect();
                  std::fs::write("/tmp/rust_debug/layer0_freq_ff_norm.bin", bytes).unwrap();
             }
         }

         let x = self.linear1.forward(x);
         // Debug print for Linear1
         let data: Vec<f32> = x.clone().into_data().to_vec().unwrap();
         let n = data.len() as f64;
         let mean: f64 = data.iter().map(|&v| v as f64).sum::<f64>() / n;
         let std: f64 = (data.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
         println!("      FF Linear1: mean={:.6}, std={:.6}", mean, std);

         if b == 801 && s == 60 {
             let w = self.linear1.weight.val();
             let data: Vec<f32> = w.clone().into_data().to_vec().unwrap();
             let n = data.len() as f64;
             let mean: f64 = data.iter().map(|&v| v as f64).sum::<f64>() / n;
             let std: f64 = (data.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
             println!("      FF Linear1 Weight: shape={:?}, mean={:.6}, std={:.6}", w.dims(), mean, std);

             let b_param = self.linear1.bias.as_ref().unwrap().val();
             let data: Vec<f32> = b_param.clone().into_data().to_vec().unwrap();
             let n = data.len() as f64;
             let mean: f64 = data.iter().map(|&v| v as f64).sum::<f64>() / n;
             let std: f64 = (data.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
             println!("      FF Linear1 Bias: shape={:?}, mean={:.6}, std={:.6}", b_param.dims(), mean, std);

             if !std::path::Path::new("/tmp/rust_debug/layer0_freq_ff_linear1.bin").exists() {
                  let data: Vec<f32> = x.clone().into_data().to_vec().unwrap();
                  let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes().to_vec()).collect();
                  std::fs::write("/tmp/rust_debug/layer0_freq_ff_linear1.bin", bytes).unwrap();
             }
         }

        let x = burn::tensor::activation::gelu(x);
        let x = self.dropout.forward(x);
        let x = self.linear2.forward(x);
        self.dropout.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_attention_shape() {
        let device = Default::default();
        let attn = Attention::<TestBackend>::new(&device, 384, 8, 64, 0.0);

        let input = Tensor::zeros([2, 100, 384], &device);
        let output = attn.forward(input, None);

        assert_eq!(output.dims(), [2, 100, 384]);
    }

    #[test]
    fn test_feedforward_shape() {
        let device = Default::default();
        let ff = FeedForward::<TestBackend>::new(&device, 384, 4, 0.0);

        let input = Tensor::zeros([2, 100, 384], &device);
        let output = ff.forward(input);

        assert_eq!(output.dims(), [2, 100, 384]);
    }
}
