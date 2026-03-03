use burn::{
    config::Config,
    module::Module,
    tensor::{backend::Backend, Tensor},
};

#[derive(Config, Debug)]
pub struct RotaryEmbeddingConfig {
    pub dim: usize,
    #[config(default = 10000.0)]
    pub theta: f64,
}

#[derive(Module, Debug)]
pub struct RotaryEmbedding<B: Backend> {
    inv_freq: Tensor<B, 1>,
    theta: f64,
}

impl<B: Backend> RotaryEmbedding<B> {
    pub fn new(config: &RotaryEmbeddingConfig, device: &B::Device) -> Self {
        let dim = config.dim;
        let theta = config.theta;
        
        let freq_seq_len = dim / 2;
        let inv_freq_vals = (0..freq_seq_len)
            .map(|i| 1.0 / (theta.powf((2 * i) as f64 / dim as f64)))
            .map(|x| x as f32)
            .collect::<Vec<_>>();
            
        let inv_freq = Tensor::from_floats(inv_freq_vals.as_slice(), device);

        Self {
            inv_freq,
            theta,
        }
    }

    fn rotate_half(x: Tensor<B, 4>) -> Tensor<B, 4> {
        // x shape: [batch, heads, seq_len, dim]
        let dims = x.dims();
        let batch = dims[0];
        let heads = dims[1];
        let seq_len = dims[2];
        let dim = dims[3];
        let half_dim = dim / 2;
        
        // Reshape to separate pairs: [batch, heads, seq_len, half_dim, 2]
        let x = x.reshape([batch, heads, seq_len, half_dim, 2]);
        
        // Split into x1 (even indices) and x2 (odd indices)
        let x1 = x.clone().slice([0..batch, 0..heads, 0..seq_len, 0..half_dim, 0..1]);
        let x2 = x.clone().slice([0..batch, 0..heads, 0..seq_len, 0..half_dim, 1..2]);
        
        let neg_x2 = x2.neg();
        
        // Stack as [-x2, x1]
        let out = Tensor::cat(vec![neg_x2, x1], 4);
        
        out.reshape([batch, heads, seq_len, dim])
    }

    pub fn forward(&self, t: Tensor<B, 4>) -> Tensor<B, 4> {
        // t shape: [batch, heads, seq_len, dim]
        
        let seq_len = t.dims()[2];
        let dim = t.dims()[3];
        let device = t.device();
        
        // Create position indices [0, 1, ..., seq_len-1]
        let t_idx = Tensor::arange(0..seq_len as i64, &device).float(); // [seq_len]
        
        // Outer product of positions and inverse frequencies
        // t_idx: [seq_len], inv_freq: [dim/2]
        // freqs = t_idx.unsqueeze(-1) * inv_freq.unsqueeze(0) -> [seq_len, dim/2]
        let t_idx_2d: Tensor<B, 2> = t_idx.unsqueeze_dim(1);
        let inv_freq_2d: Tensor<B, 2> = self.inv_freq.clone().unsqueeze_dim(0);
        let freqs: Tensor<B, 2> = t_idx_2d.matmul(inv_freq_2d);
        
        // Interleave freqs to match pairs: [f0, f0, f1, f1, ...]
        // [seq_len, dim/2] -> [seq_len, dim/2, 1] -> [seq_len, dim/2, 2] -> [seq_len, dim]
        let freqs: Tensor<B, 3> = freqs.unsqueeze_dim(2);
        let freqs = Tensor::cat(vec![freqs.clone(), freqs], 2);
        let emb: Tensor<B, 2> = freqs.reshape([seq_len, dim]);
        
        // Reshape for broadcasting: [1, 1, seq_len, dim]
        let emb: Tensor<B, 3> = emb.unsqueeze_dim(0);
        let emb: Tensor<B, 4> = emb.unsqueeze_dim(0);
        
        let cos = emb.clone().cos();
        let sin = emb.sin();
        
        // Apply rotation
        (t.clone() * cos) + (Self::rotate_half(t) * sin)
    }
}
