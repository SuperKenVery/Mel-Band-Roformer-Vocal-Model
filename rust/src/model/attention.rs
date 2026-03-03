use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Dropout, DropoutConfig, RmsNorm, RmsNormConfig},
    tensor::{backend::Backend, Tensor, activation::{softmax, sigmoid}},
};

use crate::model::rotary::RotaryEmbedding;

#[derive(Config, Debug)]
pub struct AttentionConfig {
    pub dim: usize,
    #[config(default = 8)]
    pub heads: usize,
    #[config(default = 64)]
    pub dim_head: usize,
    #[config(default = 0.0)]
    pub dropout: f64,
    #[config(default = false)]
    pub use_rotary: bool,
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    norm: RmsNorm<B>,
    to_qkv: Linear<B>,
    to_gates: Linear<B>,
    to_out: Linear<B>,
    #[module(ignore)]
    dropout: Dropout,
    #[module(ignore)]
    heads: usize,
    #[module(ignore)]
    dim_head: usize,
    #[module(ignore)]
    scale: f64,
    #[module(ignore)]
    rotary_embed: Option<RotaryEmbedding<B>>,
}

impl<B: Backend> Attention<B> {
    pub fn new(config: &AttentionConfig, device: &B::Device) -> Self {
        let dim = config.dim;
        let heads = config.heads;
        let dim_head = config.dim_head;
        let dim_inner = heads * dim_head;
        let dropout = config.dropout;

        let norm = RmsNormConfig::new(dim).init(device);
        
        let to_qkv = LinearConfig::new(dim, dim_inner * 3)
            .with_bias(false)
            .init(device);
            
        let to_gates = LinearConfig::new(dim, heads).init(device);
        
        let to_out = LinearConfig::new(dim_inner, dim)
            .with_bias(false)
            .init(device);
            
        let dropout_layer = DropoutConfig::new(dropout).init();
        
        let rotary_embed = if config.use_rotary {
            use crate::model::rotary::RotaryEmbeddingConfig;
            Some(RotaryEmbedding::new(&RotaryEmbeddingConfig::new(dim_head), device))
        } else {
            None
        };

        Self {
            norm,
            to_qkv,
            to_gates,
            to_out,
            dropout: dropout_layer,
            heads,
            dim_head,
            scale: (dim_head as f64).powf(-0.5),
            rotary_embed,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // x: [batch, seq_len, dim]
        let batch_size = x.dims()[0];
        let seq_len = x.dims()[1];
        
        let x_norm = self.norm.forward(x.clone());
        
        // qkv: [batch, seq_len, 3 * heads * dim_head]
        let qkv = self.to_qkv.forward(x_norm.clone());
        
        // Reshape to [batch, seq_len, 3, heads, dim_head]
        let qkv = qkv.reshape([batch_size, seq_len, 3, self.heads, self.dim_head]);
        
        // Permute to [3, batch, heads, seq_len, dim_head]
        let qkv = qkv.permute([2, 0, 3, 1, 4]);
        
        // Split into q, k, v
        let q = qkv.clone().slice([0..1, 0..batch_size, 0..self.heads, 0..seq_len, 0..self.dim_head])
            .flatten(0, 1);
        let k = qkv.clone().slice([1..2, 0..batch_size, 0..self.heads, 0..seq_len, 0..self.dim_head])
            .flatten(0, 1);
        let v = qkv.clone().slice([2..3, 0..batch_size, 0..self.heads, 0..seq_len, 0..self.dim_head])
            .flatten(0, 1);
            
        // Apply rotary embeddings if enabled
        let (q, k) = if let Some(rotary) = &self.rotary_embed {
            (rotary.forward(q), rotary.forward(k))
        } else {
            (q, k)
        };
        
        // Attention
        let k_t = k.transpose(); // swaps last two dims: [batch, heads, dim_head, seq_len]
        let sim = q.matmul(k_t) * self.scale; // [batch, heads, seq_len, seq_len]
        
        let attn = softmax(sim, 3); // softmax over last dim
        let attn = self.dropout.forward(attn);
        
        let out = attn.matmul(v); // [batch, heads, seq_len, dim_head]
        
        // Gating
        // gates = to_gates(x) -> [batch, seq_len, heads]
        let gates = self.to_gates.forward(x_norm);
        // reshape gates to [batch, heads, seq_len, 1] for broadcasting
        let gates = gates.permute([0, 2, 1]).unsqueeze_dim(3); // [batch, heads, seq_len, 1]
        
        let out = out * sigmoid(gates);
        
        // Recombine heads
        let out = out.permute([0, 2, 1, 3]);
        // flatten last two dims
        let out = out.reshape([batch_size, seq_len, self.heads * self.dim_head]);
        
        // Output projection
        let out = self.to_out.forward(out);
        let out = self.dropout.forward(out);
        
        out
    }
}
