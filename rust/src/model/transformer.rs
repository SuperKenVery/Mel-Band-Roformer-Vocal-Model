use burn::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, Gelu, Linear, LinearConfig, RmsNorm, RmsNormConfig},
    tensor::{backend::Backend, Tensor},
};

use crate::model::attention::{Attention, AttentionConfig};

#[derive(Config, Debug)]
pub struct FeedForwardConfig {
    pub dim: usize,
    #[config(default = 4)]
    pub mult: usize,
    #[config(default = 0.0)]
    pub dropout: f64,
}

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    pub norm: RmsNorm<B>,
    pub linear1: Linear<B>,
    #[module(ignore)]
    pub gelu: Gelu,
    #[module(ignore)]
    pub dropout1: Dropout,
    pub linear2: Linear<B>,
    #[module(ignore)]
    pub dropout2: Dropout,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(config: &FeedForwardConfig, device: &B::Device) -> Self {
        let dim = config.dim;
        let dim_inner = dim * config.mult;
        let dropout = config.dropout;

        let norm = RmsNormConfig::new(dim).init(device);
        let linear1 = LinearConfig::new(dim, dim_inner).init(device);
        let gelu = Gelu::new();
        let dropout1 = DropoutConfig::new(dropout).init();
        let linear2 = LinearConfig::new(dim_inner, dim).init(device);
        let dropout2 = DropoutConfig::new(dropout).init();

        Self {
            norm,
            linear1,
            gelu,
            dropout1,
            linear2,
            dropout2,
        }
    }
    
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.norm.forward(x);
        let x = self.linear1.forward(x);
        let x = self.gelu.forward(x);
        let x = self.dropout1.forward(x);
        let x = self.linear2.forward(x);
        let x = self.dropout2.forward(x);
        x
    }
}

#[derive(Config, Debug)]
pub struct TransformerConfig {
    pub dim: usize,
    pub depth: usize,
    #[config(default = 8)]
    pub heads: usize,
    #[config(default = 64)]
    pub dim_head: usize,
    #[config(default = 0.0)]
    pub attn_dropout: f64,
    #[config(default = 0.0)]
    pub ff_dropout: f64,
    #[config(default = 4)]
    pub ff_mult: usize,
    #[config(default = true)]
    pub norm_output: bool,
    #[config(default = false)]
    pub use_rotary: bool,
}

#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    layers: Vec<Attention<B>>,
    ff_layers: Vec<FeedForward<B>>,
    norm: Option<RmsNorm<B>>,
}

impl<B: Backend> Transformer<B> {
    pub fn new(config: &TransformerConfig, device: &B::Device) -> Self {
        let mut layers = Vec::new();
        let mut ff_layers = Vec::new();

        for _ in 0..config.depth {
            let attn_config = AttentionConfig::new(config.dim)
                .with_heads(config.heads)
                .with_dim_head(config.dim_head)
                .with_dropout(config.attn_dropout)
                .with_use_rotary(config.use_rotary);
            
            let ff_config = FeedForwardConfig::new(config.dim)
                .with_mult(config.ff_mult)
                .with_dropout(config.ff_dropout);

            layers.push(Attention::new(&attn_config, device));
            ff_layers.push(FeedForward::new(&ff_config, device));
        }

        let norm = if config.norm_output {
            Some(RmsNormConfig::new(config.dim).init(device))
        } else {
            None
        };

        Self {
            layers,
            ff_layers,
            norm,
        }
    }

    pub fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        for (attn, ff) in self.layers.iter().zip(self.ff_layers.iter()) {
            x = x.clone() + attn.forward(x.clone());
            x = x.clone() + ff.forward(x.clone());
        }

        if let Some(norm) = &self.norm {
            norm.forward(x)
        } else {
            x
        }
    }
}
