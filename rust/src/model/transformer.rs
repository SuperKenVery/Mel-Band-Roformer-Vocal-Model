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

        let x = self.attention.forward(x.clone(), Some(&rotary)) + x;
        let x = self.ff.forward(x.clone()) + x;
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
        self.norm.forward(x)
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
