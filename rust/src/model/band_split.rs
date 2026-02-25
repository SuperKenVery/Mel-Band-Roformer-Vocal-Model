use super::rms_norm::RmsNorm;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct BandSplitLayer<B: Backend> {
    pub norm: RmsNorm<B>,
    pub linear: Linear<B>,
}

impl<B: Backend> BandSplitLayer<B> {
    pub fn new(device: &B::Device, dim_in: usize, dim_out: usize) -> Self {
        Self {
            norm: RmsNorm::<B>::new(device, dim_in),
            linear: LinearConfig::new(dim_in, dim_out).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.norm.forward(x);
        self.linear.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct BandSplit<B: Backend> {
    pub to_features: Vec<BandSplitLayer<B>>,
    pub dim_inputs: Vec<usize>,
}

impl<B: Backend> BandSplit<B> {
    pub fn new(device: &B::Device, dim: usize, dim_inputs: Vec<usize>) -> Self {
        let to_features = dim_inputs
            .iter()
            .map(|&dim_in| BandSplitLayer::new(device, dim_in, dim))
            .collect();

        Self {
            to_features,
            dim_inputs,
        }
    }

    /// Forward pass for band split
    /// Input: [batch, time, total_freqs_with_complex]
    /// Output: [batch, time, num_bands, dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch, time, _total] = x.dims();
        let device = x.device();
        let num_bands = self.to_features.len();
        if self.to_features.is_empty() {
            return Tensor::zeros([batch, time, 0, 0], &device);
        }

        let mut outputs: Vec<Tensor<B, 4>> = Vec::with_capacity(num_bands);
        let mut offset = 0;

        for (layer, &dim_in) in self.to_features.iter().zip(self.dim_inputs.iter()) {
            let band_input = x.clone().slice([0..batch, 0..time, offset..offset + dim_in]);
            let band_input = band_input.reshape([batch * time, dim_in]);

            let band_output = layer.forward(band_input);
            let out_shape = band_output.dims();
            let out_dim = out_shape[1];
            let band_output = band_output.reshape([batch, time, 1, out_dim]);

            outputs.push(band_output);
            offset += dim_in;
        }

        Tensor::cat(outputs, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_band_split_shape() {
        let device = Default::default();
        let dim_inputs = vec![10, 20, 30];
        let band_split = BandSplit::<TestBackend>::new(&device, 384, dim_inputs.clone());

        let total_input: usize = dim_inputs.iter().sum();
        let input = Tensor::zeros([2, 100, total_input], &device);
        let output = band_split.forward(input);

        assert_eq!(output.dims(), [2, 100, 3, 384]);
    }
}
