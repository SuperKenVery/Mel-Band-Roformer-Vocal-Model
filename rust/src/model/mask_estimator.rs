use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    pub linear3: Linear<B>,
}

impl<B: Backend> MLP<B> {
    pub fn new(device: &B::Device, dim_in: usize, dim_out: usize, dim_hidden: usize) -> Self {
        Self {
            linear1: LinearConfig::new(dim_in, dim_hidden).init(device),
            linear2: LinearConfig::new(dim_hidden, dim_hidden).init(device),
            linear3: LinearConfig::new(dim_hidden, dim_out).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(x);
        let x = x.tanh();
        let x = self.linear2.forward(x);
        let x = x.tanh();
        self.linear3.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct MaskEstimatorBand<B: Backend> {
    pub mlp: MLP<B>,
    pub dim_out: usize,
}

impl<B: Backend> MaskEstimatorBand<B> {
    pub fn new(device: &B::Device, dim: usize, dim_out: usize, expansion_factor: usize) -> Self {
        let dim_hidden = dim * expansion_factor;
        let mlp = MLP::new(device, dim, dim_out * 2, dim_hidden);
        Self { mlp, dim_out }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let out = self.mlp.forward(x);
        let [n, total] = out.dims();

        // GLU: first_half * sigmoid(second_half) (matching PyTorch nn.GLU(dim=-1))
        let value = out.clone().slice([0..n, 0..self.dim_out]);
        let gate = out.slice([0..n, self.dim_out..total]);

        value * burn::tensor::activation::sigmoid(gate)
    }
}

#[derive(Module, Debug)]
pub struct MaskEstimator<B: Backend> {
    pub to_freqs: Vec<MaskEstimatorBand<B>>,
    pub dim_inputs: Vec<usize>,
}

impl<B: Backend> MaskEstimator<B> {
    pub fn new(device: &B::Device, dim: usize, dim_inputs: Vec<usize>, _depth: usize) -> Self {
        let expansion_factor = 4;

        let to_freqs = dim_inputs
            .iter()
            .map(|&dim_out| MaskEstimatorBand::new(device, dim, dim_out, expansion_factor))
            .collect();

        Self { to_freqs, dim_inputs }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, time, _num_bands, dim] = x.dims();
        let device = x.device();

        let total_out: usize = self.dim_inputs.iter().sum();
        let mut output = Tensor::zeros([batch, time, total_out], &device);
        let mut offset = 0;

        for (band_idx, (layer, &dim_out)) in self.to_freqs.iter().zip(self.dim_inputs.iter()).enumerate() {
            let band_input = x
                .clone()
                .slice([0..batch, 0..time, band_idx..band_idx + 1, 0..dim])
                .reshape([batch * time, dim]);

            let band_output = layer.forward(band_input);
            let band_output = band_output.reshape([batch, time, dim_out]);

            output = output.slice_assign(
                [0..batch, 0..time, offset..offset + dim_out],
                band_output,
            );
            offset += dim_out;
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_mlp_shape() {
        let device = Default::default();
        let mlp = MLP::<TestBackend>::new(&device, 384, 64, 512);

        let input = Tensor::zeros([100, 384], &device);
        let output = mlp.forward(input);

        assert_eq!(output.dims(), [100, 64]);
    }

    #[test]
    fn test_mask_estimator_shape() {
        let device = Default::default();
        let dim_inputs = vec![10, 20, 30];
        let mask_est = MaskEstimator::<TestBackend>::new(&device, 384, dim_inputs.clone(), 2);

        let input = Tensor::zeros([2, 100, 3, 384], &device);
        let output = mask_est.forward(input);

        let total: usize = dim_inputs.iter().sum();
        assert_eq!(output.dims(), [2, 100, total]);
    }
}
