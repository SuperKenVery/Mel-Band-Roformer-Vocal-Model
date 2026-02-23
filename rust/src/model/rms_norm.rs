use burn::module::{Module, Param};
use burn::nn::Initializer;
use burn::prelude::*;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    gamma: Param<Tensor<B, 1>>,
    scale: f32,
    dim: usize,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(device: &B::Device, dim: usize) -> Self {
        let gamma = Initializer::Ones.init([dim], device);
        let scale = (dim as f32).sqrt();

        Self {
            gamma: gamma.into(),
            scale,
            dim,
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let normalized = normalize_last_dim(x);
        let gamma = self.gamma.val().unsqueeze();
        normalized.mul_scalar(self.scale) * gamma
    }

    pub fn load_gamma(&mut self, gamma: Tensor<B, 1>) {
        self.gamma = Param::from_tensor(gamma);
    }
}

fn normalize_last_dim<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let eps = 1e-8;
    let squared = x.clone().powf_scalar(2.0);
    let mean_squared = squared.mean_dim(D - 1);
    let rms = (mean_squared + eps).sqrt();
    x / rms
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_rms_norm_shape() {
        let device = Default::default();
        let norm = RmsNorm::<TestBackend>::new(&device, 64);

        let input = Tensor::zeros([2, 10, 64], &device);
        let output = norm.forward(input);

        assert_eq!(output.dims(), [2, 10, 64]);
    }
}
