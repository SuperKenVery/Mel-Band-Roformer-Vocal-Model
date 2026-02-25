use burn::prelude::*;
use burn::tensor::backend::Backend;

#[derive(Clone, Debug)]
pub struct RotaryEmbedding {
    pub inv_freqs: Vec<f32>,
    pub dim: usize,
}

impl RotaryEmbedding {
    pub fn new(dim: usize) -> Self {
        let base = 10000.0f32;
        let half_dim = dim / 2;

        let inv_freqs: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / dim as f32))
            .collect();

        Self { inv_freqs, dim }
    }

    pub fn from_freqs(freqs: Vec<f32>) -> Self {
        let dim = freqs.len() * 2;
        Self { inv_freqs: freqs, dim }
    }

    pub fn rotate_queries_or_keys<B: Backend>(
        &self,
        x: Tensor<B, 4>,
        offset: usize,
    ) -> Tensor<B, 4> {
        let [batch, heads, seq_len, dim] = x.dims();
        let device = x.device();

        let inv_freq_tensor: Tensor<B, 2> =
            Tensor::<B, 1>::from_floats(self.inv_freqs.as_slice(), &device).unsqueeze_dim(0);

        let positions: Vec<f32> = (offset..seq_len + offset).map(|i| i as f32).collect();
        let positions_tensor: Tensor<B, 2> =
            Tensor::<B, 1>::from_floats(positions.as_slice(), &device).unsqueeze_dim(1);

        let freqs: Tensor<B, 2> = positions_tensor.matmul(inv_freq_tensor);

        let cos = freqs.clone().cos();
        let sin = freqs.sin();
        
        let cos: Tensor<B, 4> = cos.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);
        let sin: Tensor<B, 4> = sin.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);

        let cos = cos.expand([batch, heads, seq_len, dim / 2]);
        let sin = sin.expand([batch, heads, seq_len, dim / 2]);

        // Reshape x to [batch, heads, seq_len, dim/2, 2] to separate interleaved pairs
        let x_reshaped = x.reshape([batch, heads, seq_len, dim / 2, 2]);

        // x1 is evens (index 0), x2 is odds (index 1)
        let x1 = x_reshaped.clone().slice([0..batch, 0..heads, 0..seq_len, 0..dim / 2, 0..1])
            .reshape([batch, heads, seq_len, dim / 2]);
        let x2 = x_reshaped.slice([0..batch, 0..heads, 0..seq_len, 0..dim / 2, 1..2])
            .reshape([batch, heads, seq_len, dim / 2]);

        let rotated_x1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
        let rotated_x2 = x1 * sin + x2 * cos;

        // Stack back: [batch, heads, seq_len, dim/2, 2]
        let rotated_x1 = rotated_x1.unsqueeze_dim::<5>(4);
        let rotated_x2 = rotated_x2.unsqueeze_dim::<5>(4);
        let rotated = Tensor::cat(vec![rotated_x1, rotated_x2], 4);

        // Reshape back to [batch, heads, seq_len, dim]
        rotated.reshape([batch, heads, seq_len, dim])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_rotary_shape() {
        let device = Default::default();
        let rope = RotaryEmbedding::new(64);

        let input: Tensor<TestBackend, 4> = Tensor::zeros([2, 8, 16, 64], &device);
        let output = rope.rotate_queries_or_keys(input, 0);

        assert_eq!(output.dims(), [2, 8, 16, 64]);
    }
}
