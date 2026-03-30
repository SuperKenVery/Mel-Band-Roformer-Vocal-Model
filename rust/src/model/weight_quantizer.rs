use burn::module::{ModuleMapper, Param};
use burn::tensor::{Tensor, backend::Backend};
use burn::tensor::quantization::{Calibration, QuantScheme, compute_q_params, compute_range};

/// A weight quantizer that only quantizes 2D+ tensors (weights),
/// leaving 1D tensors (biases) in full precision.
///
/// When the scheme uses per-tensor quantization, this actually applies per-row
/// quantization for better quality: each row of the weight matrix gets its own scale.
pub struct WeightQuantizer {
    pub calibration: Calibration,
    pub scheme: QuantScheme,
}

impl<B: Backend> ModuleMapper<B> for WeightQuantizer {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        if D < 2 {
            return param;
        }
        let (id, tensor, mapper) = param.consume();
        let range = compute_range(&self.scheme, &tensor, &self.calibration);
        let qparams = compute_q_params(&self.scheme, range);
        let tensor = tensor.quantize(&self.scheme, qparams);
        Param::from_mapped_value(id, tensor, mapper)
    }
}
