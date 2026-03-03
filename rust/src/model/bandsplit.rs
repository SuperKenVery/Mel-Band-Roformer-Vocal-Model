use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig},
    tensor::{backend::Backend, Tensor},
};

#[derive(Config, Debug)]
pub struct BandSplitConfig {
    pub dim: usize,
    pub dim_inputs: Vec<usize>,
}

#[derive(Module, Debug)]
pub struct BandFeature<B: Backend> {
    pub norm: RmsNorm<B>,
    pub linear: Linear<B>,
}

impl<B: Backend> BandFeature<B> {
    pub fn new(dim_in: usize, dim_out: usize, device: &B::Device) -> Self {
        Self {
            norm: RmsNormConfig::new(dim_in).init(device),
            linear: LinearConfig::new(dim_in, dim_out).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.norm.forward(x);
        self.linear.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct BandSplit<B: Backend> {
    pub to_features: Vec<BandFeature<B>>,
}

impl<B: Backend> BandSplit<B> {
    pub fn new(config: &BandSplitConfig, device: &B::Device) -> Self {
        let mut to_features = Vec::new();
        
        for &dim_in in &config.dim_inputs {
            to_features.push(BandFeature::new(dim_in, config.dim, device));
        }

        Self {
            to_features,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        // x: [batch, time, total_freq_dim]
        let mut start = 0;
        let mut outs: Vec<Tensor<B, 3>> = Vec::new();
        let dims = x.dims();
        let batch = dims[0];
        let time = dims[1];
        
        for feature in &self.to_features {
            // Infer dim_in from linear layer weight
            // weight shape: [d_in, d_out]
            let dim_in = feature.linear.weight.dims()[0];
            
            let end = start + dim_in;
            // Slice x: [batch, time, start..end]
            let slice = x.clone().slice([0..batch, 0..time, start..end]);
            
            let out = feature.forward(slice); // [batch, time, dim]
            outs.push(out);
            
            start = end;
        }
        
        // Stack tensors
        // Burn doesn't have stack, use cat + unsqueeze
        let outs = outs.into_iter().map(|t| t.unsqueeze_dim(2)).collect(); // [batch, time, 1, dim]
        Tensor::cat(outs, 2) // [batch, time, bands, dim]
    }
}
