use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{activation::sigmoid, backend::Backend, Tensor},
};

#[derive(Config, Debug)]
pub struct MLPConfig {
    pub dim_in: usize,
    pub dim_out: usize,
    pub dim_hidden: usize,
    pub depth: usize,
}

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    pub layers: Vec<Linear<B>>,
}

impl<B: Backend> MLP<B> {
    pub fn new(config: &MLPConfig, device: &B::Device) -> Self {
        let mut layers = Vec::new();
        let dims = std::iter::once(config.dim_in)
            .chain(std::iter::repeat(config.dim_hidden).take(config.depth))
            .chain(std::iter::once(config.dim_out))
            .collect::<Vec<_>>();

        for i in 0..dims.len() - 1 {
            layers.push(LinearConfig::new(dims[i], dims[i + 1]).init(device));
        }

        Self { layers }
    }

    pub fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        for (i, layer) in self.layers.iter().enumerate() {
            if i > 0 {
                x = x.tanh();
            }
            x = layer.forward(x);
        }
        x
    }
}

#[derive(Module, Debug)]
pub struct BandMaskEstimator<B: Backend> {
    pub mlp: MLP<B>,
}

impl<B: Backend> BandMaskEstimator<B> {
    pub fn new(config: &MLPConfig, device: &B::Device) -> Self {
        let mlp = MLP::new(config, device);
        Self { mlp }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.mlp.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct MaskEstimatorConfig {
    pub dim: usize,
    pub dim_inputs: Vec<usize>,
    pub depth: usize,
    #[config(default = 4)]
    pub mlp_expansion_factor: usize,
}

#[derive(Module, Debug)]
pub struct MaskEstimator<B: Backend> {
    pub to_freqs: Vec<BandMaskEstimator<B>>,
}

impl<B: Backend> MaskEstimator<B> {
    pub fn new(config: &MaskEstimatorConfig, device: &B::Device) -> Self {
        let dim_hidden = config.dim * config.mlp_expansion_factor;
        let mut to_freqs = Vec::new();

        for &dim_in in &config.dim_inputs {
            let mlp_config = MLPConfig {
                dim_in: config.dim,
                dim_out: dim_in * 2,
                dim_hidden,
                depth: config.depth,
            };
            to_freqs.push(BandMaskEstimator::new(
                &mlp_config,
                device,
            ));
        }

        Self { to_freqs }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        // x: [batch, time, bands, dim]
        let dims = x.dims();
        let batch = dims[0];
        let time = dims[1];
        let bands = dims[2];

        let mut outs = Vec::new();

        for i in 0..bands {
            let band_features = x
                .clone()
                .slice([0..batch, 0..time, i..i + 1])
                .flatten(2, 3); // [batch, time, dim]
            let x_out = self.to_freqs[i].forward(band_features); // [batch, time, dim_band * 2]

            // GLU
            let chunks = x_out.chunk(2, 2);
            let val = chunks[0].clone();
            let gate = chunks[1].clone();
            let out = val * sigmoid(gate);

            outs.push(out);
        }

        Tensor::cat(outs, 2) // [batch, time, total_freq]
    }
}
