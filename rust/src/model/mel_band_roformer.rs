use burn::{
    config::Config,
    module::{Module, Param},
    nn::conv::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig},
    nn::PaddingConfig1d,
    tensor::{backend::Backend, Tensor, Int, IndexingUpdateOp},
};

use crate::model::transformer::{Transformer, TransformerConfig};
use crate::model::bandsplit::{BandSplit, BandSplitConfig};
use crate::model::mask_estimator::{MaskEstimator, MaskEstimatorConfig};

pub struct MelBandConstants<B: Backend> {
    pub freq_indices: Tensor<B, 1, Int>,
    pub num_bands_per_freq: Tensor<B, 1>,
    pub num_freqs_per_band: Tensor<B, 1>, // Added
    pub stft_kernel_real: Tensor<B, 3>,
    pub stft_kernel_imag: Tensor<B, 3>,
    pub istft_kernel_real: Tensor<B, 3>,
    pub istft_kernel_imag: Tensor<B, 3>,
}

#[derive(Config, Debug)]
pub struct MelBandRoformerConfig {
    pub dim: usize,
    pub depth: usize,
    #[config(default = false)]
    pub stereo: bool,
    #[config(default = 1)]
    pub num_stems: usize,
    #[config(default = 2)]
    pub time_transformer_depth: usize,
    #[config(default = 2)]
    pub freq_transformer_depth: usize,
    #[config(default = 60)]
    pub num_bands: usize,
    #[config(default = 64)]
    pub dim_head: usize,
    #[config(default = 8)]
    pub heads: usize,
    #[config(default = 0.1)]
    pub attn_dropout: f64,
    #[config(default = 0.1)]
    pub ff_dropout: f64,
    #[config(default = 1025)]
    pub dim_freqs_in: usize,
    #[config(default = 44100)]
    pub sample_rate: usize,
    #[config(default = 2048)]
    pub stft_n_fft: usize,
    #[config(default = 512)]
    pub stft_hop_length: usize,
    #[config(default = 2048)]
    pub stft_win_length: usize,
    #[config(default = 1)]
    pub mask_estimator_depth: usize,
}

#[derive(Module, Debug)]
pub struct MelBandRoformer<B: Backend> {
    time_transformers: Vec<Transformer<B>>,
    freq_transformers: Vec<Transformer<B>>,
    band_split: BandSplit<B>,
    mask_estimators: Vec<MaskEstimator<B>>,
    
    // STFT/ISTFT helpers as modules (holding weights)
    #[module(ignore)]
    stft_conv_real: Conv1d<B>,
    #[module(ignore)]
    stft_conv_imag: Conv1d<B>,
    #[module(ignore)]
    istft_conv_real: ConvTranspose1d<B>,
    #[module(ignore)]
    istft_conv_imag: ConvTranspose1d<B>,
    
    // Buffers flattened and ignored
    #[module(ignore)]
    freq_selection_matrix: Tensor<B, 2>,
    #[module(ignore)]
    freq_scatter_matrix: Tensor<B, 2>,
    #[module(ignore)]
    num_bands_per_freq: Tensor<B, 1>,
    
    #[module(ignore)]
    stft_n_fft: usize,
    #[module(ignore)]
    stft_hop_length: usize,
    #[module(ignore)]
    stft_win_length: usize,
    #[module(ignore)]
    audio_channels: usize,
    #[module(ignore)]
    num_stems: usize,
}

impl<B: Backend> MelBandRoformer<B> {
    pub fn new(
        config: &MelBandRoformerConfig,
        device: &B::Device,
        constants: MelBandConstants<B>,
    ) -> Self {
        let dim = config.dim;
        let audio_channels = if config.stereo { 2 } else { 1 };
        
        // --- Initialize Transformers ---
        let mut time_transformers = Vec::new();
        let mut freq_transformers = Vec::new();
        
        let transformer_config = TransformerConfig::new(dim, config.time_transformer_depth)
            .with_heads(config.heads)
            .with_dim_head(config.dim_head)
            .with_attn_dropout(config.attn_dropout)
            .with_ff_dropout(config.ff_dropout)
            .with_use_rotary(true);

        for _ in 0..config.depth {
            let time_transformer = Transformer::new(&transformer_config, device);
            
            let freq_config = TransformerConfig::new(dim, config.freq_transformer_depth)
                 .with_heads(config.heads)
                 .with_dim_head(config.dim_head)
                 .with_attn_dropout(config.attn_dropout)
                 .with_ff_dropout(config.ff_dropout)
                 .with_use_rotary(true);
                 
            let freq_transformer = Transformer::new(&freq_config, device);
            
            time_transformers.push(time_transformer);
            freq_transformers.push(freq_transformer);
        }

        // --- Band Split & Mask Estimator ---
        let _num_bands = config.num_bands;
        let n_fft = config.stft_n_fft;
        let num_freqs = n_fft / 2 + 1;
        
        // Use constants to get num_selected
        let freq_indices = constants.freq_indices.clone();
        let num_selected = freq_indices.dims()[0];
        
        let num_freqs_per_band_tensor = constants.num_freqs_per_band;
        
        // dim_inputs: 2 * f * audio_channels
        let d = num_freqs_per_band_tensor.clone().into_data();
        let vec_f = d.to_vec::<f32>().unwrap();
        let dim_inputs = vec_f.iter()
            .map(|&f| 2 * (f as usize) * audio_channels)
            .collect::<Vec<_>>();
            
        let band_split = BandSplit::new(&BandSplitConfig::new(dim, dim_inputs.clone()), device);
        
        let mut mask_estimators = Vec::new();
        for _ in 0..config.num_stems {
            let me_config = MaskEstimatorConfig::new(dim, dim_inputs.clone(), config.mask_estimator_depth);
            mask_estimators.push(MaskEstimator::new(&me_config, device));
        }

        // --- Constants & Matrices ---
        // freq_indices: [num_selected]
        // Already cloned above
        
        let num_bands_per_freq = constants.num_bands_per_freq;
        
        // num_selected already defined
        let num_total = num_freqs;

        // Construct freq_selection_matrix: [num_selected, num_total]
        // One-hot encoding based on freq_indices
        let indices = freq_indices.clone().unsqueeze_dim(1); // [num_selected, 1]
        let values = Tensor::ones([num_selected, 1], device);
        let freq_selection_matrix = Tensor::zeros([num_selected, num_total], device)
            .scatter(1, indices, values, IndexingUpdateOp::Add);
            
        // Construct freq_scatter_matrix: [num_selected, num_total]
        // Used for overlap-add (scatter_add). In this implementation, we use matmul with transpose.
        // It's effectively the same binary mask as selection matrix for now.
        let freq_scatter_matrix = freq_selection_matrix.clone();

        // --- STFT/ISTFT Kernels ---
        // Load kernels from constants and wrap in Conv1d/ConvTranspose1d
        // Shape: [n_freqs, 1, win_length]
        let win_length = config.stft_win_length;
        // let kernel_shape = [num_freqs, 1, win_length];
        
        let stft_kernel_real = constants.stft_kernel_real;
        let stft_kernel_imag = constants.stft_kernel_imag;
            
        let istft_kernel_real = constants.istft_kernel_real;
        let istft_kernel_imag = constants.istft_kernel_imag;

        // Helper to create Conv1d with fixed weights
        let create_conv = |weight: Tensor<B, 3>| {
            let config = Conv1dConfig::new(1, num_freqs, win_length)
                .with_stride(1) // Workaround: Use stride 1 and slice later
                .with_bias(false)
                .with_padding(PaddingConfig1d::Valid); // We pad manually
            let mut conv = config.init(device);
            conv.weight = Param::from_tensor(weight); // Set weight
            conv
        };

        // Helper to create ConvTranspose1d with fixed weights
        let create_conv_trans = |weight: Tensor<B, 3>| {
            // ConvTranspose1d: in_channels=num_freqs, out_channels=1
            let config = ConvTranspose1dConfig::new([num_freqs, 1], win_length)
                .with_stride(config.stft_hop_length)
                .with_bias(false)
                .with_padding(0); // We handle padding/cropping manually if needed
            let mut conv = config.init(device);
            // ConvTranspose1d weight shape: [in_channels, out_channels, kernel_size]
            // We have [num_freqs, 1, win_length], matches.
            conv.weight = Param::from_tensor(weight);
            conv
        };

        let stft_conv_real = create_conv(stft_kernel_real);
        let stft_conv_imag = create_conv(stft_kernel_imag);
        let istft_conv_real = create_conv_trans(istft_kernel_real);
        let istft_conv_imag = create_conv_trans(istft_kernel_imag);

        Self {
            time_transformers,
            freq_transformers,
            band_split,
            mask_estimators,
            stft_conv_real,
            stft_conv_imag,
            istft_conv_real,
            istft_conv_imag,
            freq_selection_matrix,
            freq_scatter_matrix,
            num_bands_per_freq,
            stft_n_fft: n_fft,
            stft_hop_length: config.stft_hop_length,
            stft_win_length: win_length,
            audio_channels: audio_channels,
            num_stems: config.num_stems,
        }
    }

    // --- Helpers ---

    fn stft(&self, x: Tensor<B, 3>) -> Tensor<B, 5> {
        // x: [batch, channels, time]
        let dims = x.dims();
        let batch = dims[0];
        let channels = dims[1];
        let time = dims[2];
        
        // Merge batch and channels: [batch * channels, 1, time]
        let x = x.reshape([batch * channels, 1, time]);
        
        // Pad input (reflect)
        // Pad amount: n_fft // 2
        let pad = self.stft_n_fft / 2;
        
        // Use zero padding for now
        let zeros_left = Tensor::zeros([batch * channels, 1, pad], &x.device());
        let zeros_right = Tensor::zeros([batch * channels, 1, pad], &x.device());
        let x_padded = Tensor::cat(vec![zeros_left, x, zeros_right], 2);
        
        let batch_channels = x_padded.dims()[0];
        let time_full_padded = x_padded.dims()[2];
        let hop = self.stft_hop_length;
        let win = 2048; // Configured window length
        
        // Calculate number of frames
        // Last frame: start = (N-1)*hop, end = start + win <= time_full_padded
        let num_frames = if time_full_padded < win {
            0
        } else {
            (time_full_padded - win) / hop + 1
        };
        
        let frames_per_chunk = 50; // Small enough to fit in GPU texture/buffer limit (validated < 30000 samples)
        let mut real_chunks = Vec::new();
        let mut imag_chunks = Vec::new();
        
        for i in (0..num_frames).step_by(frames_per_chunk) {
            let end_frame = usize::min(i + frames_per_chunk, num_frames);
            let n_frames = end_frame - i;
            
            let start_sample = i * hop;
            let end_sample = (end_frame - 1) * hop + win;
            
            // Ensure end_sample doesn't exceed bounds (though num_frames logic should prevent this)
            let end_sample = usize::min(end_sample, time_full_padded);
            
            let x_slice = x_padded.clone().slice([0..batch_channels, 0..1, start_sample..end_sample]);
            
            let r_out = self.stft_conv_real.forward(x_slice.clone());
            let i_out = self.stft_conv_imag.forward(x_slice);
            
            // Select frames
            let indices = Tensor::arange(0..n_frames as i64, &x_padded.device()) * (hop as i64);
            let r_sel = r_out.select(2, indices.clone());
            let i_sel = i_out.select(2, indices);
            
            real_chunks.push(r_sel);
            imag_chunks.push(i_sel);
        }
        
        let real = Tensor::cat(real_chunks, 2);
        let imag = Tensor::cat(imag_chunks, 2);
        
        // real, imag: [batch*channels, freq, time_frames]
        
        let real: Tensor<B, 4> = real.unsqueeze_dim(3);
        let imag: Tensor<B, 4> = imag.unsqueeze_dim(3);
        
        let out: Tensor<B, 4> = Tensor::cat(vec![real, imag], 3); // [batch*channels, freq, time_frames, 2]
        
        // Reshape back
        let freq = out.dims()[1];
        let time_frames = out.dims()[2];
        
        out.reshape([batch, channels, freq, time_frames, 2])
    }

    fn istft(&self, x: Tensor<B, 5>, length: usize) -> Tensor<B, 3> {
        // x: [batch, channels, freq, time_frames, 2]
        let dims = x.dims();
        let batch = dims[0];
        let channels = dims[1];
        let freq = dims[2];
        let time_frames = dims[3];
        
        let x = x.reshape([batch * channels, freq, time_frames, 2]);
        
        // Split real/imag
        let real = x.clone().slice([0..batch*channels, 0..freq, 0..time_frames, 0..1]).flatten(2, 3);
        let imag = x.slice([0..batch*channels, 0..freq, 0..time_frames, 1..2]).flatten(2, 3);
        
        let r_kr = self.istft_conv_real.forward(real.clone());
        let i_ki = self.istft_conv_imag.forward(imag.clone());
        
        // Debug
        let d = real.clone().into_data();
        let v = d.to_vec::<f32>().unwrap();
        let max_in = v.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        
        let d = self.istft_conv_real.weight.val().into_data();
        let v = d.to_vec::<f32>().unwrap();
        let max_w = v.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        
        let d = r_kr.clone().into_data();
        let v = d.to_vec::<f32>().unwrap();
        let max_out = v.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        
        println!("DEBUG: ISTFT inner: input max={}, weight max={}, output max={}", max_in, max_w, max_out);

        // Output is Real part
        let out = r_kr + i_ki; // [batch*channels, 1, time_out]
        
        let d = out.clone().into_data();
        let v = d.to_vec::<f32>().unwrap();
        let max = v.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        println!("DEBUG: ISTFT raw max: {}", max);

        // Apply scaling factor for ISTFT (empirically determined)
        let out = out / 1792.9;
        
        // Remove padding
        let pad = self.stft_n_fft / 2;
        let time_out = out.dims()[2];
        let start = pad;
        let end = time_out - pad;
        
        let out = out.slice([0..batch*channels, 0..1, start..end]);
        
        // Trim to original length if needed
        let out_len = out.dims()[2];
        let out = if out_len > length {
            out.slice([0..batch*channels, 0..1, 0..length])
        } else if out_len < length {
            // Pad with zeros
            let pad_len = length - out_len;
            let padding = Tensor::zeros([batch * channels, 1, pad_len], &out.device());
            Tensor::cat(vec![out, padding], 2)
        } else {
            out
        };

        out.reshape([batch, channels, length])
    }

    pub fn set_stft_weights(&mut self, stft_real: Tensor<B, 3>, stft_imag: Tensor<B, 3>) {
        self.stft_conv_real.weight = Param::from_tensor(stft_real);
        self.stft_conv_imag.weight = Param::from_tensor(stft_imag);
    }
    
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 4> {
        // input: [batch, channels, time]
        let batch = input.dims()[0];
        let channels = input.dims()[1];
        let time_audio = input.dims()[2];
        // let num_total = self.freq_selection_matrix.dims()[1];

        // 1. STFT
        let spec = self.stft(input.clone()); // [batch, channels, freq, time, 2]
        
        let time_frames = spec.dims()[3];
        
        // 2. Band Selection
        let spec_perm = spec.permute([0, 1, 3, 4, 2]); // [batch, channels, time, 2, freq]
        // Selection matrix: [num_selected, freq]. Transpose -> [freq, num_selected]
        let selection_t = self.freq_selection_matrix.clone().transpose();
        
        let spec_selected = spec_perm.matmul(selection_t.unsqueeze()); // Broadcast matmul?
        
        let spec_selected = spec_selected.permute([0, 1, 4, 2, 3]); // [batch, channels, num_selected, time, 2]
        let num_selected = self.freq_selection_matrix.dims()[0];
        
        // 3. Band Split
        let x = spec_selected.clone().permute([0, 3, 2, 1, 4]);
        let x = x.reshape([batch, time_frames, num_selected * channels * 2]);
        
        let mut x = self.band_split.forward(x); // [batch, time, bands, dim]

        // 4. Transformers
        for (time_transformer, freq_transformer) in self.time_transformers.iter().zip(self.freq_transformers.iter()) {
            // Time Transformer
            // x: [batch, time, bands, dim]
            // Merge bands into batch for time attention
            let dims = x.dims();
            let bands = dims[2];
            let dim_feat = dims[3];
            
            let x_time = x.clone().permute([0, 2, 1, 3]).reshape([batch * bands, time_frames, dim_feat]);
            let x_time = time_transformer.forward(x_time);
            x = x_time.reshape([batch, bands, time_frames, dim_feat]).permute([0, 2, 1, 3]);
            
            // Freq Transformer
            // Merge time into batch
            let x_freq = x.clone().reshape([batch * time_frames, bands, dim_feat]);
            let x_freq = freq_transformer.forward(x_freq);
            x = x_freq.reshape([batch, time_frames, bands, dim_feat]);
        }
        
        // 5. Mask Estimation
        let mut stem_outputs = Vec::new();
        
        for mask_estimator in &self.mask_estimators {
            let mask = mask_estimator.forward(x.clone()); // [batch, time, total_freq_dim]
            
            // Reshape mask: [batch, time, num_selected, channels, 2]
            let mask = mask.reshape([batch, time_frames, num_selected, channels, 2]);
            // Permute to match spec: [batch, channels, num_selected, time, 2]
            let mask = mask.permute([0, 3, 2, 1, 4]);
            
            // 6. Apply Mask to Selected Spec
            let spec_real: Tensor<B, 4> = spec_selected.clone().slice([0..batch, 0..channels, 0..num_selected, 0..time_frames, 0..1]).flatten(3, 4);
            let spec_imag: Tensor<B, 4> = spec_selected.clone().slice([0..batch, 0..channels, 0..num_selected, 0..time_frames, 1..2]).flatten(3, 4);
            
            let mask_real: Tensor<B, 4> = mask.clone().slice([0..batch, 0..channels, 0..num_selected, 0..time_frames, 0..1]).flatten(3, 4);
            let mask_imag: Tensor<B, 4> = mask.clone().slice([0..batch, 0..channels, 0..num_selected, 0..time_frames, 1..2]).flatten(3, 4);
            
            let out_real: Tensor<B, 4> = spec_real.clone().mul(mask_real.clone()).sub(spec_imag.clone().mul(mask_imag.clone()));
            let out_imag: Tensor<B, 4> = spec_real.mul(mask_imag).add(spec_imag.mul(mask_real));
            
            let out_selected: Tensor<B, 5> = Tensor::cat(vec![out_real.unsqueeze_dim(4), out_imag.unsqueeze_dim(4)], 4);
            
            // 7. Overlap-Add (Scatter)
            let out_perm = out_selected.permute([0, 1, 3, 4, 2]); // [B, C, T, 2, num_selected]
            let dims = out_perm.dims();
            let b = dims[0];
            let c = dims[1];
            let t = dims[2];
            let complex = dims[3];
            let ns = dims[4];
            
            let out_flat = out_perm.reshape([b * c * t * complex, ns]);
            
            let scatter = self.freq_scatter_matrix.clone(); // [num_selected, num_total]
            
            let out_scattered_flat = out_flat.matmul(scatter); // [..., num_total]
            
            let num_total = out_scattered_flat.dims()[1];
            let out_scattered = out_scattered_flat.reshape([b, c, t, complex, num_total]);
            
            // 8. Normalize
            let denom: Tensor<B, 5> = self.num_bands_per_freq.clone().reshape([1, 1, 1, 1, num_total]);
            let out_norm = out_scattered.div(denom);
            
            // Permute back: [batch, channels, freq, time, 2]
            let out_final = out_norm.permute([0, 1, 4, 2, 3]);
            
            // 9. ISTFT
            let waveform = self.istft(out_final, time_audio);
            stem_outputs.push(waveform.unsqueeze_dim(1));
        }
        
        // Stack stems: [batch, num_stems, channels, time]
        Tensor::cat(stem_outputs, 1)
    }
}
