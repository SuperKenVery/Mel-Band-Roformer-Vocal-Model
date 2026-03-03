import torch
import yaml
import os
import re
from safetensors.torch import save_file
from models.mel_band_roformer.mel_band_roformer import MelBandRoformer

# Configuration
CONFIG_PATH = 'configs/config_vocals_mel_band_roformer.yaml'
CHECKPOINT_PATH = '/Users/bytedance/Downloads/MelBandRoformer.ckpt'
OUTPUT_DIR = 'rust'

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load config
    print(f"Loading config from {CONFIG_PATH}...")
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.full_load(f)

    model_config = config['model']

    # Instantiate model
    print("Instantiating model...")
    model = MelBandRoformer(
        dim=model_config['dim'],
        depth=model_config['depth'],
        stereo=model_config['stereo'],
        num_stems=model_config['num_stems'],
        time_transformer_depth=model_config['time_transformer_depth'],
        freq_transformer_depth=model_config['freq_transformer_depth'],
        num_bands=model_config['num_bands'],
        dim_head=model_config['dim_head'],
        heads=model_config['heads'],
        attn_dropout=model_config['attn_dropout'],
        ff_dropout=model_config['ff_dropout'],
        flash_attn=model_config['flash_attn'],
        dim_freqs_in=model_config['dim_freqs_in'],
        sample_rate=model_config['sample_rate'],
        stft_n_fft=model_config['stft_n_fft'],
        stft_hop_length=model_config['stft_hop_length'],
        stft_win_length=model_config['stft_win_length'],
        stft_normalized=model_config['stft_normalized'],
        mask_estimator_depth=model_config['mask_estimator_depth'],
        multi_stft_resolution_loss_weight=model_config['multi_stft_resolution_loss_weight'],
        multi_stft_resolutions_window_sizes=tuple(model_config['multi_stft_resolutions_window_sizes']),
        multi_stft_hop_size=model_config['multi_stft_hop_size'],
        multi_stft_normalized=model_config['multi_stft_normalized'],
        match_input_audio_length=False 
    )

    # Load weights
    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}")
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Filter out unrelated keys if any (e.g. from a LightningModule wrapper)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict

    # Rename keys
    renamed_state_dict = {}

    mapping_rules = [
        (r'band_split\.to_features\.(\d+)\.0\.gamma', r'band_split.to_features.\1.norm.gamma'),
        (r'band_split\.to_features\.(\d+)\.1\.(weight|bias)', r'band_split.to_features.\1.linear.\2'),
        (r'layers\.(\d+)\.0\.layers\.(\d+)\.0\.(.*)', r'time_transformers.\1.layers.\2.\3'),
        (r'layers\.(\d+)\.0\.layers\.(\d+)\.1\.net\.0\.(.*)', r'time_transformers.\1.ff_layers.\2.norm.\3'),
        (r'layers\.(\d+)\.0\.layers\.(\d+)\.1\.net\.1\.(.*)', r'time_transformers.\1.ff_layers.\2.linear1.\3'),
        (r'layers\.(\d+)\.0\.layers\.(\d+)\.1\.net\.4\.(.*)', r'time_transformers.\1.ff_layers.\2.linear2.\3'),
        (r'layers\.(\d+)\.0\.norm\.(.*)', r'time_transformers.\1.norm.\2'),
        
        # Frequency Transformers
        (r'layers\.(\d+)\.1\.layers\.(\d+)\.0\.(.*)', r'freq_transformers.\1.layers.\2.\3'),
        (r'layers\.(\d+)\.1\.layers\.(\d+)\.1\.net\.0\.(.*)', r'freq_transformers.\1.ff_layers.\2.norm.\3'),
        (r'layers\.(\d+)\.1\.layers\.(\d+)\.1\.net\.1\.(.*)', r'freq_transformers.\1.ff_layers.\2.linear1.\3'),
        (r'layers\.(\d+)\.1\.layers\.(\d+)\.1\.net\.4\.(.*)', r'freq_transformers.\1.ff_layers.\2.linear2.\3'),
        (r'layers\.(\d+)\.1\.norm\.(.*)', r'freq_transformers.\1.norm.\2'),
        
        # Mask Estimators
        (r'mask_estimators\.(\d+)\.to_freqs\.(\d+)\.0\.0\.(.*)', r'mask_estimators.\1.to_freqs.\2.mlp.layers.0.\3'),
        (r'mask_estimators\.(\d+)\.to_freqs\.(\d+)\.0\.2\.(.*)', r'mask_estimators.\1.to_freqs.\2.mlp.layers.1.\3'),
        (r'mask_estimators\.(\d+)\.to_freqs\.(\d+)\.0\.4\.(.*)', r'mask_estimators.\1.to_freqs.\2.mlp.layers.2.\3'),
        # Fix to_out.0.weight -> to_out.weight for Attention
        (r'(.*)\.to_out\.0\.(.*)', r'\1.to_out.\2'),
    ]

    print("Renaming keys...")
    for key, value in state_dict.items():
        # Skip buffers that will be in constants
        if key in ['freq_indices', 'freqs_per_band', 'num_freqs_per_band', 'num_bands_per_freq', 'mel_filter_bank']:
            continue
        
        # Skip rotary embedding buffers (they are shared and usually fixed)
        if 'rotary_embed' in key:
            continue

        new_key = key
        matched = False
        for pattern, replacement in mapping_rules:
            if re.match(pattern, key):
                new_key = re.sub(pattern, replacement, key)
                matched = True
                break
        
        # Post-processing fixes
        if 'to_out.0.' in new_key:
             new_key = new_key.replace('to_out.0.', 'to_out.')
        
        renamed_state_dict[new_key] = value
        # if not matched and 'weight' in key:
        #     print(f"Warning: Key not matched: {key}")

    # Generate STFT Kernels first so we can add them to model weights
    print("Generating STFT kernels...")
    n_fft = model_config['stft_n_fft']
    win_length = model_config['stft_win_length']
    n_freqs = n_fft // 2 + 1

    # Window
    window = torch.hann_window(win_length)
    
    # DFT Matrix
    # W[k, n] = exp(-i * 2 * pi * k * n / N)
    k = torch.arange(n_freqs).unsqueeze(1) # [n_freqs, 1]
    n = torch.arange(win_length).unsqueeze(0) # [1, win_length]
    angle = -2 * torch.pi * k * n / n_fft
    dft_real = torch.cos(angle)
    dft_imag = torch.sin(angle)

    # Multiply by window
    # Kernel = W * window
    # Broadcasting window [1, win_length]
    stft_kernel_real = dft_real * window.unsqueeze(0)
    stft_kernel_imag = dft_imag * window.unsqueeze(0)

    # Reshape to [n_freqs, 1, win_length]
    stft_kernel_real = stft_kernel_real.unsqueeze(1)
    stft_kernel_imag = stft_kernel_imag.unsqueeze(1)

    # Add to renamed_state_dict
    renamed_state_dict['stft_conv_real.weight'] = stft_kernel_real
    renamed_state_dict['stft_conv_imag.weight'] = stft_kernel_imag
    
    # ISTFT Kernels
    # For ConvTranspose1d, Burn expects [in_channels, out_channels, kernel_size]
    # Here in=n_freqs, out=1. So [n_freqs, 1, win_length].
    renamed_state_dict['istft_conv_real.weight'] = stft_kernel_real.clone()
    renamed_state_dict['istft_conv_imag.weight'] = stft_kernel_imag.clone()

    print(f"Renamed {len(renamed_state_dict)} keys.")
    
    # Save as .pt for burn-import
    torch.save(renamed_state_dict, os.path.join(OUTPUT_DIR, 'model.pt'))
    print(f"Saved model weights to {os.path.join(OUTPUT_DIR, 'model.pt')}")

    save_file(renamed_state_dict, os.path.join(OUTPUT_DIR, 'model.safetensors'))
    print(f"Saved model weights to {os.path.join(OUTPUT_DIR, 'model.safetensors')}")

    # Constants
    print("Generating constants...")
    constants = {}

    # Extract from model
    # constants['freq_indices'] = model.freq_indices # This is stereo-expanded if stereo=True
    
    # We want base freq_indices (before stereo expansion)
    # Reconstruct it or extract from model buffer before expansion?
    # Model doesn't store base indices.
    # But we can reconstruct: num_freqs_per_band tells us how many freqs in each band.
    # And we know they are in order of frequencies?
    # Actually, model.freq_indices is created from repeated_freq_indices[freqs_per_band].
    
    # Let's recreate logic to get base indices
    freqs_per_band = model.freqs_per_band # [num_bands, num_freqs]
    # repeated_freq_indices = repeat(torch.arange(n_freqs), 'f -> b f', b=model_config['num_bands'])
    repeated_freq_indices = torch.arange(n_freqs).unsqueeze(0).expand(model_config['num_bands'], -1)
    base_freq_indices = repeated_freq_indices[freqs_per_band] # [num_selected]
    
    print(f"Freq Indices: {base_freq_indices[:10]}")
    constants['freq_indices'] = base_freq_indices
    constants['num_bands_per_freq'] = model.num_bands_per_freq.float()
    constants['num_freqs_per_band'] = model.num_freqs_per_band.float()
    constants['window'] = window
    
    # Also keep kernels in constants for now as requested/optional
    constants['stft_kernel_real'] = stft_kernel_real
    constants['stft_kernel_imag'] = stft_kernel_imag
    constants['istft_kernel_real'] = stft_kernel_real.clone()
    constants['istft_kernel_imag'] = stft_kernel_imag.clone()

    save_file(constants, os.path.join(OUTPUT_DIR, 'constants.safetensors'))
    print(f"Saved constants to {os.path.join(OUTPUT_DIR, 'constants.safetensors')}")

if __name__ == '__main__':
    main()
