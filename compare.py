import librosa
import soundfile as sf
import numpy as np
import os
import torch
import yaml
from models.mel_band_roformer.mel_band_roformer import MelBandRoformer

def main():
    print("Loading config...")
    with open('configs/config_vocals_mel_band_roformer.yaml') as f:
        config = yaml.full_load(f)

    model_config = config['model']
    print("Initializing model...")
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
        flash_attn=False, # Disable flash attn for CPU/compatibility or simple inference
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
    print("Loading weights...")
    checkpoint = torch.load('/Users/bytedance/Downloads/MelBandRoformer.ckpt', map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    # Filter 'model.' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    # Load audio
    print("Loading audio...")
    audio_path = '/Users/bytedance/Desktop/歲月無聲.mp3'
    # Use librosa for loading and resampling
    waveform, sample_rate = librosa.load(audio_path, sr=44100, mono=False)
    
    # Ensure shape [channels, time]
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    
    # Ensure stereo if needed
    if waveform.shape[0] == 1:
        waveform = np.concatenate([waveform, waveform], axis=0)
        
    waveform = torch.from_numpy(waveform)

    # Extract 35s-40s
    start = 35 * 44100
    end = 40 * 44100
    segment = waveform[:, start:end]

    # Add batch dim: [1, 2, time]
    input_tensor = segment.unsqueeze(0)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        output = model(input_tensor)

    # Output is [1, 2, time] (since num_stems=1)
    output = output.squeeze(0) # [2, time]
    print(f"Python output shape: {output.shape}")

    # Save to wav
    print("Saving python output...")
    sf.write('python_output.wav', output.T.numpy(), 44100)

    # Load rust output
    print("Comparing with rust output...")
    if not os.path.exists('rust/output.wav'):
        print("Error: rust/output.wav not found")
        return

    rust_output, sr = sf.read('rust/output.wav')
    print(f"Rust output shape: {rust_output.shape}, sr: {sr}")
    print(f"Rust mean: {rust_output.mean()}, std: {rust_output.std()}")
    
    python_output_np = output.T.numpy()
    print(f"Python mean: {python_output_np.mean()}, std: {python_output_np.std()}")
    # rust_output is [time, 2]
    
    python_output = python_output_np
    
    # Ensure same length
    min_len = min(rust_output.shape[0], python_output.shape[0])
    rust_output = rust_output[:min_len]
    python_output = python_output[:min_len]

    # Compute diff
    diff = np.abs(python_output - rust_output)
    print(f"Max diff: {diff.max()}")
    print(f"Mean diff: {diff.mean()}")

    # Try alignment
    try:
        from scipy import signal
        print("Checking alignment...")
        
        # Use first channel
        r = rust_output[:, 0]
        p = python_output[:, 0]
        
        # Normalize for correlation
        r_norm = (r - r.mean()) / (r.std() + 1e-8)
        p_norm = (p - p.mean()) / (p.std() + 1e-8)
        
        correlation = signal.correlate(r_norm, p_norm, mode='full')
        max_corr = correlation.max() / len(r_norm)
        print(f"Max correlation: {max_corr}")
        lags = signal.correlation_lags(r.size, p.size, mode='full')
        lag = lags[np.argmax(correlation)]
        
        print(f"Optimal lag: {lag}")
        
        if lag > 0:
            r_aligned = rust_output[lag:]
            p_aligned = python_output[:len(r_aligned)]
        elif lag < 0:
            p_aligned = python_output[-lag:]
            r_aligned = rust_output[:len(p_aligned)]
        else:
            r_aligned = rust_output
            p_aligned = python_output
            
        # Truncate to match
        min_l = min(len(r_aligned), len(p_aligned))
        r_aligned = r_aligned[:min_l]
        p_aligned = p_aligned[:min_l]
        
        diff_aligned = np.abs(r_aligned - p_aligned)
        print(f"Aligned Max diff: {diff_aligned.max()}")
        print(f"Aligned Mean diff: {diff_aligned.mean()}")
        
        if diff_aligned.mean() < 1e-3:
             print("SUCCESS: Aligned outputs are close enough.")
        else:
             print("WARNING: Even aligned outputs are different.")
             
    except ImportError:
        print("Scipy not available for alignment check.")

    if diff.mean() < 1e-4:
        print("SUCCESS: Outputs are close enough.")
    else:
        print("WARNING: Outputs might be too different.")

if __name__ == "__main__":
    main()
