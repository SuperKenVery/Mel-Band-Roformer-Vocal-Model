import torch
import yaml
import soundfile as sf
import numpy as np
from models.mel_band_roformer.mel_band_roformer import MelBandRoformer

def main():
    config_path = 'configs/config_vocals_mel_band_roformer.yaml'
    model_path = '/Users/bytedance/Downloads/MelBandRoformer.ckpt'
    audio_path = '/Users/bytedance/Desktop/歲月無聲.mp3'
    output_path = 'python_output.wav'

    print("Loading config...")
    with open(config_path, 'r') as f:
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

    print("Loading weights...")
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # Filter keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Force CPU for consistency if needed, but GPU is preferred
    # device = torch.device('cpu') 
    model.to(device)

    print("Loading audio...")
    audio, sr = sf.read(audio_path)
    # Resample if needed
    if sr != 44100:
        print(f"Resampling from {sr} to 44100...")
        import librosa
        audio = librosa.resample(audio.T, orig_sr=sr, target_sr=44100).T
        sr = 44100
        
    start_sec = 32
    duration_sec = 10
    start_sample = start_sec * sr
    end_sample = start_sample + duration_sec * sr
    
    audio_slice = audio[start_sample:end_sample]
    
    # Prepare input: [batch, channels, time]
    # audio is [time, channels]
    x = torch.tensor(audio_slice.T).unsqueeze(0).float().to(device)
    
    print("Running inference...")
    with torch.no_grad():
        out = model(x)
        
    # Output: [batch, stems, channels, time]
    # stems=1 (vocals)
    out = out.squeeze(0).squeeze(0) # [channels, time]
    
    print("Saving output...")
    sf.write(output_path, out.cpu().numpy().T, sr)
    print("Done.")

if __name__ == "__main__":
    main()
