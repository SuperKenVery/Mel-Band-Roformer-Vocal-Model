import torch
import numpy as np
import soundfile as sf
import yaml
from ml_collections import ConfigDict
from models.mel_band_roformer import MelBandRoformer

def dump_intermediates(audio_path, model_path, config_path, output_dir="/tmp/python_debug"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    with open(config_path) as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
    
    model = MelBandRoformer(**dict(config.model))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    mix, sr = sf.read(audio_path)
    if len(mix.shape) == 1:
        mix = np.stack([mix, mix], axis=-1)
    
    chunk_size = 352800
    start_sec = 35
    start_sample = start_sec * sr
    chunk = mix[start_sample:start_sample + chunk_size].T
    print(f"Using chunk from {start_sec}s to {start_sec + chunk_size/sr:.1f}s")
    raw_audio = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
    
    print(f"Input shape: {raw_audio.shape}")
    np.save(f"{output_dir}/input.npy", raw_audio.numpy())
    
    def hook(name):
        def fn(module, input, output):
            print(f"{name}: mean={output.mean().item():.6f}, std={output.std().item():.6f}")
            if name == "Layer 0 Freq 0 FF":
                np.save(f"{output_dir}/layer0_freq_ff.npy", output.cpu().numpy())
            if name == "Layer 0 Time 0 FF":
                np.save(f"{output_dir}/layer0_time_ff.npy", output.cpu().numpy())
            if name == "Layer 0 Freq 0 FF Linear1":
                np.save(f"{output_dir}/layer0_freq_ff_linear1.npy", output.cpu().numpy())
            if name == "Layer 0 Freq 0 FF Norm":
                np.save(f"{output_dir}/layer0_freq_ff_norm.npy", output.cpu().numpy())
            if name == "Layer 0 Freq 0 Attn":
                np.save(f"{output_dir}/layer0_freq_attn.npy", output.cpu().numpy())
            if name == "Layer 0 Time 0 Attn":
                np.save(f"{output_dir}/layer0_time_attn.npy", output.cpu().numpy())
            if name == "Layer 0 Freq 0 Attn Norm":
                np.save(f"{output_dir}/layer0_freq_attn_norm.npy", output.cpu().numpy())
            if name == "Layer 0 Freq 0 Attn to_qkv":
                np.save(f"{output_dir}/layer0_freq_attn_to_qkv.npy", output.cpu().numpy())
        return fn

    for i, (time_transformer, freq_transformer) in enumerate(model.layers):
        # Time Transformer
        for j, layer in enumerate(time_transformer.layers):
            attn, ff = layer
            attn.register_forward_hook(hook(f"Layer {i} Time {j} Attn"))
            
            # Print QKV weight stats
            w = attn.to_qkv.weight
            print(f"Layer {i} Time {j} Attn QKV Weight: shape={w.shape}, mean={w.mean().item():.6f}, std={w.std().item():.6f}")

            # Print Norm Gamma
            g = attn.norm.gamma
            print(f"Layer {i} Time {j} Attn Norm Gamma: shape={g.shape}, mean={g.mean().item():.6f}, std={g.std().item():.6f}")

            ff.register_forward_hook(hook(f"Layer {i} Time {j} FF"))
        
        # Freq Transformer
        for j, layer in enumerate(freq_transformer.layers):
            attn, ff = layer
            attn.register_forward_hook(hook(f"Layer {i} Freq {j} Attn"))
            
            # Hook Attn Norm and to_qkv
            attn.norm.register_forward_hook(hook(f"Layer {i} Freq {j} Attn Norm"))
            attn.to_qkv.register_forward_hook(hook(f"Layer {i} Freq {j} Attn to_qkv"))
            
            # Print QKV weight stats
            w = attn.to_qkv.weight
            print(f"Layer {i} Freq {j} Attn QKV Weight: shape={w.shape}, mean={w.mean().item():.6f}, std={w.std().item():.6f}")

            # Print Out weight stats
            w = attn.to_out[0].weight
            print(f"Layer {i} Freq {j} Attn Out Weight: shape={w.shape}, mean={w.mean().item():.6f}, std={w.std().item():.6f}")

            # Print FF Linear1 bias
            b = ff.net[1].bias
            print(f"Layer {i} Freq {j} FF Linear1 Bias: shape={b.shape}, mean={b.mean().item():.6f}, std={b.std().item():.6f}")

            ff.register_forward_hook(hook(f"Layer {i} Freq {j} FF"))
            # Hook linear1 of FF
            # ff is FeedForward, which has .net
            # net[1] is linear1
            # print(f"FF type: {type(ff)}")
            # Hook RMSNorm of FF
            if hasattr(ff, "net"):
                ff.net[0].register_forward_hook(hook(f"Layer {i} Freq {j} FF Norm"))
                
            if hasattr(ff, "net"):
                ff.net[1].register_forward_hook(hook(f"Layer {i} Freq {j} FF Linear1"))
                
                # Print weight stats
                w = ff.net[1].weight
                print(f"Layer {i} Freq {j} FF Linear1 Weight: shape={w.shape}, mean={w.mean().item():.6f}, std={w.std().item():.6f}")
            else:
                print(f"FF has no net attribute: {ff}")

    with torch.no_grad():
        device = raw_audio.device
        
        if raw_audio.ndim == 2:
            raw_audio = raw_audio.unsqueeze(1)
        
        batch, channels, raw_audio_length = raw_audio.shape
        print(f"After reshape: batch={batch}, channels={channels}, length={raw_audio_length}")
        
        from einops import pack, unpack, rearrange
        raw_audio_packed, batch_audio_channel_packed_shape = pack([raw_audio], "* t")
        print(f"Packed shape: {raw_audio_packed.shape}")
        
        stft_window = model.stft_window_fn(device=device)
        print(f"STFT window shape: {stft_window.shape}, sum: {stft_window.sum().item():.6f}")
        np.save(f"{output_dir}/stft_window.npy", stft_window.numpy())
        
        stft_repr = torch.stft(
            raw_audio_packed, **model.stft_kwargs, window=stft_window, return_complex=True
        )
        print(f"STFT complex shape: {stft_repr.shape}")
        
        stft_repr = torch.view_as_real(stft_repr)
        print(f"STFT real shape: {stft_repr.shape}")
        np.save(f"{output_dir}/stft_repr_before_unpack.npy", stft_repr.numpy())
        
        stft_repr = unpack(stft_repr, batch_audio_channel_packed_shape, "* f t c")[0]
        print(f"STFT unpacked shape: {stft_repr.shape}")
        
        stft_repr = rearrange(stft_repr, "b s f t c -> b (f s) t c")
        print(f"STFT rearranged shape: {stft_repr.shape}")
        np.save(f"{output_dir}/stft_repr.npy", stft_repr.numpy())
        
        batch_arange = torch.arange(batch, device=device)[..., None]
        freq_indices = model.freq_indices
        print(f"freq_indices shape: {freq_indices.shape}, first 20: {freq_indices[:20].tolist()}")
        np.save(f"{output_dir}/freq_indices.npy", freq_indices.numpy())
        
        x = stft_repr[batch_arange, freq_indices]
        print(f"Gathered x shape: {x.shape}")
        
        x = rearrange(x, "b f t c -> b t (f c)")
        print(f"x before band_split: {x.shape}")
        np.save(f"{output_dir}/x_before_band_split.npy", x.numpy())
        
        x = model.band_split(x)
        print(f"x after band_split: {x.shape}")
        np.save(f"{output_dir}/x_after_band_split.npy", x.numpy())
        
        for layer_idx, (time_transformer, freq_transformer) in enumerate(model.layers):
            x_before = x.clone()
            
            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            x = time_transformer(x)
            (x,) = unpack(x, ps, "* t d")
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            x = freq_transformer(x)
            (x,) = unpack(x, ps, "* f d")
            
            print(f"After layer {layer_idx}: mean={x.mean().item():.6f}, std={x.std().item():.6f}")
            if layer_idx == 0:
                np.save(f"{output_dir}/x_after_layer_{layer_idx}.npy", x.numpy())
        
        print(f"x before mask_estimators: {x.shape}")
        np.save(f"{output_dir}/x_before_mask.npy", x.numpy())
        
        masks = torch.stack([fn(x) for fn in model.mask_estimators], dim=1)
        print(f"masks shape: {masks.shape}")
        np.save(f"{output_dir}/masks.npy", masks.numpy())
        
        masks = rearrange(masks, "b n t (f c) -> b n f t c", c=2)
        print(f"masks reshaped: {masks.shape}")
        
        stft_repr_for_mask = rearrange(stft_repr, "b f t c -> b 1 f t c")
        stft_repr_complex = torch.view_as_complex(stft_repr_for_mask)
        masks_complex = torch.view_as_complex(masks)
        masks_complex = masks_complex.type(stft_repr_complex.dtype)
        
        from einops import repeat, reduce
        num_stems = len(model.mask_estimators)
        scatter_indices = repeat(
            freq_indices,
            "f -> b n f t",
            b=batch,
            n=num_stems,
            t=stft_repr_complex.shape[-1],
        )
        
        stft_repr_expanded = repeat(stft_repr_for_mask, "b 1 ... -> b n ...", n=num_stems)
        stft_repr_expanded_complex = torch.view_as_complex(stft_repr_expanded)
        
        masks_summed_real = torch.zeros_like(stft_repr_expanded_complex.real).scatter_add_(
            2, scatter_indices, masks_complex.real
        )
        masks_summed_imag = torch.zeros_like(stft_repr_expanded_complex.imag).scatter_add_(
            2, scatter_indices, masks_complex.imag
        )
        masks_summed = torch.complex(masks_summed_real, masks_summed_imag)
        
        denom = repeat(model.num_bands_per_freq, "f -> (f r) 1", r=channels)
        masks_averaged = masks_summed / denom.clamp(min=1e-8)
        
        print(f"masks_averaged shape: {masks_averaged.shape}")
        np.save(f"{output_dir}/masks_averaged_real.npy", masks_averaged.real.numpy())
        np.save(f"{output_dir}/masks_averaged_imag.npy", masks_averaged.imag.numpy())
        
        stft_out = stft_repr_complex * masks_averaged
        print(f"stft_out shape: {stft_out.shape}")
        np.save(f"{output_dir}/stft_out_real.npy", stft_out.real.numpy())
        np.save(f"{output_dir}/stft_out_imag.npy", stft_out.imag.numpy())
        
        stft_out = rearrange(stft_out, "b n (f s) t -> (b n s) f t", s=model.audio_channels)
        
        recon = torch.istft(
            stft_out,
            **model.stft_kwargs,
            window=stft_window,
            return_complex=False,
            length=raw_audio_length,
        )
        
        recon = rearrange(recon, "(b n s) t -> b n s t", b=batch, s=model.audio_channels, n=num_stems)
        print(f"Final output shape: {recon.shape}")
        np.save(f"{output_dir}/output.npy", recon.numpy())
        
        print(f"\nOutput stats:")
        print(f"  min: {recon.min().item():.6f}")
        print(f"  max: {recon.max().item():.6f}")
        print(f"  mean: {recon.mean().item():.6f}")
        print(f"  std: {recon.std().item():.6f}")
        
        print(f"\nDumped all intermediates to {output_dir}")
        return recon

if __name__ == "__main__":
    dump_intermediates(
        "/Users/bytedance/Desktop/music-instrumental-extract/inputs/syws.wav",
        "/Users/bytedance/Downloads/MelBandRoformer.ckpt",
        "configs/config_vocals_mel_band_roformer.yaml"
    )
