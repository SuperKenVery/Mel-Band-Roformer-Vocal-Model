#!/usr/bin/env python3
"""
Convert MelBandRoformer PyTorch checkpoint to safetensors format with
renamed keys compatible with Burn model structure.
"""

import argparse
import torch
from safetensors.torch import save_file
from collections import OrderedDict


def convert_key(key: str) -> str:
    """Convert PyTorch key name to Burn-compatible key name."""
    
    # layers.{depth}.{0=time,1=freq}.layers.{layer_idx}.{0=attn,1=ff}...
    # -> time_transformers.{depth}.layers.{layer_idx}.{attn,ff}...
    # -> freq_transformers.{depth}.layers.{layer_idx}.{attn,ff}...
    
    parts = key.split('.')
    
    if parts[0] == 'layers':
        depth = parts[1]
        time_or_freq = parts[2]  # 0 = time, 1 = freq
        
        if time_or_freq == '0':
            prefix = f'time_transformers.{depth}'
        else:
            prefix = f'freq_transformers.{depth}'
        
        rest = parts[3:]
        
        if rest[0] == 'layers':
            layer_idx = rest[1]
            attn_or_ff = rest[2]  # 0 = attn, 1 = ff
            component_rest = rest[3:]
            
            if attn_or_ff == '0':
                # Attention layer
                new_key = convert_attention_key(prefix, layer_idx, component_rest)
            else:
                # FeedForward layer
                new_key = convert_ff_key(prefix, layer_idx, component_rest)
        elif rest[0] == 'norm':
            # Final norm: time_transformers.{depth}.norm.gamma
            new_key = f'{prefix}.norm.{rest[1]}'
        else:
            new_key = f'{prefix}.' + '.'.join(rest)
            
        return new_key
    
    elif parts[0] == 'band_split':
        # band_split.to_features.{band_idx}.{0=norm,1=linear}...
        return convert_band_split_key(parts)
    
    elif parts[0] == 'mask_estimators':
        # mask_estimators.{stem_idx}.to_freqs.{band_idx}...
        return convert_mask_estimator_key(parts)
    
    return key


def convert_attention_key(prefix: str, layer_idx: str, rest: list) -> str:
    """Convert attention layer key."""
    # rest: [component, ...]
    # rotary_embed.freqs -> skip (we compute this)
    # norm.gamma -> attn.norm.gamma
    # to_qkv.weight -> attn.to_qkv.weight
    # to_gates.weight/bias -> attn.to_gates.weight/bias
    # to_out.0.weight -> attn.to_out.weight
    
    if rest[0] == 'rotary_embed':
        return None  # Skip rotary embeddings, we compute them
    
    component = rest[0]
    param = rest[1] if len(rest) > 1 else None
    
    if component == 'norm':
        return f'{prefix}.layers.{layer_idx}.attn.norm.{param}'
    elif component == 'to_qkv':
        return f'{prefix}.layers.{layer_idx}.attn.to_qkv.{param}'
    elif component == 'to_gates':
        return f'{prefix}.layers.{layer_idx}.attn.to_gates.{param}'
    elif component == 'to_out':
        # to_out.0.weight -> to_out.weight
        return f'{prefix}.layers.{layer_idx}.attn.to_out.{rest[2]}'
    
    return f'{prefix}.layers.{layer_idx}.attn.' + '.'.join(rest)


def convert_ff_key(prefix: str, layer_idx: str, rest: list) -> str:
    """Convert feedforward layer key."""
    # rest: [net, idx, param]
    # net.0.gamma -> ff.norm.gamma
    # net.1.weight/bias -> ff.linear1.weight/bias
    # net.4.weight/bias -> ff.linear2.weight/bias
    
    if rest[0] != 'net':
        return f'{prefix}.layers.{layer_idx}.ff.' + '.'.join(rest)
    
    idx = rest[1]
    param = rest[2] if len(rest) > 2 else None
    
    if idx == '0':
        # RMSNorm
        return f'{prefix}.layers.{layer_idx}.ff.norm.{param}'
    elif idx == '1':
        # First linear
        return f'{prefix}.layers.{layer_idx}.ff.linear1.{param}'
    elif idx == '4':
        # Second linear
        return f'{prefix}.layers.{layer_idx}.ff.linear2.{param}'
    
    return f'{prefix}.layers.{layer_idx}.ff.' + '.'.join(rest)


def convert_band_split_key(parts: list) -> str:
    """Convert band_split key."""
    # band_split.to_features.{band_idx}.{0=norm,1=linear}.{param}
    # -> band_split.to_features.{band_idx}.norm.{param}
    # -> band_split.to_features.{band_idx}.linear.{param}
    
    band_idx = parts[2]
    component_idx = parts[3]
    param = parts[4] if len(parts) > 4 else None
    
    if component_idx == '0':
        return f'band_split.to_features.{band_idx}.norm.{param}'
    else:
        return f'band_split.to_features.{band_idx}.linear.{param}'


def convert_mask_estimator_key(parts: list) -> str:
    """Convert mask_estimators key."""
    # mask_estimators.{stem_idx}.to_freqs.{band_idx}.{0=mlp}.{...}
    # The MLP structure: Linear, Tanh, Linear (for depth=1)
    # -> mask_estimators.{stem_idx}.to_freqs.{band_idx}.layers.{layer_idx}.{weight/bias}
    
    stem_idx = parts[1]
    band_idx = parts[3]
    
    if len(parts) <= 4:
        return '.'.join(parts)
    
    # parts[4] is usually '0' (the MLP sequential index before GLU)
    # parts[5] is the layer index in MLP
    # parts[6] is weight/bias
    
    if parts[4] == '0':
        mlp_layer_idx = parts[5]
        param = parts[6] if len(parts) > 6 else None
        return f'mask_estimators.{stem_idx}.to_freqs.{band_idx}.mlp.{mlp_layer_idx}.{param}'
    
    return '.'.join(parts)


def convert_checkpoint(input_path: str, output_path: str):
    """Convert PyTorch checkpoint to safetensors."""
    print(f"Loading checkpoint from {input_path}")
    state_dict = torch.load(input_path, map_location='cpu', weights_only=True)
    
    new_state_dict = OrderedDict()
    skipped = []
    
    for key, value in state_dict.items():
        new_key = convert_key(key)
        
        if new_key is None:
            skipped.append(key)
            continue
        
        # Transpose weight matrices for Linear layers (PyTorch uses [out, in], Burn uses [in, out])
        if 'weight' in new_key and value.dim() == 2:
            value = value.t().contiguous()
        
        new_state_dict[new_key] = value
    
    print(f"Converted {len(new_state_dict)} tensors")
    print(f"Skipped {len(skipped)} tensors: {skipped[:5]}...")
    
    print(f"Saving to {output_path}")
    save_file(new_state_dict, output_path)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Convert MelBandRoformer checkpoint to safetensors")
    parser.add_argument("-i", "--input", required=True, help="Input PyTorch checkpoint path")
    parser.add_argument("-o", "--output", required=True, help="Output safetensors path")
    args = parser.parse_args()
    
    convert_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()
