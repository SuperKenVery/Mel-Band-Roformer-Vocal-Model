#!/usr/bin/env python3
"""
Extract freqs_per_bands from a MelBandRoformer checkpoint.
"""
import sys
import torch

def main():
    if len(sys.argv) < 2:
        print("Usage: python get_band_dims.py <checkpoint_path>", file=sys.stderr)
        sys.exit(1)
    
    ckpt_path = sys.argv[1]
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    band_split_dims = []
    for i in range(100):
        key = f'band_split.to_features.{i}.1.weight'
        if key in ckpt:
            weight = ckpt[key]
            dim_in = weight.shape[1]
            band_split_dims.append(dim_in)
        else:
            break
    
    print(f"# Number of bands: {len(band_split_dims)}")
    print(f"# Total input size: {sum(band_split_dims)}")
    print(f"freqs_per_bands: {band_split_dims}")

if __name__ == "__main__":
    main()
