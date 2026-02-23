#!/usr/bin/env python3
"""
Convert MelBand RoFormer PyTorch checkpoint to GGML format.
Supports quantization to INT4, INT8, or FP16 for reduced model size.
"""

import argparse
import os
import struct
import sys

import numpy as np
import torch
import yaml
from ml_collections import ConfigDict

# Add parent directory to path to import model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_model_from_config

# GGML quantization types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9


def quantize_q8_0(data):
    """
    Quantize to Q8_0 format (8-bit integers with per-block scale).
    Block size: 32 elements.
    Format: [scale (f32)] [32 x int8]
    """
    block_size = 32
    n_blocks = (data.size + block_size - 1) // block_size
    data_flat = data.flatten()

    # Pad to multiple of block_size
    padded_size = n_blocks * block_size
    if data_flat.size < padded_size:
        data_flat = np.pad(
            data_flat, (0, padded_size - data_flat.size), mode="constant"
        )

    quantized = bytearray()

    for i in range(n_blocks):
        block = data_flat[i * block_size : (i + 1) * block_size]

        # Compute scale
        amax = np.abs(block).max()
        scale = amax / 127.0 if amax > 0 else 1.0

        # Quantize
        q = np.round(block / scale).astype(np.int8)

        # Write scale (f32) and quantized values (32 x i8)
        quantized.extend(struct.pack("f", scale))
        quantized.extend(q.tobytes())

    return quantized


def quantize_q4_0(data):
    """
    Quantize to Q4_0 format (4-bit integers with per-block scale).
    Block size: 32 elements.
    Format: [scale (f16)] [16 bytes for 32 x 4-bit values]
    """
    block_size = 32
    n_blocks = (data.size + block_size - 1) // block_size
    data_flat = data.flatten()

    # Pad to multiple of block_size
    padded_size = n_blocks * block_size
    if data_flat.size < padded_size:
        data_flat = np.pad(
            data_flat, (0, padded_size - data_flat.size), mode="constant"
        )

    quantized = bytearray()

    for i in range(n_blocks):
        block = data_flat[i * block_size : (i + 1) * block_size]

        # Compute scale
        amax = np.abs(block).max()
        scale = amax / 7.0 if amax > 0 else 1.0

        # Quantize to 4-bit (range: -7 to 7)
        q = np.round(block / scale).astype(np.int8)
        q = np.clip(q, -7, 7)

        # Pack two 4-bit values into one byte
        q_packed = np.zeros(16, dtype=np.uint8)
        for j in range(16):
            low = q[j * 2] & 0x0F
            high = q[j * 2 + 1] & 0x0F
            q_packed[j] = (high << 4) | low

        # Write scale (f16) and packed values
        scale_f16 = np.float16(scale)
        quantized.extend(scale_f16.tobytes())
        quantized.extend(q_packed.tobytes())

    return quantized


def quantize_f16(data):
    """Quantize to FP16."""
    return data.astype(np.float16).tobytes()


def write_header(fout, config, quant_type):
    """Write GGML header with model configuration."""
    # Magic number for GGML MelBand Roformer
    fout.write(struct.pack("i", 0x67676D6C))  # "ggml" in hex
    fout.write(struct.pack("i", 2))  # version 2 (with quantization)

    # Model hyperparameters
    fout.write(struct.pack("i", config.model.dim))
    fout.write(struct.pack("i", config.model.depth))
    fout.write(struct.pack("i", config.model.num_stems))
    fout.write(struct.pack("i", config.model.time_transformer_depth))
    fout.write(struct.pack("i", config.model.freq_transformer_depth))
    fout.write(struct.pack("i", config.model.num_bands))
    fout.write(struct.pack("i", config.model.dim_head))
    fout.write(struct.pack("i", config.model.heads))
    fout.write(struct.pack("i", 1 if config.model.stereo else 0))
    fout.write(struct.pack("i", config.model.mask_estimator_depth))

    # STFT parameters
    fout.write(struct.pack("i", config.model.stft_n_fft))
    fout.write(struct.pack("i", config.model.stft_hop_length))
    fout.write(struct.pack("i", config.model.stft_win_length))
    fout.write(struct.pack("i", config.model.sample_rate))

    # Quantization type
    fout.write(struct.pack("i", quant_type))


def write_tensor(fout, tensor, name=None, quant_type=GGML_TYPE_F32, force_f32=False):
    """Write a tensor to file in GGML format with optional quantization."""
    if name:
        quant_name = {
            GGML_TYPE_F32: "F32",
            GGML_TYPE_F16: "F16",
            GGML_TYPE_Q8_0: "Q8_0",
            GGML_TYPE_Q4_0: "Q4_0",
        }.get(quant_type if not force_f32 else GGML_TYPE_F32, "F32")
        print(f"Writing tensor [{quant_name}]: {name} with shape {tensor.shape}")

    # Convert to float32 numpy array
    data = tensor.detach().cpu().numpy().astype(np.float32)

    # Write tensor dimensions
    n_dims = len(data.shape)
    fout.write(struct.pack("i", n_dims))
    for dim in data.shape:
        fout.write(struct.pack("i", dim))

    # Determine actual quantization type
    actual_quant_type = GGML_TYPE_F32 if force_f32 else quant_type
    fout.write(struct.pack("i", actual_quant_type))

    # Quantize and write tensor data
    if force_f32 or quant_type == GGML_TYPE_F32:
        data.tofile(fout)
    elif quant_type == GGML_TYPE_F16:
        fout.write(quantize_f16(data))
    elif quant_type == GGML_TYPE_Q8_0:
        fout.write(quantize_q8_0(data))
    elif quant_type == GGML_TYPE_Q4_0:
        fout.write(quantize_q4_0(data))
    else:
        raise ValueError(f"Unsupported quantization type: {quant_type}")


def convert_model(model, config, output_path, quant_type=GGML_TYPE_F32):
    """Convert PyTorch model to GGML format with optional quantization."""

    print(
        f"Quantization type: {['F32', 'F16', 'Q4_0', 'Q4_1', '', '', 'Q5_0', 'Q5_1', 'Q8_0', 'Q8_1'][quant_type]}"
    )

    # Calculate and display expected size reduction
    if quant_type == GGML_TYPE_F16:
        print("Expected size: ~50% of original (FP16)")
    elif quant_type == GGML_TYPE_Q8_0:
        print("Expected size: ~25% of original (INT8)")
    elif quant_type == GGML_TYPE_Q4_0:
        print("Expected size: ~12.5% of original (INT4)")

    with open(output_path, "wb") as fout:
        # Write header
        write_header(fout, config, quant_type)

        # Write mel filter bank tensors (keep in F32 for precision)
        write_tensor(
            fout, model.freq_indices.cpu(), "freq_indices", quant_type, force_f32=True
        )
        write_tensor(
            fout,
            model.freqs_per_band.float().cpu(),
            "freqs_per_band",
            quant_type,
            force_f32=True,
        )
        write_tensor(
            fout,
            model.num_freqs_per_band.cpu(),
            "num_freqs_per_band",
            quant_type,
            force_f32=True,
        )
        write_tensor(
            fout,
            model.num_bands_per_freq.cpu(),
            "num_bands_per_freq",
            quant_type,
            force_f32=True,
        )

        # Write band split layers
        for i, to_feature in enumerate(model.band_split.to_features):
            # RMSNorm (keep F32)
            write_tensor(
                fout,
                to_feature[0].gamma,
                f"band_split.{i}.norm.gamma",
                quant_type,
                force_f32=True,
            )
            # Linear (quantize weights, keep bias F32)
            write_tensor(
                fout, to_feature[1].weight, f"band_split.{i}.linear.weight", quant_type
            )
            write_tensor(
                fout,
                to_feature[1].bias,
                f"band_split.{i}.linear.bias",
                quant_type,
                force_f32=True,
            )

        # Write transformer layers
        for layer_idx, (time_transformer, freq_transformer) in enumerate(model.layers):
            # Time transformer
            for trans_idx, (attn, ff) in enumerate(time_transformer.layers):
                prefix = f"time_transformer.{layer_idx}.{trans_idx}"

                # Attention
                write_tensor(
                    fout,
                    attn.norm.gamma,
                    f"{prefix}.attn.norm.gamma",
                    quant_type,
                    force_f32=True,
                )
                write_tensor(
                    fout, attn.to_qkv.weight, f"{prefix}.attn.to_qkv.weight", quant_type
                )
                write_tensor(
                    fout,
                    attn.to_gates.weight,
                    f"{prefix}.attn.to_gates.weight",
                    quant_type,
                )
                write_tensor(
                    fout,
                    attn.to_gates.bias,
                    f"{prefix}.attn.to_gates.bias",
                    quant_type,
                    force_f32=True,
                )
                write_tensor(
                    fout,
                    attn.to_out[0].weight,
                    f"{prefix}.attn.to_out.weight",
                    quant_type,
                )

                # FeedForward
                write_tensor(
                    fout,
                    ff.net[0].gamma,
                    f"{prefix}.ff.norm.gamma",
                    quant_type,
                    force_f32=True,
                )
                write_tensor(
                    fout, ff.net[1].weight, f"{prefix}.ff.linear1.weight", quant_type
                )
                write_tensor(
                    fout,
                    ff.net[1].bias,
                    f"{prefix}.ff.linear1.bias",
                    quant_type,
                    force_f32=True,
                )
                write_tensor(
                    fout, ff.net[4].weight, f"{prefix}.ff.linear2.weight", quant_type
                )
                write_tensor(
                    fout,
                    ff.net[4].bias,
                    f"{prefix}.ff.linear2.bias",
                    quant_type,
                    force_f32=True,
                )

            # Time transformer output norm
            if hasattr(time_transformer.norm, "gamma"):
                write_tensor(
                    fout,
                    time_transformer.norm.gamma,
                    f"time_transformer.{layer_idx}.norm.gamma",
                    quant_type,
                    force_f32=True,
                )

            # Freq transformer
            for trans_idx, (attn, ff) in enumerate(freq_transformer.layers):
                prefix = f"freq_transformer.{layer_idx}.{trans_idx}"

                # Attention
                write_tensor(
                    fout,
                    attn.norm.gamma,
                    f"{prefix}.attn.norm.gamma",
                    quant_type,
                    force_f32=True,
                )
                write_tensor(
                    fout, attn.to_qkv.weight, f"{prefix}.attn.to_qkv.weight", quant_type
                )
                write_tensor(
                    fout,
                    attn.to_gates.weight,
                    f"{prefix}.attn.to_gates.weight",
                    quant_type,
                )
                write_tensor(
                    fout,
                    attn.to_gates.bias,
                    f"{prefix}.attn.to_gates.bias",
                    quant_type,
                    force_f32=True,
                )
                write_tensor(
                    fout,
                    attn.to_out[0].weight,
                    f"{prefix}.attn.to_out.weight",
                    quant_type,
                )

                # FeedForward
                write_tensor(
                    fout,
                    ff.net[0].gamma,
                    f"{prefix}.ff.norm.gamma",
                    quant_type,
                    force_f32=True,
                )
                write_tensor(
                    fout, ff.net[1].weight, f"{prefix}.ff.linear1.weight", quant_type
                )
                write_tensor(
                    fout,
                    ff.net[1].bias,
                    f"{prefix}.ff.linear1.bias",
                    quant_type,
                    force_f32=True,
                )
                write_tensor(
                    fout, ff.net[4].weight, f"{prefix}.ff.linear2.weight", quant_type
                )
                write_tensor(
                    fout,
                    ff.net[4].bias,
                    f"{prefix}.ff.linear2.bias",
                    quant_type,
                    force_f32=True,
                )

            # Freq transformer output norm
            if hasattr(freq_transformer.norm, "gamma"):
                write_tensor(
                    fout,
                    freq_transformer.norm.gamma,
                    f"freq_transformer.{layer_idx}.norm.gamma",
                    quant_type,
                    force_f32=True,
                )

        # Write mask estimators
        for stem_idx, mask_estimator in enumerate(model.mask_estimators):
            for band_idx, mlp in enumerate(mask_estimator.to_freqs):
                # MLP layers
                for layer_idx in range(0, len(mlp[0]) - 1, 2):  # Skip GLU
                    write_tensor(
                        fout,
                        mlp[0][layer_idx].weight,
                        f"mask_estimator.{stem_idx}.{band_idx}.mlp.{layer_idx // 2}.weight",
                        quant_type,
                    )
                    write_tensor(
                        fout,
                        mlp[0][layer_idx].bias,
                        f"mask_estimator.{stem_idx}.{band_idx}.mlp.{layer_idx // 2}.bias",
                        quant_type,
                        force_f32=True,
                    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nModel converted successfully to {output_path}")
    print(f"File size: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MelBand RoFormer checkpoint to GGML format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quantization types:
  f32   - Full precision (no quantization, largest file)
  f16   - Half precision (~50% size reduction, minimal quality loss)
  q8_0  - 8-bit quantization (~75% size reduction, slight quality loss)
  q4_0  - 4-bit quantization (~87.5% size reduction, noticeable quality loss)

Example:
  # Full precision (largest, best quality)
  python convert_checkpoint.py --quant f32 --config_path config.yaml --model_path model.ckpt --output_path model.ggml

  # Recommended: INT8 (good balance)
  python convert_checkpoint.py --quant q8_0 --config_path config.yaml --model_path model.ckpt --output_path model-q8.ggml

  # Smallest: INT4 (fastest, lowest quality)
  python convert_checkpoint.py --quant q4_0 --config_path config.yaml --model_path model.ckpt --output_path model-q4.ggml
""",
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to PyTorch checkpoint"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Output path for GGML model"
    )
    parser.add_argument(
        "--model_type", type=str, default="mel_band_roformer", help="Model type"
    )
    parser.add_argument(
        "--quant",
        type=str,
        default="f32",
        choices=["f32", "f16", "q8_0", "q4_0"],
        help="Quantization type (default: f32)",
    )

    args = parser.parse_args()

    # Map quantization string to GGML type
    quant_map = {
        "f32": GGML_TYPE_F32,
        "f16": GGML_TYPE_F16,
        "q8_0": GGML_TYPE_Q8_0,
        "q4_0": GGML_TYPE_Q4_0,
    }
    quant_type = quant_map[args.quant]

    # Load config
    with open(args.config_path) as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    # Load model
    print(f"Loading model from {args.model_path}")
    model = get_model_from_config(args.model_type, config)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device("cpu")))
    model.eval()

    # Convert
    print(f"Converting model to GGML format...")
    convert_model(model, config, args.output_path, quant_type)

    print("\nConversion complete!")
    print(f"\nTo use this model:")
    print(
        f"  ./mel_band_roformer_inference -m {args.output_path} -i input.wav -o output_dir"
    )


if __name__ == "__main__":
    main()
