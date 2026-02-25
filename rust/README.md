# Mel-Band-Roformer Rust Implementation

GPU-accelerated audio source separation using the Mel-Band-Roformer model, implemented in Rust with the [Burn](https://burn.dev/) deep learning framework.

## Prerequisites

- Rust 2024 edition
- A PyTorch checkpoint (`.ckpt` or `.pt`) of the MelBandRoformer model
- Model config YAML file

## Workflow

### Step 1: Convert PyTorch checkpoint to Burn format (once)

```bash
cargo run --release --bin convert -- \
    -i /path/to/model.ckpt \
    -o model.bpk \
    -c ../configs/config_vocals_mel_band_roformer.yaml
```

This converts the PyTorch weights to Burn's native `.bpk` format. You only need to do this once per model.

### Step 2: Run inference

```bash
cargo run --release -- \
    -m model.bpk \
    -i input.wav \
    -o output/ \
    -c ../configs/config_vocals_mel_band_roformer.yaml
```

Options:
- `-m, --model` - Path to the Burn model file (`.bpk`)
- `-i, --input` - Input audio file (WAV format)
- `-o, --output` - Output directory for separated stems
- `-c, --config` - Path to config YAML file (optional if config is next to model)
- `-d, --device` - Device to use: `gpu` (default) or `cpu`
- `--chunk-size` - Chunk size for processing (default: 352800)
- `--num-overlap` - Number of overlapping chunks (default: 2)

### Output

The inference produces an `instrumental.wav` file in the output directory, which is the input audio with vocals removed.

## Building

```bash
# Debug build
cargo build

# Release build (recommended for inference)
cargo build --release
```

## Project Structure

```
rust/
├── src/
│   ├── main.rs           # Inference CLI
│   ├── lib.rs            # Library exports
│   ├── inference.rs      # InferenceEngine implementation
│   ├── bin/
│   │   └── convert.rs    # PyTorch→Burn converter
│   ├── model/            # Neural network components
│   │   ├── mel_band_roformer.rs
│   │   ├── attention.rs
│   │   ├── transformer.rs
│   │   └── ...
│   ├── stft/             # STFT/ISTFT implementation
│   └── io/               # WAV and config loading
└── Cargo.toml
```
