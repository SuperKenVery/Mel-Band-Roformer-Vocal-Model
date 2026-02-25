---
name: "compare-rust-python"
description: "Compare intermediate outputs between Python and Rust MelBand Roformer implementations. Invoke when debugging divergence between implementations."
---

# Compare Rust vs Python Intermediate Outputs

This skill documents how to compare intermediate tensor outputs between the Python reference implementation and the Rust implementation of MelBand Roformer.

## Overview

The debugging workflow uses two scripts that dump intermediate tensors to `/tmp/`:
- **Python**: `debug_compare.py` → saves `.npy` files to `/tmp/python_debug/`
- **Rust**: `rust/src/bin/debug_compare.rs` → saves `.bin` files to `/tmp/rust_debug/`

Then `compare_rust_python.py` loads both and computes diff statistics.

## Step 1: Generate Python Intermediates

```bash
python debug_compare.py
```

This script:
1. Loads the model from `configs/config_vocals_mel_band_roformer.yaml`
2. Processes a specific audio chunk (35s to ~43s of the test audio)
3. Uses PyTorch forward hooks to capture intermediate tensors
4. Saves `.npy` files to `/tmp/python_debug/`

**Key outputs:**
| File | Description |
|------|-------------|
| `input.npy` | Raw audio input |
| `stft_repr.npy` | STFT representation |
| `x_before_band_split.npy` | Input to band split module |
| `x_after_band_split.npy` | Output of band split module |
| `x_after_layer_0.npy` | Output after first transformer layer |
| `layer0_time_attn.npy` | Time attention output (layer 0) |
| `layer0_time_ff.npy` | Time feedforward output (layer 0) |
| `layer0_freq_attn.npy` | Freq attention output (layer 0) |
| `layer0_freq_ff.npy` | Freq feedforward output (layer 0) |
| `layer0_freq_attn_norm.npy` | Attention norm output |
| `layer0_freq_attn_to_qkv.npy` | QKV projection output |
| `masks.npy` | Mask estimator outputs |
| `output.npy` | Final model output |

## Step 2: Generate Rust Intermediates

```bash
cd rust
cargo run --bin debug_compare
```

This script:
1. Loads the same model and config
2. Processes the same audio chunk
3. Manually runs through the pipeline, printing tensor statistics
4. Saves `.bin` files (raw f32 little-endian bytes) to `/tmp/rust_debug/`

**Key outputs:**
| File | Description |
|------|-------------|
| `x_after_layer_0.bin` | Output after first transformer layer |
| `layer0_freq_attn.bin` | Freq attention output |
| `layer0_freq_ff.bin` | Freq feedforward output |
| ... | (similar to Python outputs) |

## Step 3: Compare Outputs

```bash
python compare_rust_python.py
```

The comparison script:
1. Loads Python `.npy` from `/tmp/python_debug/`
2. Loads Rust `.bin` (raw f32 bytes) from `/tmp/rust_debug/`
3. Reshapes Rust data to match Python shape
4. Computes statistics: mean, std, min, max
5. Computes absolute difference
6. Reports max diff location and values
7. Reports "MATCH!" if max diff < 1e-4

### Example Output

```
Comparing layer0_freq_ff:
  Python: shape=(801, 60, 512), mean=0.012345, std=0.234567, min=-1.234, max=2.345
  Rust  : shape=(801, 60, 512), mean=0.012340, std=0.234560, min=-1.234, max=2.345
  Diff  : shape=(801, 60, 512), mean=0.000001, std=0.000002, min=0.000, max=0.000012
  MATCH!
```

## Debugging Strategy

Use a **binary search approach** through the model pipeline:

```
1. STFT
   └── stft_repr.npy
2. Band Split
   └── x_after_band_split.npy
3. Transformer Layers (for each layer i)
   ├── Time Transformer
   │   ├── layer{i}_time_attn.npy
   │   └── layer{i}_time_ff.npy
   └── Freq Transformer
       ├── layer{i}_freq_attn.npy
       │   ├── layer{i}_freq_attn_norm.npy
       │   └── layer{i}_freq_attn_to_qkv.npy
       └── layer{i}_freq_ff.npy
           ├── layer{i}_freq_ff_norm.npy
           └── layer{i}_freq_ff_linear1.npy
4. Mask Estimators
   └── masks.npy
5. Final Output
   └── output.npy
```

When divergence is found at a high level, drill down into sub-components to isolate the bug.

## Adding New Debug Points

### Python Side (debug_compare.py)

Add a hook to capture intermediate output:

```python
def hook(name):
    def fn(module, input, output):
        np.save(f"{output_dir}/{name}.npy", output.cpu().numpy())
    return fn

# Register hook
module.register_forward_hook(hook("my_tensor_name"))
```

### Rust Side (debug_compare.rs)

Save tensor data as raw f32 bytes:

```rust
let data: Vec<f32> = tensor.clone().into_data().to_vec().unwrap();
let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes().to_vec()).collect();
std::fs::write("/tmp/rust_debug/my_tensor_name.bin", bytes).unwrap();
```

### Comparison Script (compare_rust_python.py)

Add to the `main()` function:

```python
compare("my_tensor_name")
```

The `compare()` function automatically:
- Loads from `/tmp/python_debug/{name}.npy`
- Loads from `/tmp/rust_debug/{name}.bin`
- Reshapes and computes diff statistics

## File Format Details

| Format | Python | Rust |
|--------|--------|------|
| Extension | `.npy` | `.bin` |
| Data type | numpy array | raw f32 little-endian |
| Shape info | embedded in .npy | inferred from Python |
| Load method | `np.load()` | `np.frombuffer(..., dtype=np.float32)` |

## Common Issues

1. **Shape mismatch**: Rust tensor may have different dimension ordering. Check transpose/reshape operations.

2. **Weight loading**: Ensure weights are loaded correctly. Use `check_weights.rs` to verify.

3. **Numerical precision**: Small differences (< 1e-5) are expected due to floating-point operations. Focus on large divergences.

4. **Dropout**: Ensure dropout is disabled in both implementations for deterministic comparison.

## End-to-End Comparison

To compare final wav file outputs (after running full inference):

```bash
python compare_outputs.py --python /path/to/python_output.wav --rust /path/to/rust_output.wav
```

Options:
- `--seconds N`: Only compare first N seconds
- `--sample-rate SR`: Sample rate for seconds calculation (default: 44100)

Example:
```bash
# Compare first 30 seconds
python compare_outputs.py \
    -p /path/to/python_instrumental.wav \
    -r /path/to/rust_instrumental.wav \
    --seconds 30
```

The script reports:
- RMS (root mean square) for each file
- Diff RMS and ratio to Python RMS
- Correlation coefficient
- MATCH/CLOSE/SIGNIFICANT DIFFERENCE verdict
