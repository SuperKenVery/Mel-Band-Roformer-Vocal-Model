---
name: "compare-rust-python"
description: "Compare intermediate outputs between Python and Rust MelBand Roformer implementations. Invoke when debugging divergence between implementations."
---

# Compare Rust vs Python Outputs

## Quick Start

```bash
# Step 1: Generate Python intermediates
python debug_compare.py

# Step 2: Generate Rust intermediates
cd rust && cargo run --bin debug_compare

# Step 3: Compare
python compare_rust_python.py

# Step 4 (optional): Compare final wav outputs
python compare_outputs.py -p /path/to/python.wav -r /path/to/rust.wav
```

## How It Works

- `debug_compare.py` → saves `.npy` to `/tmp/python_debug/`
- `debug_compare.rs` → saves `.bin` (raw f32) to `/tmp/rust_debug/`
- `compare_rust_python.py` → loads both, computes diff, reports "MATCH!" if < 1e-4

## Debugging Strategy

Binary search through the pipeline. When divergence found, drill down:

```
STFT → Band Split → Transformer Layers → Mask Estimators → Output
                         ↓
              Time Attn → Time FF → Freq Attn → Freq FF
                                        ↓
                              Norm → QKV → Rotary → Softmax → Out
```

## Adding New Debug Points

**Python** (`debug_compare.py`):
```python
module.register_forward_hook(hook("my_tensor"))
```

**Rust** (`debug_compare.rs`):
```rust
let data: Vec<f32> = tensor.clone().into_data().to_vec().unwrap();
let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes().to_vec()).collect();
std::fs::write("/tmp/rust_debug/my_tensor.bin", bytes).unwrap();
```

**Compare** (`compare_rust_python.py`):
```python
compare("my_tensor")
```

## Common Issues

1. **Shape mismatch**: Check transpose/reshape - Rust may have different dim ordering
2. **Weight loading**: Use `check_weights.rs` to verify
3. **Numerical precision**: < 1e-5 diff is normal, focus on large divergences
4. **Dropout**: Ensure disabled in both for deterministic comparison
