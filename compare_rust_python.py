
import numpy as np
import os

def stats(name, arr):
    print(f"  {name}: shape={arr.shape}, mean={arr.mean():.6f}, std={arr.std():.6f}, min={arr.min():.6f}, max={arr.max():.6f}")

def compare(name, shape=None):
    py_path = f"/tmp/python_debug/{name}.npy"
    rust_path = f"/tmp/rust_debug/{name}.bin"
    
    if not os.path.exists(py_path):
        print(f"Skipping {name}: Python file not found")
        return
    if not os.path.exists(rust_path):
        print(f"Skipping {name}: Rust file not found")
        return
        
    py_data = np.load(py_path)
    
    # Rust data is flat f32 bytes
    with open(rust_path, "rb") as f:
        rust_data = np.frombuffer(f.read(), dtype=np.float32)
        
    if shape is None:
        shape = py_data.shape
        
    try:
        rust_data = rust_data.reshape(shape)
    except ValueError:
        print(f"Shape mismatch for {name}: Python {py_data.shape}, Rust {rust_data.shape} (flat)")
        return
        
    print(f"\nComparing {name}:")
    stats("Python", py_data)
    stats("Rust  ", rust_data)
    
    diff = np.abs(py_data - rust_data)
    stats("Diff  ", diff)
    
    if diff.max() > 1e-4:
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"  Max diff at {idx}: {diff[idx]:.6f}")
        print(f"  Python: {py_data[idx]:.6f}")
        print(f"  Rust:   {rust_data[idx]:.6f}")
    else:
        print("  MATCH!")

def main():
    compare("x_before_band_split")
    compare("x_after_band_split")

    compare("layer0_time_attn")
    compare("layer0_time_ff")
    
    compare("layer0_time_output")
    compare("layer0_time_norm_gamma")

    compare("layer0_freq_attn_input")
    compare("layer0_freq_attn_norm_gamma")
    compare("layer0_freq_attn_norm")
    compare("layer0_freq_attn_to_qkv")
    compare("layer0_freq_attn")
    
    compare("layer0_freq_ff_norm")
    compare("layer0_freq_ff_linear1")
    compare("layer0_freq_ff")

    compare("band59_norm")
    compare("band59_out")

if __name__ == "__main__":
    main()
