import soundfile as sf
import numpy as np
import argparse
import os

def stats(name, arr):
    rms = np.sqrt(np.mean(arr**2))
    print(f"  {name}: shape={arr.shape}, rms={rms:.6f}, min={arr.min():.6f}, max={arr.max():.6f}")

def compare_wav(py_path, rust_path, max_samples=None):
    if not os.path.exists(py_path):
        print(f"Python file not found: {py_path}")
        return
    if not os.path.exists(rust_path):
        print(f"Rust file not found: {rust_path}")
        return
    
    py_audio, py_sr = sf.read(py_path)
    rust_audio, rust_sr = sf.read(rust_path)
    
    if max_samples:
        py_audio = py_audio[:max_samples]
        rust_audio = rust_audio[:max_samples]
    
    print(f"\nComparing: {os.path.basename(py_path)} vs {os.path.basename(rust_path)}")
    print(f"  Sample rates: Python={py_sr}, Rust={rust_sr}")
    
    if py_sr != rust_sr:
        print("  WARNING: Sample rates differ!")
    
    stats("Python", py_audio)
    stats("Rust  ", rust_audio)
    
    min_len = min(len(py_audio), len(rust_audio))
    if len(py_audio) != len(rust_audio):
        print(f"  WARNING: Length mismatch (Python={len(py_audio)}, Rust={len(rust_audio)}), comparing first {min_len} samples")
        py_audio = py_audio[:min_len]
        rust_audio = rust_audio[:min_len]
    
    diff = rust_audio - py_audio
    stats("Diff  ", diff)
    
    py_rms = np.sqrt(np.mean(py_audio**2))
    diff_rms = np.sqrt(np.mean(diff**2))
    ratio = diff_rms / py_rms if py_rms > 0 else float('inf')
    print(f"  Diff/Python RMS ratio: {ratio:.6f}")
    
    if ratio < 0.01:
        print("  MATCH! (< 1% difference)")
    elif ratio < 0.05:
        print("  CLOSE (< 5% difference)")
    else:
        print("  SIGNIFICANT DIFFERENCE")
    
    if py_audio.ndim == 1:
        corr = np.corrcoef(py_audio, rust_audio)[0, 1]
    else:
        corr = np.corrcoef(py_audio.flatten(), rust_audio.flatten())[0, 1]
    print(f"  Correlation: {corr:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Compare Python and Rust audio outputs")
    parser.add_argument("--python", "-p", required=True, help="Path to Python output wav")
    parser.add_argument("--rust", "-r", required=True, help="Path to Rust output wav")
    parser.add_argument("--seconds", "-s", type=float, default=None, help="Only compare first N seconds")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Sample rate for --seconds calculation")
    args = parser.parse_args()
    
    max_samples = int(args.seconds * args.sample_rate) if args.seconds else None
    compare_wav(args.python, args.rust, max_samples)

if __name__ == "__main__":
    main()
