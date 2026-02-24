import soundfile as sf
import numpy as np

print('Loading files...')
py_instr, _ = sf.read('/Users/bytedance/Desktop/music-instrumental-extract/outputs/syws_instrumental.wav', 
                       start=0, stop=44100*30)
rust_instr, _ = sf.read('/tmp/rust_test_output3/instrumental.wav',
                        start=0, stop=44100*30)

print(f'Python shape: {py_instr.shape}')
print(f'Rust shape: {rust_instr.shape}')

print(f'Python instrumental RMS: {np.sqrt(np.mean(py_instr**2)):.6f}')
print(f'Rust instrumental RMS: {np.sqrt(np.mean(rust_instr**2)):.6f}')

instr_diff = rust_instr - py_instr
print(f'Diff RMS: {np.sqrt(np.mean(instr_diff**2)):.6f}')
print(f'Ratio (diff/py): {np.sqrt(np.mean(instr_diff**2)) / np.sqrt(np.mean(py_instr**2)):.4f}')
