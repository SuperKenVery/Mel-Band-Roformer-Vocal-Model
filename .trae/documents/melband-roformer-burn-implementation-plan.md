# Mel-Band-Roformer Burn Implementation Plan

## Overview

Implement GPU-accelerated inference for Mel-Band-Roformer in Rust using:
- **Burn** - Neural network framework with CubeCL backend
- **gpu-fft** - GPU-accelerated FFT (Cooley-Tukey O(n log n)) using CubeCL
- Quantization support (INT8, INT4, FP16)

All code goes to `rust/` directory.

---

## Architecture Analysis

### Model Structure (from Python reference)

```
Input Audio [batch, channels, samples]
    ↓
STFT (n_fft=2048, hop=441, win=2048)
    ↓
Band Split (60 mel bands → each has RMSNorm + Linear)
    ↓
┌─────────────────────────────────────────┐
│  depth=6 iterations:                     │
│    Time Transformer (depth=1)            │
│      - RMSNorm → Linear(QKV) → RoPE     │
│      - Attention (8 heads, dim_head=64) │
│      - Gates → Output projection        │
│      - FeedForward (dim=384, mult=4)    │
│    Freq Transformer (depth=1)            │
│      - Same structure as Time           │
└─────────────────────────────────────────┘
    ↓
Mask Estimator (per stem, per band: MLP with GLU)
    ↓
Apply Complex Masks + Scatter-Average
    ↓
ISTFT
    ↓
Output Audio [batch, stems, channels, samples]
```

### Key Parameters (from config)

| Parameter | Value |
|-----------|-------|
| dim | 384 |
| depth | 6 |
| time_transformer_depth | 1 |
| freq_transformer_depth | 1 |
| num_bands | 60 |
| heads | 8 |
| dim_head | 64 |
| stft_n_fft | 2048 |
| stft_hop_length | 441 |
| stft_win_length | 2048 |
| stereo | true |
| mask_estimator_depth | 2 |

---

## Project Structure

```
rust/
├── Cargo.toml
├── src/
│   ├── lib.rs                 # Library entry point
│   ├── main.rs                # CLI inference binary
│   │
│   ├── stft/
│   │   ├── mod.rs
│   │   ├── fft_plan.rs        # FFT plan with precomputed indices
│   │   ├── stft.rs            # STFT implementation
│   │   └── istft.rs           # ISTFT with overlap-add
│   │
│   ├── model/
│   │   ├── mod.rs
│   │   ├── config.rs          # Model hyperparameters
│   │   ├── mel_band_roformer.rs
│   │   ├── attention.rs       # Multi-head attention with RoPE
│   │   ├── transformer.rs     # Time/Freq transformer blocks
│   │   ├── band_split.rs      # Mel band splitting
│   │   ├── mask_estimator.rs  # MLP with GLU
│   │   ├── rms_norm.rs        # RMS normalization
│   │   └── rotary.rs          # Rotary position embeddings
│   │
│   ├── quantization/
│   │   ├── mod.rs
│   │   ├── q8_0.rs            # INT8 quantization
│   │   ├── q4_0.rs            # INT4 quantization
│   │   └── dequant.rs         # Dequantization kernels
│   │
│   ├── io/
│   │   ├── mod.rs
│   │   ├── weights.rs         # Load .ggml / safetensors weights
│   │   └── wav.rs             # WAV file I/O (use hound crate)
│   │
│   └── inference.rs           # High-level inference API
│
├── convert/
│   └── convert_checkpoint.py  # Python script to export weights
│
└── tests/
    ├── test_stft.rs           # STFT/ISTFT round-trip tests
    ├── test_attention.rs      # Attention correctness
    └── test_model.rs          # Full model integration test
```

---

## Implementation Phases

### Phase 1: Project Setup & FFT Foundation

**Goal**: Working STFT/ISTFT with gpu-fft integration

#### Tasks:

1. **Initialize Cargo project**
   ```toml
   [dependencies]
   burn = { version = "0.16", features = ["wgpu"] }
   gpu-fft = { git = "https://github.com/eugenehp/gpu-fft", features = ["wgpu"] }
   hound = "3.5"          # WAV I/O
   ndarray = "0.16"       # Array operations
   clap = "4"             # CLI
   ```

2. **Create FftPlan struct**
   - Precompute bit-reversal indices on init
   - Store indices on GPU for gather operations
   - Handle fixed n_fft sizes (2048) for inference

3. **Implement STFT**
   - Frame input with hop_length
   - Apply Hann window
   - GPU FFT via gpu-fft
   - Return complex tensor [batch, freq_bins, time_frames, 2]

4. **Implement ISTFT**
   - GPU IFFT
   - Overlap-add reconstruction
   - Window normalization

5. **Test**: STFT → ISTFT round-trip (verify reconstruction error < 1e-5)

### Phase 2: Core Model Components

**Goal**: Implement all building blocks in Burn

#### Tasks:

1. **RMSNorm**
   ```rust
   // x / ||x||_2 * sqrt(dim) * gamma
   fn forward(&self, x: Tensor<B, D>) -> Tensor<B, D>
   ```

2. **Rotary Position Embeddings**
   - Precompute sin/cos tables
   - Apply to Q and K in attention

3. **Multi-Head Attention**
   - Linear projection to Q, K, V
   - RoPE application
   - Scaled dot-product attention (use Burn's built-in if available)
   - Gated output projection

4. **FeedForward**
   - RMSNorm → Linear → GELU → Dropout → Linear → Dropout

5. **Transformer Block**
   - Attention + residual
   - FeedForward + residual
   - Output norm

6. **Test**: Compare outputs against PyTorch reference for each component

### Phase 3: Band Split & Mask Estimation

**Goal**: Frequency band processing

#### Tasks:

1. **Mel Filter Bank**
   - Precompute mel band boundaries (60 bands)
   - Store freq_indices for gathering
   - Handle stereo (interleaved frequencies)

2. **BandSplit Module**
   - Per-band: RMSNorm → Linear
   - Stack outputs [batch, time, bands, dim]

3. **MaskEstimator**
   - Per-band MLP with GLU activation
   - Output complex mask [batch, stems, freq, time, 2]

4. **Mask Application**
   - Scatter-add for overlapping bands
   - Average by num_bands_per_freq
   - Complex multiplication

### Phase 4: Full Model Assembly

**Goal**: Complete forward pass

#### Tasks:

1. **MelBandRoformer struct**
   - Hold all sub-modules
   - Implement forward pass matching Python exactly

2. **Inference pipeline**
   ```rust
   pub fn separate(
       &self,
       audio: &[f32],
       sample_rate: u32,
   ) -> Vec<Vec<f32>>  // stems
   ```

3. **Chunked processing**
   - Handle long audio with overlap
   - Match Python's `chunk_size=352800` and `num_overlap=2`

### Phase 5: Weight Loading & Quantization

**Goal**: Load pre-trained weights, support quantized inference

#### Tasks:

1. **GGML format loader**
   - Parse header (hyperparameters)
   - Load tensors with quantization type detection
   - Dequantize on load or keep quantized

2. **Safetensors support** (optional)
   - Alternative format for weights

3. **Quantization support**
   - Q8_0 dequantization (per-block INT8)
   - Q4_0 dequantization (per-block INT4)
   - FP16 support

4. **Python conversion script**
   - Export PyTorch weights to Burn-compatible format
   - Reuse existing `convert_checkpoint.py` logic

### Phase 6: CLI & Integration

**Goal**: Production-ready inference binary

#### Tasks:

1. **CLI interface**
   ```bash
   mel-band-roformer \
     --model model.ggml \
     --input input.wav \
     --output output_dir/ \
     --device gpu \
     --threads 8
   ```

2. **Batch processing**
   - Process multiple files
   - Progress reporting

3. **Performance optimization**
   - Profile hot paths
   - Optimize memory allocation
   - Tune GPU kernel launch parameters

---

## Technical Decisions

### 1. gpu-fft Integration

**Challenge**: gpu-fft returns `Vec<f32>` and manages its own device.

**Solution**: 
- Fork or contribute to gpu-fft to add:
  1. `FftPlan` struct with precomputed indices
  2. Handle-based API for GPU-resident data
  3. Integration point with Burn's tensor handles

- Alternatively: Implement O(n²) matrix DFT in Burn initially, optimize later.

### 2. Complex Number Handling

Burn doesn't have native complex tensor support.

**Solution**: Use real tensors with shape [..., 2] where last dim is [real, imag].
- Define helper functions for complex multiply, abs, etc.
- This matches PyTorch's `torch.view_as_real()` pattern.

### 3. Quantization in Burn

**Solution**: 
- Load quantized weights, dequantize to FP32/FP16 at load time
- Keep weights on GPU in dequantized form
- Future: Implement fused dequant+matmul kernels in CubeCL

### 4. Memory Management

For long audio, avoid OOM by:
- Processing in chunks (352800 samples = 8 seconds)
- Overlap-add at chunk boundaries
- Stream-allocate intermediate buffers

---

## Testing Strategy

1. **Unit tests**: Each component against PyTorch reference
2. **Integration test**: Full model output comparison
3. **Round-trip test**: STFT → ISTFT reconstruction
4. **Quantization test**: Verify acceptable quality degradation

### Test Data
- Generate synthetic test cases (sine waves)
- Use small real audio clips
- Compare SNR/SDR metrics

---

## Dependencies

| Crate | Purpose |
|-------|---------|
| `burn` | Neural network framework |
| `gpu-fft` | GPU FFT (may need fork) |
| `hound` | WAV file I/O |
| `clap` | CLI argument parsing |
| `ndarray` | Array utilities |
| `byteorder` | Binary file parsing |
| `serde` | Config serialization |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| gpu-fft doesn't integrate cleanly with Burn | Use matrix DFT initially; optimize later |
| Burn lacks required ops (scatter, gather) | Implement custom CubeCL kernels |
| Quantization accuracy loss | Test with real audio, provide multiple quant levels |
| Memory pressure on GPU | Implement chunked processing from start |

---

## Success Criteria

1. **Correctness**: Output matches PyTorch within acceptable tolerance (SNR > 40dB)
2. **Performance**: Faster than PyTorch CPU inference
3. **Usability**: Single binary, no Python runtime required
4. **Flexibility**: Support multiple quantization levels

---

## Estimated Effort

| Phase | Effort |
|-------|--------|
| Phase 1: FFT Foundation | Medium (gpu-fft integration unknowns) |
| Phase 2: Core Components | Medium (straightforward Burn port) |
| Phase 3: Band Processing | Low-Medium |
| Phase 4: Full Model | Low (assembly) |
| Phase 5: Quantization | Medium |
| Phase 6: CLI | Low |

**Total**: Significant project, recommend iterative milestones with testing at each phase.
