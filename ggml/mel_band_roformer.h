#ifndef MEL_BAND_ROFORMER_H
#define MEL_BAND_ROFORMER_H

#include "ggml/ggml.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Model hyperparameters
struct mel_band_roformer_hparams {
    int32_t dim;
    int32_t depth;
    int32_t num_stems;
    int32_t time_transformer_depth;
    int32_t freq_transformer_depth;
    int32_t num_bands;
    int32_t dim_head;
    int32_t heads;
    bool stereo;
    int32_t mask_estimator_depth;

    // STFT parameters
    int32_t stft_n_fft;
    int32_t stft_hop_length;
    int32_t stft_win_length;
    int32_t sample_rate;
};

// Model weights
struct mel_band_roformer_model {
    struct mel_band_roformer_hparams hparams;

    // Mel filter bank
    struct ggml_tensor * freq_indices;
    struct ggml_tensor * freqs_per_band;
    struct ggml_tensor * num_freqs_per_band;
    struct ggml_tensor * num_bands_per_freq;

    // Band split
    struct ggml_tensor * band_split_norm_gamma[60];  // max num_bands
    struct ggml_tensor * band_split_linear_weight[60];
    struct ggml_tensor * band_split_linear_bias[60];

    // Time and frequency transformers
    struct ggml_tensor * time_attn_norm_gamma[6][2];  // [depth][time_transformer_depth]
    struct ggml_tensor * time_attn_qkv_weight[6][2];
    struct ggml_tensor * time_attn_gates_weight[6][2];
    struct ggml_tensor * time_attn_gates_bias[6][2];
    struct ggml_tensor * time_attn_out_weight[6][2];

    struct ggml_tensor * time_ff_norm_gamma[6][2];
    struct ggml_tensor * time_ff_linear1_weight[6][2];
    struct ggml_tensor * time_ff_linear1_bias[6][2];
    struct ggml_tensor * time_ff_linear2_weight[6][2];
    struct ggml_tensor * time_ff_linear2_bias[6][2];

    struct ggml_tensor * time_transformer_norm_gamma[6];

    struct ggml_tensor * freq_attn_norm_gamma[6][2];
    struct ggml_tensor * freq_attn_qkv_weight[6][2];
    struct ggml_tensor * freq_attn_gates_weight[6][2];
    struct ggml_tensor * freq_attn_gates_bias[6][2];
    struct ggml_tensor * freq_attn_out_weight[6][2];

    struct ggml_tensor * freq_ff_norm_gamma[6][2];
    struct ggml_tensor * freq_ff_linear1_weight[6][2];
    struct ggml_tensor * freq_ff_linear1_bias[6][2];
    struct ggml_tensor * freq_ff_linear2_weight[6][2];
    struct ggml_tensor * freq_ff_linear2_bias[6][2];

    struct ggml_tensor * freq_transformer_norm_gamma[6];

    // Mask estimators
    struct ggml_tensor * mask_mlp_weight[4][60][4];  // [num_stems][num_bands][depth]
    struct ggml_tensor * mask_mlp_bias[4][60][4];

    // GGML context
    struct ggml_context * ctx;
    size_t ctx_size;
};

// Load model from GGML file
bool mel_band_roformer_model_load(const char * fname, struct mel_band_roformer_model * model);

// Free model
void mel_band_roformer_model_free(struct mel_band_roformer_model * model);

// Inference
bool mel_band_roformer_eval(
    const struct mel_band_roformer_model * model,
    int n_threads,
    const float * audio_data,
    int audio_length,
    int num_channels,
    float ** output_stems,
    int * output_length
);

#ifdef __cplusplus
}
#endif

#endif // MEL_BAND_ROFORMER_H
