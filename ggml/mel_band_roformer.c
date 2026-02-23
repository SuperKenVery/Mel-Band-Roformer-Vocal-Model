#include "mel_band_roformer.h"
#include "ggml/ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper function to read int32 from file
static bool read_int32(FILE * f, int32_t * value) {
    return fread(value, sizeof(int32_t), 1, f) == 1;
}

// Helper function to get size of quantized tensor data
static size_t get_quantized_size(enum ggml_type type, size_t nelements) {
    switch (type) {
        case GGML_TYPE_F32:
            return nelements * sizeof(float);
        case GGML_TYPE_F16:
            return nelements * sizeof(uint16_t);
        case GGML_TYPE_Q8_0:
            // Q8_0: 32 elements per block, 1 float scale + 32 int8
            return ((nelements + 31) / 32) * (sizeof(float) + 32);
        case GGML_TYPE_Q4_0:
            // Q4_0: 32 elements per block, 1 f16 scale + 16 bytes (32 x 4-bit)
            return ((nelements + 31) / 32) * (sizeof(uint16_t) + 16);
        default:
            return nelements * sizeof(float);
    }
}

// Helper function to read tensor from file (with quantization support)
static bool read_tensor(FILE * f, struct ggml_context * ctx, struct ggml_tensor ** tensor) {
    int32_t n_dims;
    if (!read_int32(f, &n_dims)) {
        return false;
    }

    int32_t dims[4] = {1, 1, 1, 1};
    for (int i = 0; i < n_dims; i++) {
        if (!read_int32(f, &dims[i])) {
            return false;
        }
    }
    
    // Read quantization type (version 2+)
    int32_t quant_type_int;
    if (!read_int32(f, &quant_type_int)) {
        return false;
    }
    enum ggml_type quant_type = (enum ggml_type)quant_type_int;

    // Create tensor with appropriate type
    if (n_dims == 1) {
        *tensor = ggml_new_tensor_1d(ctx, quant_type, dims[0]);
    } else if (n_dims == 2) {
        *tensor = ggml_new_tensor_2d(ctx, quant_type, dims[0], dims[1]);
    } else if (n_dims == 3) {
        *tensor = ggml_new_tensor_3d(ctx, quant_type, dims[0], dims[1], dims[2]);
    } else if (n_dims == 4) {
        *tensor = ggml_new_tensor_4d(ctx, quant_type, dims[0], dims[1], dims[2], dims[3]);
    } else {
        fprintf(stderr, "Unsupported number of dimensions: %d\n", n_dims);
        return false;
    }

    // Read data (size depends on quantization)
    size_t nelements = ggml_nelements(*tensor);
    size_t nbytes = get_quantized_size(quant_type, nelements);
    
    if (fread((*tensor)->data, 1, nbytes, f) != nbytes) {
        return false;
    }

    return true;
}

// Load model from file
bool mel_band_roformer_model_load(const char * fname, struct mel_band_roformer_model * model) {
    FILE * f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", fname);
        return false;
    }

    // Read magic and version
    int32_t magic, version;
    if (!read_int32(f, &magic) || !read_int32(f, &version)) {
        fprintf(stderr, "Failed to read header\n");
        fclose(f);
        return false;
    }

    if (magic != 0x67676d6c) {
        fprintf(stderr, "Invalid magic number\n");
        fclose(f);
        return false;
    }

    // Read hyperparameters
    struct mel_band_roformer_hparams * hparams = &model->hparams;
    if (!read_int32(f, &hparams->dim) ||
        !read_int32(f, &hparams->depth) ||
        !read_int32(f, &hparams->num_stems) ||
        !read_int32(f, &hparams->time_transformer_depth) ||
        !read_int32(f, &hparams->freq_transformer_depth) ||
        !read_int32(f, &hparams->num_bands) ||
        !read_int32(f, &hparams->dim_head) ||
        !read_int32(f, &hparams->heads)) {
        fprintf(stderr, "Failed to read hyperparameters\n");
        fclose(f);
        return false;
    }

    int32_t stereo_int;
    if (!read_int32(f, &stereo_int) ||
        !read_int32(f, &hparams->mask_estimator_depth) ||
        !read_int32(f, &hparams->stft_n_fft) ||
        !read_int32(f, &hparams->stft_hop_length) ||
        !read_int32(f, &hparams->stft_win_length) ||
        !read_int32(f, &hparams->sample_rate)) {
        fprintf(stderr, "Failed to read STFT parameters\n");
        fclose(f);
        return false;
    }
    hparams->stereo = (stereo_int != 0);
    
    // Read quantization type (version 2+)
    int32_t quant_type = 0;
    if (version >= 2) {
        if (!read_int32(f, &quant_type)) {
            fprintf(stderr, "Failed to read quantization type\n");
            fclose(f);
            return false;
        }
        const char* quant_names[] = {"F32", "F16", "Q4_0", "Q4_1", "", "", "Q5_0", "Q5_1", "Q8_0", "Q8_1"};
        if (quant_type < 10) {
            printf("Model quantization: %s\n", quant_names[quant_type]);
        }
    }

    // Allocate GGML context
    size_t ctx_size = 0;
    ctx_size += 1024 * 1024 * 512; // 512 MB for weights

    struct ggml_init_params params = {
        .mem_size = ctx_size,
        .mem_buffer = NULL,
        .no_alloc = false,
    };

    model->ctx = ggml_init(params);
    if (!model->ctx) {
        fprintf(stderr, "Failed to initialize GGML context\n");
        fclose(f);
        return false;
    }
    model->ctx_size = ctx_size;

    // Read mel filter bank tensors
    if (!read_tensor(f, model->ctx, &model->freq_indices) ||
        !read_tensor(f, model->ctx, &model->freqs_per_band) ||
        !read_tensor(f, model->ctx, &model->num_freqs_per_band) ||
        !read_tensor(f, model->ctx, &model->num_bands_per_freq)) {
        fprintf(stderr, "Failed to read mel filter bank tensors\n");
        mel_band_roformer_model_free(model);
        fclose(f);
        return false;
    }

    // Read band split layers
    for (int i = 0; i < hparams->num_bands; i++) {
        if (!read_tensor(f, model->ctx, &model->band_split_norm_gamma[i]) ||
            !read_tensor(f, model->ctx, &model->band_split_linear_weight[i]) ||
            !read_tensor(f, model->ctx, &model->band_split_linear_bias[i])) {
            fprintf(stderr, "Failed to read band split layer %d\n", i);
            mel_band_roformer_model_free(model);
            fclose(f);
            return false;
        }
    }

    // Read transformer layers
    for (int layer_idx = 0; layer_idx < hparams->depth; layer_idx++) {
        // Time transformer
        for (int trans_idx = 0; trans_idx < hparams->time_transformer_depth; trans_idx++) {
            if (!read_tensor(f, model->ctx, &model->time_attn_norm_gamma[layer_idx][trans_idx]) ||
                !read_tensor(f, model->ctx, &model->time_attn_qkv_weight[layer_idx][trans_idx]) ||
                !read_tensor(f, model->ctx, &model->time_attn_gates_weight[layer_idx][trans_idx]) ||
                !read_tensor(f, model->ctx, &model->time_attn_gates_bias[layer_idx][trans_idx]) ||
                !read_tensor(f, model->ctx, &model->time_attn_out_weight[layer_idx][trans_idx]) ||
                !read_tensor(f, model->ctx, &model->time_ff_norm_gamma[layer_idx][trans_idx]) ||
                !read_tensor(f, model->ctx, &model->time_ff_linear1_weight[layer_idx][trans_idx]) ||
                !read_tensor(f, model->ctx, &model->time_ff_linear1_bias[layer_idx][trans_idx]) ||
                !read_tensor(f, model->ctx, &model->time_ff_linear2_weight[layer_idx][trans_idx]) ||
                !read_tensor(f, model->ctx, &model->time_ff_linear2_bias[layer_idx][trans_idx])) {
                fprintf(stderr, "Failed to read time transformer %d,%d\n", layer_idx, trans_idx);
                mel_band_roformer_model_free(model);
                fclose(f);
                return false;
            }
        }

        // Time transformer output norm
        if (!read_tensor(f, model->ctx, &model->time_transformer_norm_gamma[layer_idx])) {
            fprintf(stderr, "Failed to read time transformer norm %d\n", layer_idx);
            mel_band_roformer_model_free(model);
            fclose(f);
            return false;
        }

        // Freq transformer
        for (int trans_idx = 0; trans_idx < hparams->freq_transformer_depth; trans_idx++) {
            if (!read_tensor(f, model->ctx, &model->freq_attn_norm_gamma[layer_idx][trans_idx]) ||
                !read_tensor(f, model->ctx, &model->freq_attn_qkv_weight[layer_idx][trans_idx]) ||
                !read_tensor(f, model->ctx, &model->freq_attn_gates_weight[layer_idx][trans_idx]) ||
                !read_tensor(f, model->ctx, &model->freq_attn_gates_bias[layer_idx][trans_idx]) ||
                !read_tensor(f, model->ctx, &model->freq_attn_out_weight[layer_idx][trans_idx]) ||
                !read_tensor(f, model->ctx, &model->freq_ff_norm_gamma[layer_idx][trans_idx]) ||
                !read_tensor(f, model->ctx, &model->freq_ff_linear1_weight[layer_idx][trans_idx]) ||
                !read_tensor(f, model->ctx, &model->freq_ff_linear1_bias[layer_idx][trans_idx]) ||
                !read_tensor(f, model->ctx, &model->freq_ff_linear2_weight[layer_idx][trans_idx]) ||
                !read_tensor(f, model->ctx, &model->freq_ff_linear2_bias[layer_idx][trans_idx])) {
                fprintf(stderr, "Failed to read freq transformer %d,%d\n", layer_idx, trans_idx);
                mel_band_roformer_model_free(model);
                fclose(f);
                return false;
            }
        }

        // Freq transformer output norm
        if (!read_tensor(f, model->ctx, &model->freq_transformer_norm_gamma[layer_idx])) {
            fprintf(stderr, "Failed to read freq transformer norm %d\n", layer_idx);
            mel_band_roformer_model_free(model);
            fclose(f);
            return false;
        }
    }

    // Read mask estimators
    for (int stem_idx = 0; stem_idx < hparams->num_stems; stem_idx++) {
        for (int band_idx = 0; band_idx < hparams->num_bands; band_idx++) {
            for (int layer_idx = 0; layer_idx < hparams->mask_estimator_depth; layer_idx++) {
                if (!read_tensor(f, model->ctx, &model->mask_mlp_weight[stem_idx][band_idx][layer_idx]) ||
                    !read_tensor(f, model->ctx, &model->mask_mlp_bias[stem_idx][band_idx][layer_idx])) {
                    fprintf(stderr, "Failed to read mask estimator %d,%d,%d\n", stem_idx, band_idx, layer_idx);
                    mel_band_roformer_model_free(model);
                    fclose(f);
                    return false;
                }
            }
        }
    }

    fclose(f);
    return true;
}

// Free model
void mel_band_roformer_model_free(struct mel_band_roformer_model * model) {
    if (model->ctx) {
        ggml_free(model->ctx);
        model->ctx = NULL;
    }
}

// RMS Normalization
static struct ggml_tensor * rms_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * gamma
) {
    // norm = x / ||x||_2 * sqrt(dim) * gamma
    int dim = x->ne[0];
    float scale = sqrtf((float)dim);

    struct ggml_tensor * norm = ggml_norm(ctx, x, 1e-5f);
    norm = ggml_scale(ctx, norm, scale);
    norm = ggml_mul(ctx, norm, gamma);

    return norm;
}

// Rotary positional embeddings
static void apply_rotary_emb_inplace(float * q, float * k, int dim_head, int seq_len, int heads) {
    // Simplified rotary embeddings
    for (int h = 0; h < heads; h++) {
        for (int t = 0; t < seq_len; t++) {
            for (int d = 0; d < dim_head / 2; d++) {
                float theta = (float)t / powf(10000.0f, 2.0f * (float)d / (float)dim_head);
                float cos_theta = cosf(theta);
                float sin_theta = sinf(theta);

                int idx = h * seq_len * dim_head + t * dim_head + d * 2;
                float q0 = q[idx];
                float q1 = q[idx + 1];
                q[idx] = q0 * cos_theta - q1 * sin_theta;
                q[idx + 1] = q0 * sin_theta + q1 * cos_theta;

                float k0 = k[idx];
                float k1 = k[idx + 1];
                k[idx] = k0 * cos_theta - k1 * sin_theta;
                k[idx + 1] = k0 * sin_theta + k1 * cos_theta;
            }
        }
    }
}

// Attention layer
static struct ggml_tensor * attention(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * norm_gamma,
    struct ggml_tensor * qkv_weight,
    struct ggml_tensor * gates_weight,
    struct ggml_tensor * gates_bias,
    struct ggml_tensor * out_weight,
    int heads,
    int dim_head
) {
    // Normalize
    struct ggml_tensor * normed = rms_norm(ctx, x, norm_gamma);

    // Project to Q, K, V
    struct ggml_tensor * qkv = ggml_mul_mat(ctx, qkv_weight, normed);

    int dim = heads * dim_head;
    int seq_len = normed->ne[1];
    int batch = normed->ne[2];

    // Split into Q, K, V (simplified - would need proper reshaping)
    // This is a placeholder for the actual implementation
    struct ggml_tensor * attn_out = ggml_mul_mat(ctx, qkv_weight, normed);

    // Gates
    struct ggml_tensor * gates = ggml_mul_mat(ctx, gates_weight, normed);
    gates = ggml_add(ctx, gates, gates_bias);
    gates = ggml_sigmoid(ctx, gates);

    // Output projection
    attn_out = ggml_mul(ctx, attn_out, gates);
    attn_out = ggml_mul_mat(ctx, out_weight, attn_out);

    return attn_out;
}

// Feed-forward network
static struct ggml_tensor * feed_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * norm_gamma,
    struct ggml_tensor * linear1_weight,
    struct ggml_tensor * linear1_bias,
    struct ggml_tensor * linear2_weight,
    struct ggml_tensor * linear2_bias
) {
    // Normalize
    struct ggml_tensor * normed = rms_norm(ctx, x, norm_gamma);

    // First linear + GELU
    struct ggml_tensor * out = ggml_mul_mat(ctx, linear1_weight, normed);
    out = ggml_add(ctx, out, linear1_bias);
    out = ggml_gelu(ctx, out);

    // Second linear
    out = ggml_mul_mat(ctx, linear2_weight, out);
    out = ggml_add(ctx, out, linear2_bias);

    return out;
}

// Main inference function (placeholder - full STFT/ISTFT implementation needed)
bool mel_band_roformer_eval(
    const struct mel_band_roformer_model * model,
    int n_threads,
    const float * audio_data,
    int audio_length,
    int num_channels,
    float ** output_stems,
    int * output_length
) {
    const struct mel_band_roformer_hparams * hparams = &model->hparams;

    // Create computation context
    size_t compute_size = 1024 * 1024 * 1024; // 1 GB for computation
    struct ggml_init_params params = {
        .mem_size = compute_size,
        .mem_buffer = NULL,
        .no_alloc = false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    if (!ctx0) {
        fprintf(stderr, "Failed to initialize computation context\n");
        return false;
    }

    // TODO: Implement full STFT transform
    // TODO: Implement band splitting
    // TODO: Implement transformer forward pass
    // TODO: Implement mask estimation
    // TODO: Implement ISTFT

    fprintf(stderr, "Full inference not yet implemented\n");

    ggml_free(ctx0);
    return false;
}
