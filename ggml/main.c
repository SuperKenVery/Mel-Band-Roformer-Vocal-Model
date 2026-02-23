#include "mel_band_roformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// Simple WAV file reader (assumes 16-bit PCM)
typedef struct {
    int sample_rate;
    int num_channels;
    int num_samples;
    float * data;
} wav_data_t;

static bool read_wav_file(const char * filename, wav_data_t * wav) {
    FILE * f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", filename);
        return false;
    }
    
    // Read WAV header (simplified - assumes standard format)
    char riff[4], wave[4], fmt[4];
    fread(riff, 1, 4, f);
    fseek(f, 4, SEEK_CUR); // file size
    fread(wave, 1, 4, f);
    fread(fmt, 1, 4, f);
    
    if (strncmp(riff, "RIFF", 4) != 0 || strncmp(wave, "WAVE", 4) != 0) {
        fprintf(stderr, "Not a valid WAV file\n");
        fclose(f);
        return false;
    }
    
    // Read fmt chunk
    int32_t fmt_size;
    fread(&fmt_size, 4, 1, f);
    
    int16_t audio_format, num_channels;
    int32_t sample_rate, byte_rate;
    int16_t block_align, bits_per_sample;
    
    fread(&audio_format, 2, 1, f);
    fread(&num_channels, 2, 1, f);
    fread(&sample_rate, 4, 1, f);
    fread(&byte_rate, 4, 1, f);
    fread(&block_align, 2, 1, f);
    fread(&bits_per_sample, 2, 1, f);
    
    if (audio_format != 1 && audio_format != 3) {
        fprintf(stderr, "Only PCM and float WAV files supported\n");
        fclose(f);
        return false;
    }
    
    // Skip any extra fmt data
    if (fmt_size > 16) {
        fseek(f, fmt_size - 16, SEEK_CUR);
    }
    
    // Find data chunk
    char chunk_id[4];
    int32_t chunk_size;
    while (fread(chunk_id, 1, 4, f) == 4) {
        fread(&chunk_size, 4, 1, f);
        if (strncmp(chunk_id, "data", 4) == 0) {
            break;
        }
        fseek(f, chunk_size, SEEK_CUR);
    }
    
    // Read audio data
    int num_samples = chunk_size / (num_channels * (bits_per_sample / 8));
    int total_samples = num_samples * num_channels;
    
    wav->sample_rate = sample_rate;
    wav->num_channels = num_channels;
    wav->num_samples = num_samples;
    wav->data = (float *)malloc(total_samples * sizeof(float));
    
    if (!wav->data) {
        fprintf(stderr, "Failed to allocate memory for audio data\n");
        fclose(f);
        return false;
    }
    
    if (audio_format == 3) { // Float format
        fread(wav->data, sizeof(float), total_samples, f);
    } else { // PCM format
        int16_t * pcm_data = (int16_t *)malloc(total_samples * sizeof(int16_t));
        fread(pcm_data, sizeof(int16_t), total_samples, f);
        
        // Convert to float [-1.0, 1.0]
        for (int i = 0; i < total_samples; i++) {
            wav->data[i] = (float)pcm_data[i] / 32768.0f;
        }
        
        free(pcm_data);
    }
    
    fclose(f);
    return true;
}

// Simple WAV file writer
static bool write_wav_file(const char * filename, const float * data, int num_samples, int num_channels, int sample_rate) {
    FILE * f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return false;
    }
    
    int total_samples = num_samples * num_channels;
    int data_size = total_samples * sizeof(float);
    
    // Write RIFF header
    fwrite("RIFF", 1, 4, f);
    int32_t file_size = 36 + data_size;
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    
    // Write fmt chunk
    fwrite("fmt ", 1, 4, f);
    int32_t fmt_size = 16;
    fwrite(&fmt_size, 4, 1, f);
    int16_t audio_format = 3; // Float
    fwrite(&audio_format, 2, 1, f);
    int16_t nc = num_channels;
    fwrite(&nc, 2, 1, f);
    int32_t sr = sample_rate;
    fwrite(&sr, 4, 1, f);
    int32_t byte_rate = sample_rate * num_channels * sizeof(float);
    fwrite(&byte_rate, 4, 1, f);
    int16_t block_align = num_channels * sizeof(float);
    fwrite(&block_align, 2, 1, f);
    int16_t bits_per_sample = 32;
    fwrite(&bits_per_sample, 2, 1, f);
    
    // Write data chunk
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);
    fwrite(data, sizeof(float), total_samples, f);
    
    fclose(f);
    return true;
}

void print_usage(const char * program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -m, --model <path>        Path to GGML model file (required)\n");
    printf("  -i, --input <path>        Input audio file (required)\n");
    printf("  -o, --output <path>       Output directory (required)\n");
    printf("  -t, --threads <n>         Number of threads (default: 4)\n");
    printf("  -h, --help                Show this help message\n");
}

int main(int argc, char ** argv) {
    const char * model_path = NULL;
    const char * input_path = NULL;
    const char * output_dir = NULL;
    int n_threads = 4;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (i + 1 < argc) {
                model_path = argv[++i];
            }
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) {
            if (i + 1 < argc) {
                input_path = argv[++i];
            }
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (i + 1 < argc) {
                output_dir = argv[++i];
            }
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) {
            if (i + 1 < argc) {
                n_threads = atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    if (!model_path || !input_path || !output_dir) {
        fprintf(stderr, "Error: Missing required arguments\n\n");
        print_usage(argv[0]);
        return 1;
    }
    
    printf("Loading model from %s...\n", model_path);
    
    // Load model
    struct mel_band_roformer_model model;
    if (!mel_band_roformer_model_load(model_path, &model)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    
    printf("Model loaded successfully\n");
    printf("  dim: %d\n", model.hparams.dim);
    printf("  depth: %d\n", model.hparams.depth);
    printf("  num_stems: %d\n", model.hparams.num_stems);
    printf("  stereo: %s\n", model.hparams.stereo ? "yes" : "no");
    
    // Read input audio
    printf("Reading input audio from %s...\n", input_path);
    wav_data_t input_wav;
    if (!read_wav_file(input_path, &input_wav)) {
        fprintf(stderr, "Failed to read input audio\n");
        mel_band_roformer_model_free(&model);
        return 1;
    }
    
    printf("Input audio: %d Hz, %d channels, %d samples\n", 
           input_wav.sample_rate, input_wav.num_channels, input_wav.num_samples);
    
    // Allocate output buffers
    float ** output_stems = (float **)malloc(model.hparams.num_stems * sizeof(float *));
    for (int i = 0; i < model.hparams.num_stems; i++) {
        output_stems[i] = (float *)malloc(input_wav.num_samples * input_wav.num_channels * sizeof(float));
    }
    
    int output_length = 0;
    
    // Run inference
    printf("Running inference with %d threads...\n", n_threads);
    if (!mel_band_roformer_eval(&model, n_threads, input_wav.data, 
                                 input_wav.num_samples, input_wav.num_channels,
                                 output_stems, &output_length)) {
        fprintf(stderr, "Inference failed\n");
        
        for (int i = 0; i < model.hparams.num_stems; i++) {
            free(output_stems[i]);
        }
        free(output_stems);
        free(input_wav.data);
        mel_band_roformer_model_free(&model);
        return 1;
    }
    
    // Write output files
    printf("Writing output files to %s...\n", output_dir);
    
    const char * stem_names[] = {"vocals", "drums", "bass", "other"};
    for (int i = 0; i < model.hparams.num_stems; i++) {
        char output_path[1024];
        snprintf(output_path, sizeof(output_path), "%s/%s.wav", output_dir, stem_names[i]);
        
        if (!write_wav_file(output_path, output_stems[i], output_length, 
                           input_wav.num_channels, input_wav.sample_rate)) {
            fprintf(stderr, "Failed to write %s\n", output_path);
        } else {
            printf("  Wrote %s\n", output_path);
        }
    }
    
    // Cleanup
    for (int i = 0; i < model.hparams.num_stems; i++) {
        free(output_stems[i]);
    }
    free(output_stems);
    free(input_wav.data);
    mel_band_roformer_model_free(&model);
    
    printf("Done!\n");
    return 0;
}
