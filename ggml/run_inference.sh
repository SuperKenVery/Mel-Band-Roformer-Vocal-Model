#!/bin/bash
# Example usage script for MelBand RoFormer GGML inference
# This mirrors the functionality of process_song.sh but uses the GGML implementation

set -e

# Configuration (adjust these paths)
CONFIG_PATH="../configs/config_vocals_mel_band_roformer.yaml"
PYTORCH_MODEL_PATH="$HOME/Downloads/MelBandRoformer.ckpt"
QUANTIZATION="q8_0"  # Options: f32, f16, q8_0, q4_0
GGML_MODEL_PATH="./mel_band_roformer-${QUANTIZATION}.ggml"
INPUT_FOLDER="$HOME/Desktop/music-instrumental-extract/inputs"
OUTPUT_DIR="$HOME/Desktop/music-instrumental-extract/outputs"
THREADS=8

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "============================================="
echo "MelBand RoFormer GGML Inference Example"
echo "============================================="
echo -e "${BLUE}Quantization: $QUANTIZATION${NC}"

# Step 1: Check if GGML model exists, if not convert it
if [ ! -f "$GGML_MODEL_PATH" ]; then
    echo -e "${YELLOW}GGML model not found. Converting from PyTorch checkpoint...${NC}"

    if [ ! -f "$PYTORCH_MODEL_PATH" ]; then
        echo "Error: PyTorch checkpoint not found at $PYTORCH_MODEL_PATH"
        echo "Please download the model or update PYTORCH_MODEL_PATH in this script"
        exit 1
    fi

    echo "Converting with $QUANTIZATION quantization..."
    python convert_checkpoint.py \
        --config_path "$CONFIG_PATH" \
        --model_path "$PYTORCH_MODEL_PATH" \
        --output_path "$GGML_MODEL_PATH" \
        --quant "$QUANTIZATION"

    echo -e "${GREEN}Conversion complete!${NC}"
else
    echo -e "${GREEN}Using existing GGML model: $GGML_MODEL_PATH${NC}"
fi

# Step 2: Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 3: Process all WAV files in input folder
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: Input folder not found: $INPUT_FOLDER"
    exit 1
fi

WAV_FILES=("$INPUT_FOLDER"/*.wav)
if [ ! -e "${WAV_FILES[0]}" ]; then
    echo "Error: No WAV files found in $INPUT_FOLDER"
    exit 1
fi

echo ""
echo "Processing ${#WAV_FILES[@]} audio file(s)..."
echo ""

for input_file in "${WAV_FILES[@]}"; do
    filename=$(basename "$input_file" .wav)
    echo -e "${YELLOW}Processing: $filename${NC}"

    # Create temporary output directory for this file
    temp_output="$OUTPUT_DIR/${filename}_separated"
    mkdir -p "$temp_output"

    # Run inference
    ./mel_band_roformer_inference \
        --model "$GGML_MODEL_PATH" \
        --input "$input_file" \
        --output "$temp_output" \
        --threads "$THREADS"

    # Rename output files to match original naming convention
    if [ -f "$temp_output/vocals.wav" ]; then
        mv "$temp_output/vocals.wav" "$OUTPUT_DIR/${filename}_vocals.wav"
    fi

    # Calculate instrumental (if needed)
    # Note: The GGML implementation should handle this, but here's how to do it with sox if needed
    # sox "$input_file" "$temp_output/vocals.wav" "$OUTPUT_DIR/${filename}_instrumental.wav" remix 1,2 1,2 --norm

    # Clean up temp directory
    rm -rf "$temp_output"

    echo -e "${GREEN}Completed: $filename${NC}"
    echo ""
done

echo "============================================="
echo -e "${GREEN}All files processed successfully!${NC}"
echo "Output directory: $OUTPUT_DIR"
echo "============================================="
