#!/bin/bash
# Build script for MelBand RoFormer GGML implementation

set -e

echo "==================================="
echo "MelBand RoFormer GGML Build Script"
echo "==================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="Release"
USE_CMAKE=true
GGML_DIR=""
CLEAN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --use-make)
            USE_CMAKE=false
            shift
            ;;
        --ggml-dir)
            GGML_DIR="$2"
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --debug          Build in debug mode (default: Release)"
            echo "  --use-make       Use Makefile instead of CMake"
            echo "  --ggml-dir DIR   Specify GGML directory"
            echo "  --clean          Clean before building"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check for GGML
if [ -z "$GGML_DIR" ]; then
    echo -e "${YELLOW}Checking for GGML...${NC}"

    # Try to find GGML
    if pkg-config --exists ggml; then
        echo -e "${GREEN}Found GGML via pkg-config${NC}"
    elif [ -d "/usr/local/include/ggml" ]; then
        echo -e "${GREEN}Found GGML in /usr/local${NC}"
        GGML_DIR="/usr/local"
    elif [ -d "./ggml" ]; then
        echo -e "${GREEN}Found local GGML directory${NC}"
        GGML_DIR="./ggml"
    else
        echo -e "${RED}GGML not found!${NC}"
        echo "Please install GGML or specify its location with --ggml-dir"
        echo ""
        echo "To install GGML:"
        echo "  git clone https://github.com/ggerganov/ggml.git"
        echo "  cd ggml"
        echo "  mkdir build && cd build"
        echo "  cmake .."
        echo "  make -j\$(nproc)"
        echo "  sudo make install"
        exit 1
    fi
fi

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}Cleaning...${NC}"
    if [ "$USE_CMAKE" = true ]; then
        rm -rf build
    else
        make clean 2>/dev/null || true
    fi
fi

# Build
if [ "$USE_CMAKE" = true ]; then
    echo -e "${YELLOW}Building with CMake ($BUILD_TYPE)...${NC}"

    mkdir -p build
    cd build

    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    if [ -n "$GGML_DIR" ]; then
        CMAKE_ARGS="$CMAKE_ARGS -DGGML_DIR=$GGML_DIR"
    fi

    cmake .. $CMAKE_ARGS
    make -j$(nproc)

    echo ""
    echo -e "${GREEN}Build successful!${NC}"
    echo "Executable: $(pwd)/mel_band_roformer_inference"

else
    echo -e "${YELLOW}Building with Make...${NC}"

    MAKE_ARGS=""
    if [ -n "$GGML_DIR" ]; then
        MAKE_ARGS="GGML_DIR=$GGML_DIR"
    fi

    make -j$(nproc) $MAKE_ARGS

    echo ""
    echo -e "${GREEN}Build successful!${NC}"
    echo "Executable: $(pwd)/mel_band_roformer_inference"
fi

echo ""
echo "Next steps:"
echo "1. Convert your PyTorch checkpoint:"
echo "   python convert_checkpoint.py \\"
echo "     --config_path ../configs/config_vocals_mel_band_roformer.yaml \\"
echo "     --model_path /path/to/checkpoint.ckpt \\"
echo "     --output_path model.ggml"
echo ""
echo "2. Run inference:"
echo "   ./mel_band_roformer_inference \\"
echo "     -m model.ggml \\"
echo "     -i input.wav \\"
echo "     -o output_dir"
