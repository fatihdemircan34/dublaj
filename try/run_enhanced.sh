#!/bin/bash

# Enhanced ASR+Diarization çalıştırma scripti

# Virtual environment'ı aktif et
source ../venv/bin/activate

# Default values
AUDIO_FILE=""
OUTPUT_DIR="../output"
MODEL_SIZE="large-v2"
LANGUAGE=""
MIN_SPEAKERS=""
MAX_SPEAKERS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            AUDIO_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_SIZE="$2"
            shift 2
            ;;
        -l|--language)
            LANGUAGE="$2"
            shift 2
            ;;
        --min-speakers)
            MIN_SPEAKERS="$2"
            shift 2
            ;;
        --max-speakers)
            MAX_SPEAKERS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Enhanced ASR+Diarization Script"
            echo "Usage: $0 -i audio_file [options]"
            echo ""
            echo "Options:"
            echo "  -i, --input       Input audio file (required)"
            echo "  -o, --output      Output directory (default: ../output)"
            echo "  -m, --model       Model size (default: large-v2)"
            echo "  -l, --language    Language code (default: auto-detect)"
            echo "      --min-speakers Minimum number of speakers"
            echo "      --max-speakers Maximum number of speakers"
            echo "  -h, --help        Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$AUDIO_FILE" ]; then
    echo "Error: Input audio file is required"
    echo "Use -h for help"
    exit 1
fi

# Check file exists
if [ ! -f "$AUDIO_FILE" ]; then
    echo "Error: Audio file not found: $AUDIO_FILE"
    exit 1
fi

# Build command
CMD="python ../src/enhanced_asr_diarization.py \"$AUDIO_FILE\" --output \"$OUTPUT_DIR\" --model-size \"$MODEL_SIZE\""

if [ -n "$LANGUAGE" ]; then
    CMD="$CMD --language \"$LANGUAGE\""
fi

if [ -n "$MIN_SPEAKERS" ]; then
    CMD="$CMD --min-speakers $MIN_SPEAKERS"
fi

if [ -n "$MAX_SPEAKERS" ]; then
    CMD="$CMD --max-speakers $MAX_SPEAKERS"
fi

echo "Running: $CMD"
eval $CMD
