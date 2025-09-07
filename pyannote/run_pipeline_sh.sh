#!/bin/bash

#############################################################################
# Unified Pipeline Runner
# Diarization + VAD + ASR with Perfect Timestamp Synchronization
#############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
VENV_NAME="whisperx_env"
DEFAULT_MODEL="large-v3"
DEFAULT_OUTPUT_DIR="./output"

# Functions
print_message() {
    echo -e "${2}${1}${NC}"
}

print_header() {
    echo
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo
}

check_requirements() {
    print_header "Checking Requirements"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_message "✗ Python 3 not found" "$RED"
        exit 1
    fi
    print_message "✓ Python 3 found" "$GREEN"
    
    # Check CUDA
    if command -v nvidia-smi &> /dev/null; then
        print_message "✓ CUDA available (GPU acceleration enabled)" "$GREEN"
        USE_GPU="--device cuda"
    else
        print_message "⚠ CUDA not available (using CPU)" "$YELLOW"
        USE_GPU="--device cpu"
    fi
    
    # Check HF Token
    if [ -z "$HF_TOKEN" ]; then
        print_message "⚠ HF_TOKEN not set (diarization may be limited)" "$YELLOW"
        print_message "  Set with: export HF_TOKEN=your_token" "$NC"
    else
        print_message "✓ HuggingFace token found" "$GREEN"
    fi
}

setup_environment() {
    print_header "Setting Up Environment"
    
    # Create virtual environment if needed
    if [ ! -d "$VENV_NAME" ]; then
        print_message "Creating virtual environment..." "$BLUE"
        python3 -m venv "$VENV_NAME"
    fi
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    print_message "✓ Virtual environment activated" "$GREEN"
    
    # Check if packages are installed
    if ! python -c "import whisperx" &> /dev/null; then
        print_header "Installing Dependencies"
        
        print_message "Upgrading pip..." "$BLUE"
        pip install --upgrade pip
        
        print_message "Installing PyTorch..." "$BLUE"
        if [ "$USE_GPU" = "--device cuda" ]; then
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        else
            pip install torch torchvision torchaudio
        fi
        
        print_message "Installing WhisperX..." "$BLUE"
        pip install git+https://github.com/m-bain/whisperx.git
        
        print_message "Installing additional packages..." "$BLUE"
        pip install pyannote.audio transformers
        
        print_message "✓ Dependencies installed" "$GREEN"
    else
        print_message "✓ Dependencies already installed" "$GREEN"
    fi
}

process_single_file() {
    local audio_file="$1"
    local output_dir="${2:-$DEFAULT_OUTPUT_DIR}"
    
    if [ ! -f "$audio_file" ]; then
        print_message "✗ File not found: $audio_file" "$RED"
        return 1
    fi
    
    print_header "Processing: $(basename "$audio_file")"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Run the pipeline
    print_message "Running unified pipeline..." "$BLUE"
    
    python whisperx_unified.py \
        "$audio_file" \
        --output-dir "$output_dir" \
        --model "$DEFAULT_MODEL" \
        $USE_GPU \
        --min-speakers ${MIN_SPEAKERS:-2} \
        --max-speakers ${MAX_SPEAKERS:-10}
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_message "✓ Processing completed successfully!" "$GREEN"
        
        # Show output location
        local base_name=$(basename "$audio_file" | sed 's/\.[^.]*$//')
        print_message "Output files:" "$BLUE"
        print_message "  - $output_dir/${base_name}_output.jsonl" "$NC"
        print_message "  - $output_dir/${base_name}_transcript.txt" "$NC"
    else
        print_message "✗ Processing failed" "$RED"
        return $exit_code
    fi
}

process_batch() {
    local pattern="$1"
    local output_dir="${2:-$DEFAULT_OUTPUT_DIR}"
    
    print_header "Batch Processing"
    
    # Find all matching files
    shopt -s nullglob
    files=($pattern)
    
    if [ ${#files[@]} -eq 0 ]; then
        print_message "✗ No files found matching: $pattern" "$RED"
        exit 1
    fi
    
    print_message "Found ${#files[@]} files to process" "$BLUE"
    
    # Process each file
    local success=0
    local failed=0
    
    for file in "${files[@]}"; do
        if process_single_file "$file" "$output_dir"; then
            ((success++))
        else
            ((failed++))
        fi
        echo
    done
    
    # Summary
    print_header "Batch Processing Complete"
    print_message "Successful: $success files" "$GREEN"
    if [ $failed -gt 0 ]; then
        print_message "Failed: $failed files" "$RED"
    fi
}

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] <audio_file>

Unified Pipeline for Diarization + VAD + ASR

OPTIONS:
    -h, --help          Show this help message
    -o, --output DIR    Output directory (default: ./output)
    -b, --batch         Process multiple files (use with pattern)
    --min-speakers N    Minimum number of speakers (default: 2)
    --max-speakers N    Maximum number of speakers (default: 10)
    --model MODEL       Whisper model (default: large-v3)

EXAMPLES:
    # Single file
    $0 audio.wav
    
    # Single file with custom output
    $0 -o ./results audio.mp3
    
    # Batch processing
    $0 -b "*.wav"
    
    # With speaker limits
    MIN_SPEAKERS=2 MAX_SPEAKERS=4 $0 audio.wav

ENVIRONMENT:
    HF_TOKEN    HuggingFace token for better diarization

EOF
}

# Main execution
main() {
    # Parse arguments
    BATCH_MODE=false
    OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
    AUDIO_FILE=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -b|--batch)
                BATCH_MODE=true
                shift
                ;;
            --min-speakers)
                MIN_SPEAKERS="$2"
                shift 2
                ;;
            --max-speakers)
                MAX_SPEAKERS="$2"
                shift 2
                ;;
            --model)
                DEFAULT_MODEL="$2"
                shift 2
                ;;
            *)
                AUDIO_FILE="$1"
                shift
                ;;
        esac
    done
    
    # Check if audio file is provided
    if [ -z "$AUDIO_FILE" ]; then
        print_message "✗ No audio file specified" "$RED"
        show_usage
        exit 1
    fi
    
    # Setup
    check_requirements
    setup_environment
    
    # Process
    if [ "$BATCH_MODE" = true ]; then
        process_batch "$AUDIO_FILE" "$OUTPUT_DIR"
    else
        process_single_file "$AUDIO_FILE" "$OUTPUT_DIR"
    fi
    
    # Cleanup
    deactivate 2>/dev/null || true
    
    print_header "Pipeline Complete"
    print_message "✓ All tasks completed successfully!" "$GREEN"
}

# Run main
main "$@"