#!/usr/bin/env python3
"""
WhisperX Pipeline for Speaker Diarization and Transcription
"""

import argparse
import json
import os
import sys
import torch
import whisperx
import gc
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WhisperXPipeline:
    def __init__(self, 
                 model_size="large-v3",
                 device=None,
                 compute_type=None,
                 hf_token=None,
                 language=None,
                 batch_size=16):
        """
        Initialize WhisperX Pipeline
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)
            device: cuda or cpu (auto-detect if None)
            compute_type: float16 for GPU, int8 for CPU (auto-detect if None)
            hf_token: HuggingFace token for speaker diarization
            language: Force specific language (auto-detect if None)
            batch_size: Batch size for processing
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type or ("float16" if self.device == "cuda" else "int8")
        self.model_size = model_size
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.language = language
        self.batch_size = batch_size
        
        if not self.hf_token:
            logger.warning("No HuggingFace token provided. Speaker diarization will be skipped.")
        
        logger.info(f"Initializing WhisperX Pipeline:")
        logger.info(f"  - Device: {self.device}")
        logger.info(f"  - Model: {self.model_size}")
        logger.info(f"  - Compute Type: {self.compute_type}")
        logger.info(f"  - Batch Size: {self.batch_size}")
        
        # Load Whisper model
        self.model = None
        self.diarize_model = None
        
    def load_models(self):
        """Load Whisper and diarization models"""
        try:
            logger.info("Loading Whisper model...")
            self.model = whisperx.load_model(
                self.model_size, 
                self.device, 
                compute_type=self.compute_type,
                language=self.language
            )
            
            if self.hf_token:
                logger.info("Loading diarization model...")
                self.diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=self.hf_token, 
                    device=self.device
                )
            
            logger.info("Models loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def process_audio(self, audio_path, output_dir=None, min_speakers=None, max_speakers=None):
        """
        Process audio file with transcription and diarization
        
        Args:
            audio_path: Path to audio file
            output_dir: Directory for output files (default: same as audio)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return None
        
        # Setup output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = audio_path.parent
        
        base_name = audio_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Load audio
            logger.info(f"Loading audio: {audio_path}")
            audio = whisperx.load_audio(str(audio_path))
            
            # 1. Transcribe with Whisper
            logger.info("Starting transcription...")
            result = self.model.transcribe(
                audio, 
                batch_size=self.batch_size,
                language=self.language
            )
            
            detected_language = result.get("language", "unknown")
            logger.info(f"Detected language: {detected_language}")
            
            # Save initial transcription
            transcription_file = output_dir / f"{base_name}_transcription_{timestamp}.json"
            with open(transcription_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved transcription: {transcription_file}")
            
            # 2. Align timestamps
            logger.info("Aligning timestamps...")
            model_a, metadata = whisperx.load_align_model(
                language_code=detected_language, 
                device=self.device
            )
            result = whisperx.align(
                result["segments"], 
                model_a, 
                metadata, 
                audio, 
                self.device, 
                return_char_alignments=False
            )
            
            # Save aligned transcription
            aligned_file = output_dir / f"{base_name}_aligned_{timestamp}.json"
            with open(aligned_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved aligned transcription: {aligned_file}")
            
            # 3. Speaker Diarization
            if self.diarize_model:
                logger.info("Starting speaker diarization...")
                diarize_params = {}
                if min_speakers:
                    diarize_params["min_speakers"] = min_speakers
                if max_speakers:
                    diarize_params["max_speakers"] = max_speakers
                
                diarize_segments = self.diarize_model(audio, **diarize_params)
                
                # Save diarization
                diarization_file = output_dir / f"{base_name}_diarization_{timestamp}.json"
                with open(diarization_file, 'w', encoding='utf-8') as f:
                    json.dump(diarize_segments, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved diarization: {diarization_file}")
                
                # 4. Assign speakers to words
                logger.info("Assigning speakers to segments...")
                result = whisperx.assign_word_speakers(diarize_segments, result)
            else:
                logger.warning("Skipping diarization (no HF token)")
            
            # Save final result
            final_file = output_dir / f"{base_name}_final_{timestamp}.json"
            with open(final_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved final result: {final_file}")
            
            # Generate readable output
            self.generate_readable_output(result, output_dir / f"{base_name}_transcript_{timestamp}.txt")
            
            # Generate JSONL format
            self.generate_jsonl_output(result, output_dir / f"{base_name}_output_{timestamp}.jsonl")
            
            # Clean up memory
            del model_a
            gc.collect()
            torch.cuda.empty_cache()
            
            logger.info("Processing completed successfully!")
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_readable_output(self, result, output_path):
        """Generate human-readable transcript"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("TRANSCRIPT\n")
                f.write("=" * 80 + "\n\n")
                
                current_speaker = None
                
                for segment in result.get("segments", []):
                    speaker = segment.get("speaker", "UNKNOWN")
                    text = segment.get("text", "").strip()
                    start = segment.get("start", 0)
                    end = segment.get("end", 0)
                    
                    if speaker != current_speaker:
                        f.write(f"\n[{speaker}]\n")
                        current_speaker = speaker
                    
                    f.write(f"[{start:.2f} - {end:.2f}] {text}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            logger.info(f"Saved readable transcript: {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating readable output: {e}")
    
    def generate_jsonl_output(self, result, output_path):
        """Generate JSONL format output"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for segment in result.get("segments", []):
                    line = {
                        "text": segment.get("text", "").strip(),
                        "start": segment.get("start", 0),
                        "end": segment.get("end", 0),
                        "speaker": segment.get("speaker", "UNKNOWN")
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
            
            logger.info(f"Saved JSONL output: {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating JSONL output: {e}")
    
    def cleanup(self):
        """Clean up models and memory"""
        if self.model:
            del self.model
        if self.diarize_model:
            del self.diarize_model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Cleanup completed")

def main():
    parser = argparse.ArgumentParser(description="WhisperX Pipeline for Speech Processing")
    
    # Input/Output arguments
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("-o", "--output-dir", help="Output directory (default: same as audio)")
    
    # Model arguments
    parser.add_argument("-m", "--model", default="large-v3", 
                       choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                       help="Whisper model size (default: large-v3)")
    parser.add_argument("-l", "--language", help="Force specific language (auto-detect if not specified)")
    
    # Processing arguments
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--min-speakers", type=int, help="Minimum number of speakers")
    parser.add_argument("--max-speakers", type=int, help="Maximum number of speakers")
    
    # Device arguments
    parser.add_argument("-d", "--device", choices=["cuda", "cpu"], help="Device to use (auto-detect if not specified)")
    parser.add_argument("-c", "--compute-type", choices=["float16", "int8", "float32"], help="Compute type (auto-detect if not specified)")
    
    # Authentication
    parser.add_argument("--hf-token", help="HuggingFace token for diarization (or set HF_TOKEN env variable)")
    
    # Other
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize pipeline
    pipeline = WhisperXPipeline(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        hf_token=args.hf_token,
        language=args.language,
        batch_size=args.batch_size
    )
    
    # Load models
    if not pipeline.load_models():
        logger.error("Failed to load models")
        sys.exit(1)
    
    # Process audio
    result = pipeline.process_audio(
        args.audio_file,
        output_dir=args.output_dir,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers
    )
    
    # Cleanup
    pipeline.cleanup()
    
    if result:
        logger.info("✓ Pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("✗ Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()