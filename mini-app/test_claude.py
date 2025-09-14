# test_claude.py
import sys
from pathlib import Path

# Mini-app dizinini Python path'ine ekle
sys.path.append(str(Path(__file__).resolve().parent))

from models.tts.xtts import XTTSEngine

import torch

print("CUDA?", torch.cuda.is_available())
eng = XTTSEngine(model_name="tts_models/multilingual/multi-dataset/xtts_v2", language="tr")
print("Model hazır.")

