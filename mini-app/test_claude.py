# test_claude.py
import sys, os, glob, importlib.util

# 1) Proje kökünü belirleyelim (bu dosyanın bulunduğu klasör)
ROOT = os.path.dirname(os.path.abspath(__file__))

# 2) mini-app altında xtts*.py'yi ara (XTTS.py, xtts.py vb.)
candidates = glob.glob(os.path.join(ROOT, "xtts.py")) + \
             glob.glob(os.path.join(ROOT, "XTTS.py")) + \
             glob.glob(os.path.join(ROOT, "xtts", "__init__.py"))

if not candidates:
    # alt klasörlerde de ara
    candidates = glob.glob(os.path.join(ROOT, "**", "xtts.py"), recursive=True) + \
                 glob.glob(os.path.join(ROOT, "**", "XTTS.py"), recursive=True)

if not candidates:
    raise FileNotFoundError("xtts.py bulunamadı. Dosya adını ve konumunu kontrol et.")

xtts_path = candidates[0]
spec = importlib.util.spec_from_file_location("xtts", xtts_path)
xtts = importlib.util.module_from_spec(spec)
sys.modules["xtts"] = xtts
spec.loader.exec_module(xtts)

import xtts

import torch
print("CUDA?", torch.cuda.is_available())
eng = xtts.XTTSEngine(model_name="tts_models/multilingual/multi-dataset/xtts_v2", language="tr")
print("Model hazır.")
