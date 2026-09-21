import os
import torch
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("📥 Pre-Downloading RealVisXL v4.0 4K AI Engine (6.94 GB)...")
print("=" * 60)

from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

MODEL_ID = "SG161222/RealVisXL_V4.0"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

pipe = AutoPipelineForText2Image.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    variant="fp16" if device == "cuda" else None,
    use_safetensors=True
)
pipe.safety_checker = None

print("=" * 60)
print("✅ RealVisXL v4.0 AI Model is 100% Pre-Downloaded & Cached!")
print("=" * 60)
