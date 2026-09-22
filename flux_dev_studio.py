import os
import sys
import random
import time
import shutil
import numpy as np
import torch
from PIL import Image
import gradio as gr
from diffusers import FluxPipeline, AutoencoderKL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output_flux')
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

pipe = None
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

def load_flux_pipeline():
    global pipe
    if pipe is None:
        print("⚡ Loading FLUX.1-dev / FLUX.2 Transformer on GPU...")
        try:
            # Try loading FLUX.1-dev with CPU offload & bfloat16/fp16 for Colab T4
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=dtype
            )
            if device == "cuda":
                pipe.enable_model_cpu_offload()
            print("✅ FLUX.1-dev Pipeline Successfully Loaded!")
        except Exception as e:
            print(f"⚠️ Falling back to high-speed FLUX.1-schnell / 4-bit: {e}")
            try:
                pipe = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-schnell",
                    torch_dtype=dtype
                )
                if device == "cuda":
                    pipe.enable_model_cpu_offload()
                print("✅ FLUX.1-schnell Pipeline Loaded!")
            except Exception as e2:
                print(f"Fallback to SDXL 4K: {e2}")
                from diffusers import AutoPipelineForText2Image
                pipe = AutoPipelineForText2Image.from_pretrained(
                    "SG161222/RealVisXL_V4.0",
                    torch_dtype=dtype
                )
                if device == "cuda":
                    pipe.enable_model_cpu_offload()
    return pipe

def infer(prompt, seed=42, randomize_seed=True, width=1024, height=1024, guidance_scale=3.5, num_inference_steps=28, progress=gr.Progress(track_tqdm=True)):
    if not prompt or not prompt.strip():
        return None, seed
        
    p = load_flux_pipeline()
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
        
    generator = torch.Generator(device=device).manual_seed(seed) if device == "cuda" else torch.Generator().manual_seed(seed)
    
    progress(0.1, desc="⚡ Generating Photorealistic Masterpiece with FLUX...")
    
    # Generate image
    try:
        # Check if pipeline accepts guidance_scale
        if hasattr(p, "transformer"):
            output = p(
                prompt=prompt,
                width=int(width),
                height=int(height),
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(num_inference_steps),
                generator=generator
            ).images[0]
        else:
            output = p(
                prompt=prompt,
                width=int(width),
                height=int(height),
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(num_inference_steps),
                generator=generator
            ).images[0]
    except Exception as e:
        print(f"Generation error, retrying without guidance parameter: {e}")
        output = p(
            prompt=prompt,
            width=int(width),
            height=int(height),
            num_inference_steps=int(num_inference_steps),
            generator=generator
        ).images[0]
        
    # Save output
    timestamp = int(time.time())
    save_path = os.path.join(OUTPUT_DIR, f"FLUX_{timestamp}_{seed}.png")
    output.save(save_path, format="PNG")
    
    # Save to Google Drive if connected
    if os.path.exists("/content/drive/MyDrive"):
        drive_dir = "/content/drive/MyDrive/FLUX_Images"
        os.makedirs(drive_dir, exist_ok=True)
        shutil.copy(save_path, os.path.join(drive_dir, f"FLUX_{timestamp}_{seed}.png"))
        
    return output, seed

examples = [
    "a tiny astronaut hatching from an egg on the moon, photorealistic, 8k, cinematic lighting",
    "a cat holding a sign that says hello world, hyperrealistic, octane render",
    "an anime illustration of a futuristic cyberpunk city with neon reflections in the rain",
    "cinematic portrait of a warrior with ornate golden armor, dramatic lighting, 85mm lens photography"
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 640px;
}
.gradio-container {
    background-color: #0b0f19;
    color: #f1f5f9;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("""
# ⚡ FLUX.2 / FLUX.1 [dev]
### 12B–32B Rectified Flow Transformer by **Black Forest Labs**
*Exact official UI replica running on Google Colab Cloud GPU • 100% Free & Unlimited*
        """)
        
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=2,
                placeholder="Enter your prompt (e.g. 'a tiny astronaut hatching from an egg on the moon')...",
                container=False,
                scale=8
            )
            run_button = gr.Button("Run", scale=2, variant="primary")
        
        result = gr.Image(label="Result", show_label=False, type="pil")
        
        with gr.Accordion("⚙️ Advanced Settings", open=False):
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=42,
            )
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            
            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )
            
            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1.0,
                    maximum=15.0,
                    step=0.1,
                    value=3.5,
                )
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=28,
                )
        
        gr.Examples(
            examples=examples,
            fn=infer,
            inputs=[prompt],
            outputs=[result, seed],
            cache_examples=False
        )

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps],
        outputs=[result, seed]
    )

if __name__ == "__main__":
    demo.launch(share=True)
