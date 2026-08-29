import os
import gc
import time
import torch
import numpy as np
from PIL import Image
import gradio as gr
from diffusers import FluxPipeline

print("⚡ Initializing Black Forest Labs FLUX Engine on GPU...")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

# Load Black Forest Labs FLUX.1 [schnell] for ultra-fast high-fidelity 4-step generation on Colab GPU
MODEL_ID = "black-forest-labs/FLUX.1-schnell"

print(f"Loading {MODEL_ID} in {dtype}...")
pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    safety_checker=None
)

# Enable memory optimizations for Google Colab GPU (Fits within 16GB VRAM smoothly)
if device == "cuda":
    pipe.enable_model_cpu_offload()
    print("✅ Model CPU Offload enabled (VRAM Optimized)")
else:
    pipe.to(device)

print("🚀 Black Forest Labs FLUX.1 Engine is Ready!")

ASPECT_RATIOS = {
    "1:1 Square (1024x1024)": (1024, 1024),
    "9:16 Mobile / Story (768x1344)": (768, 1344),
    "16:9 Cinema Widescreen (1344x768)": (1344, 768),
    "4:5 Instagram Portrait (896x1152)": (896, 1152),
    "3:4 Classic Portrait (832x1104)": (832, 1104),
    "2:3 Poster (768x1152)": (768, 1152),
    "3:2 Landscape (1152x768)": (1152, 768)
}

def generate_flux_image(
    prompt,
    aspect_ratio="1:1 Square (1024x1024)",
    num_steps=4,
    guidance_scale=0.0,
    seed=-1,
    randomize_seed=True,
    progress=gr.Progress(track_tqdm=True)
):
    if not prompt or not prompt.strip():
        return None, "❌ Please enter a prompt describing the image you want to generate.", seed

    width, height = ASPECT_RATIOS.get(aspect_ratio, (1024, 1024))
    
    if randomize_seed or seed == -1 or seed is None:
        seed = int(torch.randint(0, 2147483647, (1,)).item())

    generator = torch.Generator("cpu").manual_seed(seed)

    print(f"Generating FLUX image (Seed: {seed} | Resolution: {width}x{height} | Steps: {num_steps})...")
    start_time = time.time()
    
    image = pipe(
        prompt=prompt.strip(),
        width=width,
        height=height,
        num_inference_steps=int(num_steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
        max_sequence_length=256
    ).images[0]

    elapsed = time.time() - start_time
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    status_msg = f"✨ Generated in {elapsed:.2f}s! (Seed: {seed} | Size: {width}x{height} | Steps: {num_steps})"
    return image, status_msg, seed

# ── GRADIO UI ──
with gr.Blocks(title="FLUX.1 Image Generator (Black Forest Labs)") as demo:
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="background: linear-gradient(90deg, #ff7e5f, #feb47b, #ff2a6d); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; font-weight: 900; margin: 0;">⚡ FLUX.1 • Black Forest Labs</h1>
        <p style="color: #94a3b8; font-size: 1.05rem; margin-top: 5px;">State-of-the-Art 12B Flow Transformer • Unmatched Photorealism & Typography • 100% Uncensored • Free GPU</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            prompt_box = gr.Textbox(
                label="✏️ Image Prompt (Describe your vision)",
                placeholder="e.g. A photorealistic cinematic close-up portrait of a woman in Tokyo neon rain, 8k resolution, authentic skin pores, volumetric lighting, masterpiece, camera lens 85mm f/1.4",
                lines=4
            )
            
            with gr.Row():
                ratio_dropdown = gr.Dropdown(
                    label="📐 Aspect Ratio & Resolution",
                    choices=list(ASPECT_RATIOS.keys()),
                    value="1:1 Square (1024x1024)"
                )
                steps_slider = gr.Slider(
                    minimum=1,
                    maximum=12,
                    value=4,
                    step=1,
                    label="⚡ Inference Steps (4 is optimal for schnell)"
                )
            
            with gr.Accordion("⚙️ Advanced Generation Settings", open=False):
                with gr.Row():
                    guidance_slider = gr.Slider(
                        minimum=0.0,
                        maximum=10.0,
                        value=0.0,
                        step=0.5,
                        label="🎯 Guidance Scale (0.0 recommended for schnell)"
                    )
                    seed_input = gr.Number(value=-1, label="🎲 Seed (-1 for Random)")
                randomize_check = gr.Checkbox(label="🎲 Randomize Seed Every Generation", value=True)

            gen_btn = gr.Button("🚀 Generate with FLUX.1", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated FLUX.1 Masterpiece", type="pil")
            status_box = gr.Textbox(label="Status & Render Time", interactive=False)

    gen_btn.click(
        fn=generate_flux_image,
        inputs=[prompt_box, ratio_dropdown, steps_slider, guidance_slider, seed_input, randomize_check],
        outputs=[output_image, status_box, seed_input]
    )

if __name__ == "__main__":
    demo.launch(share=True)
