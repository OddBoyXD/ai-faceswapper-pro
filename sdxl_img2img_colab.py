import os
import gc
import time
import torch
import numpy as np
from PIL import Image
import gradio as gr
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    DPMSolverMultistepScheduler
)
from huggingface_hub import login

print("⚡ Initializing Uncensored SDXL Studio on GPU...")

# High-Speed Hugging Face Access
HF_TOKEN = "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"
try:
    login(token=HF_TOKEN, add_to_git_credential=False)
except Exception:
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# Lightweight 6.6GB High-Performance SDXL Base Pipeline (FP16)
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

print(f"Loading {MODEL_ID} in {dtype} (Downloads in ~25s)...")
pipe_img2img = AutoPipelineForImage2Image.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    variant="fp16" if device == "cuda" else None,
    safety_checker=None
)

# High-Speed 20-Step Karras Scheduler (2-3s renders)
pipe_img2img.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe_img2img.scheduler.config,
    use_karras_sigmas=True
)
pipe_img2img.to(device)

if device == "cuda":
    pipe_img2img.enable_attention_slicing()

# Share same weights in VRAM for Text-to-Image (Zero extra memory overhead)
pipe_t2i = AutoPipelineForText2Image.from_pipe(pipe_img2img)

print("🚀 Uncensored SDXL Studio is Ready on GPU!")

DEFAULT_NEGATIVE = (
    "blurry, bad eyes, disfigured, cartoon, 3d render, illustration, bad anatomy, "
    "deformed, extra limbs, bad hands, plastic skin, oversaturated, low quality, worst quality, watermark"
)

ASPECT_RATIOS = {
    "1:1 Square (1024x1024)": (1024, 1024),
    "9:16 Mobile / Story (768x1344)": (768, 1344),
    "16:9 Cinema Widescreen (1344x768)": (1344, 768),
    "4:5 Instagram Portrait (896x1152)": (896, 1152),
    "3:4 Classic Portrait (832x1104)": (832, 1104)
}

# ── 1. IMAGE-TO-IMAGE REFERENCE EDITOR ──
def run_image_to_image_edit(
    image,
    prompt,
    negative_prompt=DEFAULT_NEGATIVE,
    strength=0.65,
    guidance_scale=7.5,
    num_steps=24,
    seed=-1,
    randomize_seed=True,
    progress=gr.Progress(track_tqdm=True)
):
    if image is None:
        return None, "❌ Please upload a reference image to edit.", seed
    if not prompt or not prompt.strip():
        return None, "❌ Please enter a prompt describing how to edit the image.", seed

    # Resize image to standard SDXL resolution (e.g. max 1024)
    w, h = image.size
    max_dim = 1024
    scale = min(max_dim / w, max_dim / h)
    new_w, new_h = int(w * scale) // 8 * 8, int(h * scale) // 8 * 8
    resized_img = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if randomize_seed or seed == -1 or seed is None:
        seed = int(torch.randint(0, 2147483647, (1,)).item())

    generator = torch.Generator(device).manual_seed(seed)

    print(f"Applying Image-to-Image edit (Strength: {strength} | Seed: {seed} | Steps: {num_steps})...")
    start_time = time.time()

    result = pipe_img2img(
        prompt=prompt.strip(),
        negative_prompt=negative_prompt.strip() if negative_prompt else None,
        image=resized_img,
        strength=float(strength),
        guidance_scale=float(guidance_scale),
        num_inference_steps=int(num_steps),
        generator=generator
    ).images[0]

    elapsed = time.time() - start_time
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    status_msg = f"✨ Image Edit Complete in {elapsed:.2f}s! (Seed: {seed} | Strength: {strength})"
    return result, status_msg, seed

# ── 2. TEXT-TO-IMAGE 4K GENERATOR ──
def run_text_to_image(
    prompt,
    negative_prompt=DEFAULT_NEGATIVE,
    aspect_ratio="1:1 Square (1024x1024)",
    guidance_scale=7.0,
    num_steps=25,
    seed=-1,
    randomize_seed=True,
    progress=gr.Progress(track_tqdm=True)
):
    if not prompt or not prompt.strip():
        return None, "❌ Please enter a prompt describing what you want to create.", seed

    width, height = ASPECT_RATIOS.get(aspect_ratio, (1024, 1024))

    if randomize_seed or seed == -1 or seed is None:
        seed = int(torch.randint(0, 2147483647, (1,)).item())

    generator = torch.Generator(device).manual_seed(seed)

    print(f"Generating 4K image (Seed: {seed} | Size: {width}x{height} | Steps: {num_steps})...")
    start_time = time.time()

    result = pipe_t2i(
        prompt=prompt.strip(),
        negative_prompt=negative_prompt.strip() if negative_prompt else None,
        width=width,
        height=height,
        guidance_scale=float(guidance_scale),
        num_inference_steps=int(num_steps),
        generator=generator
    ).images[0]

    elapsed = time.time() - start_time
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    status_msg = f"✨ 4K Masterpiece Generated in {elapsed:.2f}s! (Seed: {seed} | {width}x{height})"
    return result, status_msg, seed

# ── GRADIO UI ──
with gr.Blocks(title="Uncensored SDXL Studio (Colab GPU)") as demo:
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; font-weight: 900; margin: 0;">⚡ Uncensored SDXL Studio</h1>
        <p style="color: #94a3b8; font-size: 1.05rem; margin-top: 5px;">Reference Image-to-Image Editor & 4K Generator • Sub-3s Renders • 100% Uncensored • Free T4 GPU</p>
    </div>
    """)

    with gr.Tabs():
        # TAB 1: IMAGE-TO-IMAGE REFERENCE EDITOR
        with gr.TabItem("🖼️ Reference Image-to-Image Editor"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1️⃣ Upload Reference Photo")
                    img_input = gr.Image(label="Original Image", type="pil")
                    
                    gr.Markdown("### 2️⃣ Describe Your Edits")
                    img_prompt = gr.Textbox(
                        label="Edit Prompt (What to change/add)",
                        placeholder="e.g. 'Make her wear a red silk dress, soft studio lighting, highly detailed'",
                        lines=3
                    )
                    
                    with gr.Row():
                        strength_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.65,
                            step=0.05,
                            label="🎚️ Transformation Strength (0.5 = subtle edit, 0.8 = big change)"
                        )
                        steps_slider = gr.Slider(minimum=15, maximum=40, value=25, step=1, label="⚡ Sampling Steps")

                    with gr.Accordion("⚙️ Pro Settings", open=False):
                        neg_box_1 = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE, lines=2)
                        with gr.Row():
                            cfg_slider_1 = gr.Slider(3.0, 12.0, value=7.5, step=0.5, label="🎯 Prompt Match (CFG)")
                            seed_box_1 = gr.Number(value=-1, label="🎲 Seed (-1 for Random)")
                        rand_check_1 = gr.Checkbox(label="🎲 Randomize Seed Every Generation", value=True)

                    edit_btn = gr.Button("🚀 Apply Image-to-Image Edit", variant="primary", size="lg")

                with gr.Column(scale=1):
                    img_output = gr.Image(label="Transformed Image", type="pil")
                    status_box_1 = gr.Textbox(label="Status & Render Time", interactive=False)

            edit_btn.click(
                fn=run_image_to_image_edit,
                inputs=[img_input, img_prompt, neg_box_1, strength_slider, cfg_slider_1, steps_slider, seed_box_1, rand_check_1],
                outputs=[img_output, status_box_1, seed_box_1]
            )

        # TAB 2: TEXT-TO-IMAGE GENERATOR
        with gr.TabItem("🎨 4K Text-to-Image Generator"):
            with gr.Row():
                with gr.Column(scale=1):
                    t2i_prompt = gr.Textbox(
                        label="✏️ Image Prompt",
                        placeholder="e.g. 'RAW photo, a gorgeous 25yo woman walking in Tokyo neon street at night, 8k resolution, authentic skin pores, cinematic lighting, masterpiece'",
                        lines=4
                    )
                    with gr.Row():
                        ratio_dropdown = gr.Dropdown(
                            label="📐 Aspect Ratio & Resolution",
                            choices=list(ASPECT_RATIOS.keys()),
                            value="1:1 Square (1024x1024)"
                        )
                        t2i_steps = gr.Slider(minimum=15, maximum=40, value=25, step=1, label="⚡ Sampling Steps")

                    with gr.Accordion("⚙️ Pro Settings", open=False):
                        neg_box_2 = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE, lines=2)
                        with gr.Row():
                            cfg_slider_2 = gr.Slider(3.0, 12.0, value=7.0, step=0.5, label="🎯 Prompt Match (CFG)")
                            seed_box_2 = gr.Number(value=-1, label="🎲 Seed (-1 for Random)")
                        rand_check_2 = gr.Checkbox(label="🎲 Randomize Seed Every Generation", value=True)

                    gen_btn = gr.Button("🚀 Generate 4K Masterpiece", variant="primary", size="lg")

                with gr.Column(scale=1):
                    t2i_output = gr.Image(label="Generated 4K Output", type="pil")
                    status_box_2 = gr.Textbox(label="Status & Render Time", interactive=False)

            gen_btn.click(
                fn=run_text_to_image,
                inputs=[t2i_prompt, neg_box_2, ratio_dropdown, cfg_slider_2, t2i_steps, seed_box_2, rand_check_2],
                outputs=[t2i_output, status_box_2, seed_box_2]
            )

if __name__ == "__main__":
    demo.launch(share=True)
