import os
import gc
import io
import time
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from PIL import Image
import gradio as gr
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, DPMSolverMultistepScheduler

# Environment & Device Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

MODEL_ID = "SG161222/RealVisXL_V4.0"

print(f"🚀 Initializing AI Unrestricted Studio on: {device}")

# Load pipelines cleanly
pipe_t2i = AutoPipelineForText2Image.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    variant="fp16" if device == "cuda" else None,
    use_safetensors=True
)
pipe_t2i.scheduler = DPMSolverMultistepScheduler.from_config(pipe_t2i.scheduler.config, use_karras_sigmas=True)
pipe_t2i.safety_checker = None

if device == "cuda":
    pipe_t2i.enable_model_cpu_offload()

pipe_i2i = AutoPipelineForImage2Image.from_pipe(pipe_t2i)
pipe_i2i.safety_checker = None
print("✅ Unrestricted AI Engines 100% Ready!")

RATIOS = {
    "📱 9:16 (Story / Reel / Phone)": (720, 1280),
    "📸 4:5 (Instagram Portrait)": (832, 1040),
    "🖼️ 1:1 (Square)": (1024, 1024),
    "🖥️ 16:9 (Landscape / YouTube)": (1280, 720),
    "🎬 21:9 (Cinematic Ultrawide)": (1344, 576)
}

def generate_text_to_image(prompt, negative_prompt, ratio_name, steps, cfg, seed):
    if not prompt or not prompt.strip():
        return None, "❌ Please enter a prompt."
        
    width, height = RATIOS.get(ratio_name, (1024, 1024))
    
    generator = None
    if seed and int(seed) != -1:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        
    t0 = time.time()
    try:
        image = pipe_t2i(
            prompt=prompt.strip(),
            negative_prompt=negative_prompt.strip() if negative_prompt else "deformed, blurry, bad anatomy, bad eyes, disfigured, low quality",
            width=width,
            height=height,
            num_inference_steps=int(steps),
            guidance_scale=float(cfg),
            generator=generator
        ).images[0]
        
        duration = time.time() - t0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return image, f"✅ Rendered in {duration:.2f}s ({width}x{height}) • 100% Unrestricted"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def edit_image_to_image(init_image, instruction, strength, steps, cfg, seed):
    if init_image is None:
        return None, "❌ Please upload a reference image to edit."
    if not instruction or not instruction.strip():
        return None, "❌ Please type what to change (e.g. 'wearing a black hat', 'sunglasses', 'curly hair')."
        
    if isinstance(init_image, np.ndarray):
        init_image = Image.fromarray(init_image).convert("RGB")
    else:
        init_image = init_image.convert("RGB")
        
    w, h = init_image.size
    scale = min(1024 / max(w, h), 1.0)
    new_w, new_h = int((w * scale) // 8) * 8, int((h * scale) // 8) * 8
    init_image = init_image.resize((new_w, new_h), Image.LANCZOS)
    
    generator = None
    if seed and int(seed) != -1:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        
    t0 = time.time()
    try:
        full_prompt = f"photorealistic 8k, raw photo, {instruction.strip()}, highly detailed, natural lighting"
        neg_prompt = "cartoon, bad anatomy, blurry, artifacts, lowres, deformed"
        
        result = pipe_i2i(
            prompt=full_prompt,
            negative_prompt=neg_prompt,
            image=init_image,
            strength=float(strength),
            num_inference_steps=int(steps),
            guidance_scale=float(cfg),
            generator=generator
        ).images[0]
        
        duration = time.time() - t0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return result, f"✅ Edited in {duration:.2f}s! ({new_w}x{new_h})"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

with gr.Blocks(title="⚡ AI Unrestricted Studio • Pro Edition") as demo:
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 20px; padding: 10px 0;">
        <h1 style="background: linear-gradient(90deg, #ff007f, #8a2be2, #00e5ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.3rem; font-weight: 900; margin: 0;">⚡ AI UNRESTRICTED PRO STUDIO</h1>
        <p style="color: #94a3b8; font-size: 1rem; margin-top: 5px;">High-Accuracy 4K Text-to-Image & Smart Instruction Photo Editor • 100% Uncensored</p>
    </div>
    """)
    
    with gr.Tabs():
        # TAB 1: SMART INSTRUCTION IMAGE EDITOR
        with gr.TabItem("🪄 Smart Instruction Photo Editor"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1️⃣ Upload Reference Photo")
                    i_img = gr.Image(label="Reference Image", type="pil", height=320)
                    
                    gr.Markdown("### 2️⃣ Exact Modification Prompt")
                    i_prompt = gr.Textbox(
                        label="Instruction / Transformation Prompt",
                        placeholder="Type exactly what to change (e.g. 'wearing a black fedora hat', 'add dark aviator sunglasses', 'change hairstyle to textured fade', 'standing in Tokyo neon street')",
                        lines=3
                    )
                    
                    with gr.Accordion("⚙️ Precision Tuning Controls", open=True):
                        i_strength = gr.Slider(0.15, 0.85, value=0.45, step=0.05, label="🎯 Edit Intensity (Lower = preserves original photo more, Higher = stronger changes)")
                        i_steps = gr.Slider(15, 45, value=28, step=1, label="Sampling Steps")
                        i_cfg = gr.Slider(4.0, 12.0, value=7.5, step=0.5, label="Prompt Adherence (CFG)")
                        i_seed = gr.Textbox(value="-1", label="Seed (-1 = Random)")
                        
                    i_btn = gr.Button("🪄 Render AI Photo Edit", variant="primary")
                    
                with gr.Column(scale=1):
                    i_out = gr.Image(label="Edited Output", type="pil")
                    i_status = gr.Textbox(label="Process Status", interactive=False)
                    
            i_btn.click(
                fn=edit_image_to_image,
                inputs=[i_img, i_prompt, i_strength, i_steps, i_cfg, i_seed],
                outputs=[i_out, i_status]
            )
            
        # TAB 2: UNRESTRICTED TEXT-TO-IMAGE
        with gr.TabItem("🎨 4K Unrestricted Text-to-Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1️⃣ Detailed Text Prompt")
                    t_prompt = gr.Textbox(
                        label="Prompt (100% Uncensored • Any Theme / Subject)",
                        placeholder="e.g. '8k raw photo of a cyberpunk mercenary standing in rain, neon reflections, cinematic lighting, 85mm lens, masterpiece'",
                        lines=3
                    )
                    t_neg = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="deformed, blurry, bad anatomy, bad eyes, disfigured, low quality",
                        lines=1
                    )
                    
                    t_ratio = gr.Dropdown(
                        label="📐 Aspect Ratio",
                        choices=list(RATIOS.keys()),
                        value="📱 9:16 (Story / Reel / Phone)"
                    )
                    
                    with gr.Accordion("⚙️ Generation Settings", open=True):
                        t_steps = gr.Slider(20, 50, value=30, step=1, label="Sampling Steps")
                        t_cfg = gr.Slider(4.0, 12.0, value=7.0, step=0.5, label="Prompt Guidance (CFG)")
                        t_seed = gr.Textbox(value="-1", label="Seed (-1 = Random)")
                        
                    t_btn = gr.Button("⚡ Generate 4K Image", variant="primary")
                    
                with gr.Column(scale=1):
                    t_out = gr.Image(label="Generated 4K Output", type="pil")
                    t_status = gr.Textbox(label="Status", interactive=False)
                    
            t_btn.click(
                fn=generate_text_to_image,
                inputs=[t_prompt, t_neg, t_ratio, t_steps, t_cfg, t_seed],
                outputs=[t_out, t_status]
            )

if __name__ == "__main__":
    demo.launch(share=True)
