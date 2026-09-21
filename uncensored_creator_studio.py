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

with gr.Blocks(title="⚡ AI Unrestricted Studio • Mobile & 4K Edition") as demo:
    gr.HTML("""
    <style>
    @media (max-width: 768px) {
        .gradio-container { padding: 4px !important; margin: 0 !important; max-width: 100% !important; }
        button { min-height: 48px !important; font-size: 16px !important; font-weight: 700 !important; }
    }
    .touch-btn {
        background: linear-gradient(135deg, #ff007f 0%, #7928ca 50%, #0070f3 100%) !important;
        color: white !important; border: none !important; font-size: 1.1rem !important;
        font-weight: 800 !important; border-radius: 12px !important; padding: 12px 20px !important;
    }
    </style>
    <div style="text-align: center; margin-bottom: 16px; padding: 8px 0;">
        <h1 style="background: linear-gradient(90deg, #ff007f, #8a2be2, #00e5ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.2rem; font-weight: 900; margin: 0;">⚡ AI UNRESTRICTED STUDIO</h1>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 4px;">4K Text-to-Image & Smart Instruction Photo Editor • 100% Uncensored • Mobile & Android Ready</p>
    </div>
    """)
    
    with gr.Tabs():
        # TAB 1: SMART INSTRUCTION IMAGE EDITOR
        with gr.TabItem("🪄 Smart Photo Editor (Add Hat, Hair, Clothes)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1️⃣ Upload Your Photo")
                    i_img = gr.Image(label="Your Reference Photo", type="pil", height=280)
                    
                    gr.Markdown("### 2️⃣ Type What to Change")
                    i_prompt = gr.Textbox(
                        label="Command / Instruction Prompt",
                        placeholder="e.g. 'wearing a black cowboy hat', 'add sunglasses', 'change hair to curly fade', 'standing in Paris at sunset'",
                        lines=2
                    )
                    
                    gr.Markdown("#### ⚡ Quick Android One-Tap Presets:")
                    with gr.Row():
                        p_hat = gr.Button("🎩 Add Hat", size="sm")
                        p_glasses = gr.Button("🕶️ Sunglasses", size="sm")
                        p_hair = gr.Button("💇 Curly Hair", size="sm")
                        p_suit = gr.Button("👔 Black Suit", size="sm")
                        p_beach = gr.Button("🏖️ Beach", size="sm")
                        
                    p_hat.click(fn=lambda: "wearing a stylish black fedora hat", outputs=i_prompt)
                    p_glasses.click(fn=lambda: "wearing cool black designer sunglasses", outputs=i_prompt)
                    p_hair.click(fn=lambda: "change hairstyle to a modern curly textured fade", outputs=i_prompt)
                    p_suit.click(fn=lambda: "wearing a luxury tailored black Italian suit", outputs=i_prompt)
                    p_beach.click(fn=lambda: "standing on a tropical sunny beach with ocean background", outputs=i_prompt)
                    
                    with gr.Accordion("⚙️ Precision Tuning (Optional)", open=False):
                        i_strength = gr.Slider(0.15, 0.85, value=0.45, step=0.05, label="🎯 Edit Intensity (Lower = keeps original image more, Higher = more new changes)")
                        i_steps = gr.Slider(15, 40, value=25, step=1, label="Steps")
                        i_cfg = gr.Slider(4.0, 10.0, value=7.0, step=0.5, label="Prompt Adherence")
                        i_seed = gr.Textbox(value="-1", label="Seed (-1 = Random)")
                        
                    i_btn = gr.Button("🪄 Apply AI Edit Now", variant="primary", elem_classes=["touch-btn"])
                    
                with gr.Column(scale=1):
                    i_out = gr.Image(label="Edited Photo", type="pil")
                    i_status = gr.Textbox(label="Status", interactive=False)
                    
            i_btn.click(
                fn=edit_image_to_image,
                inputs=[i_img, i_prompt, i_strength, i_steps, i_cfg, i_seed],
                outputs=[i_out, i_status]
            )
            
        # TAB 2: UNRESTRICTED TEXT-TO-IMAGE
        with gr.TabItem("🎨 Unrestricted Text-to-Image (4K 0% Censored)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1️⃣ Describe Anything You Want")
                    t_prompt = gr.Textbox(
                        label="Prompt (Anything Allowed • 100% Unrestricted)",
                        placeholder="e.g. '8k raw photo of a cyberpunk samurai standing in neon rain, photorealistic, cinematic lighting, masterpiece'",
                        lines=3
                    )
                    t_neg = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="deformed, blurry, bad anatomy, disfigured",
                        lines=1
                    )
                    
                    t_ratio = gr.Dropdown(
                        label="📐 Aspect Ratio",
                        choices=list(RATIOS.keys()),
                        value="📱 9:16 (Story / Reel / Phone)"
                    )
                    
                    with gr.Accordion("⚙️ Quality & Seed Settings", open=False):
                        t_steps = gr.Slider(20, 50, value=30, step=1, label="Sampling Steps")
                        t_cfg = gr.Slider(4.0, 12.0, value=7.0, step=0.5, label="Prompt Guidance (CFG)")
                        t_seed = gr.Textbox(value="-1", label="Seed (-1 for Random)")
                        
                    t_btn = gr.Button("⚡ Generate 4K Image", variant="primary", elem_classes=["touch-btn"])
                    
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
