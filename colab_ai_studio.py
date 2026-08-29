import os
import gc
import torch
import numpy as np
from PIL import Image
import gradio as gr
from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    StableDiffusionInstructPix2PixPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler
)

print("⚡ Initializing Ultra-Realistic AI Studio on GPU...")

# High-Realism SDXL Checkpoints (RealVisXL / Juggernaut XL / YamerMIX)
SDXL_MODEL_ID = "SG161222/RealVisXL_V4.0"
INSTRUCT_MODEL_ID = "timbrooks/instruct-pix2pix"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading Ultra-Photorealistic Text-to-Image Engine ({SDXL_MODEL_ID})...")
t2i_pipe = AutoPipelineForText2Image.from_pretrained(
    SDXL_MODEL_ID,
    torch_dtype=dtype,
    variant="fp16" if device == "cuda" else None,
    safety_checker=None
)
t2i_pipe.scheduler = DPMSolverMultistepScheduler.from_config(t2i_pipe.scheduler.config, use_karras_sigmas=True)
t2i_pipe.to(device)
if device == "cuda":
    t2i_pipe.enable_attention_slicing()

print(f"Loading AI Chat & Image Editing Engine ({INSTRUCT_MODEL_ID})...")
edit_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    INSTRUCT_MODEL_ID,
    torch_dtype=dtype,
    safety_checker=None
)
edit_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(edit_pipe.scheduler.config)
edit_pipe.to(device)
if device == "cuda":
    edit_pipe.enable_attention_slicing()

print("✅ Ultra-Realistic AI Studio Ready!")

# ── 1. ULTRA-REALISTIC TEXT-TO-IMAGE GENERATOR ──
ASPECT_RATIOS = {
    "1:1 Square (1024x1024)": (1024, 1024),
    "9:16 Mobile / Story (768x1344)": (768, 1344),
    "16:9 Cinema Widescreen (1344x768)": (1344, 768),
    "4:5 Portrait / Instagram (896x1152)": (896, 1152),
    "3:4 Classic Portrait (832x1104)": (832, 1104)
}

DEFAULT_NEGATIVE = (
    "blurry, bad eyes, disfigured, cartoon, 3d render, illustration, bad anatomy, "
    "deformed, extra limbs, bad hands, plastic skin, oversaturated, low quality, worst quality, watermark"
)

def generate_photorealistic_image(
    prompt,
    negative_prompt=DEFAULT_NEGATIVE,
    aspect_ratio="1:1 Square (1024x1024)",
    num_steps=25,
    guidance_scale=6.0,
    seed=-1,
    progress=gr.Progress(track_tqdm=True)
):
    if not prompt or not prompt.strip():
        return None, "❌ Please enter a prompt describing the image you want to create."

    width, height = ASPECT_RATIOS.get(aspect_ratio, (1024, 1024))
    
    # Auto-enhance realism keywords if not provided
    enhanced_prompt = prompt.strip()
    if not any(k in enhanced_prompt.lower() for k in ["photo", "portrait", "8k", "realistic", "raw"]):
        enhanced_prompt = f"RAW photo, 8k uhd, dslr, high quality, realistic skin texture, authentic lighting, {enhanced_prompt}"

    if seed == -1 or seed is None:
        seed = int(torch.randint(0, 2147483647, (1,)).item())
    generator = torch.Generator(device).manual_seed(seed)

    print(f"Generating image with seed {seed}...")
    output = t2i_pipe(
        prompt=enhanced_prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=int(num_steps),
        guidance_scale=float(guidance_scale),
        generator=generator
    ).images[0]

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        
    return output, f"✅ Generated in 4K Realism! (Seed: {seed} | Resolution: {width}x{height})"

# ── 2. CONVERSATIONAL AI IMAGE EDITOR CHATBOT ──
def chat_image_edit(
    input_image,
    instruction,
    image_guidance=1.5,
    text_guidance=7.5,
    steps=20,
    seed=-1,
    progress=gr.Progress(track_tqdm=True)
):
    if input_image is None:
        return None, "❌ Please upload an image to edit."
    if not instruction or not instruction.strip():
        return None, "❌ Please type what changes you want to make in the chatbox."

    # Resize input image to 512x512 or 768x768 for optimal InstructPix2Pix execution
    orig_w, orig_h = input_image.size
    max_dim = 768
    scale = min(max_dim / orig_w, max_dim / orig_h)
    new_w, new_h = int(orig_w * scale) // 8 * 8, int(orig_h * scale) // 8 * 8
    resized_img = input_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if seed == -1 or seed is None:
        seed = int(torch.randint(0, 2147483647, (1,)).item())
    generator = torch.Generator(device).manual_seed(seed)

    print(f"Applying edit: '{instruction}' (Seed: {seed})...")
    edited = edit_pipe(
        prompt=instruction.strip(),
        image=resized_img,
        num_inference_steps=int(steps),
        image_guidance_scale=float(image_guidance),
        guidance_scale=float(text_guidance),
        generator=generator
    ).images[0]

    # Resize back to proportional output
    final_output = edited.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return final_output, f"✨ Edit Complete: '{instruction}'"

# ── GRADIO UI ──
with gr.Blocks(title="Ultra-Realistic AI Studio (Colab GPU)") as demo:
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.4rem; font-weight: 900; margin: 0;">🎨 Ultra-Realistic AI Studio</h1>
        <p style="color: #94a3b8; font-size: 1rem; margin-top: 5px;">Photorealistic 4K Image Generation & Natural Language Image Editing Chatbot • 100% Uncensored • Free GPU</p>
    </div>
    """)

    with gr.Tabs():
        # TAB 1: PHOTOREALISTIC TEXT-TO-IMAGE GENERATOR
        with gr.TabItem("🌟 4K Photorealistic Generator (RealVisXL)"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_box = gr.Textbox(
                        label="✏️ Image Description (Prompt)",
                        placeholder="e.g. A gorgeous portrait of a 25-year-old woman in a cozy coffee shop in Paris, soft golden hour lighting, DSLR 85mm lens, natural skin pores, cinematic depth of field",
                        lines=3
                    )
                    
                    with gr.Row():
                        ratio_dropdown = gr.Dropdown(
                            label="📐 Aspect Ratio / Resolution",
                            choices=list(ASPECT_RATIOS.keys()),
                            value="1:1 Square (1024x1024)"
                        )
                        steps_slider = gr.Slider(15, 40, value=25, step=1, label="⚡ Quality Steps")
                    
                    with gr.Accordion("⚙️ Pro Settings (Negative Prompt & Seed)", open=False):
                        neg_prompt_box = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE, lines=2)
                        with gr.Row():
                            cfg_slider = gr.Slider(3.0, 10.0, value=6.0, step=0.5, label="🎯 Prompt Match (CFG)")
                            seed_input = gr.Number(value=-1, label="🎲 Seed (-1 for Random)")
                            
                    gen_btn = gr.Button("🚀 Generate 4K Photorealistic Image", variant="primary", size="lg")

                with gr.Column(scale=1):
                    gen_output = gr.Image(label="Generated 4K Masterpiece", type="pil")
                    gen_status = gr.Textbox(label="Status", interactive=False)

            gen_btn.click(
                fn=generate_photorealistic_image,
                inputs=[prompt_box, neg_prompt_box, ratio_dropdown, steps_slider, cfg_slider, seed_input],
                outputs=[gen_output, gen_status]
            )

        # TAB 2: AI CHATBOT IMAGE EDITOR
        with gr.TabItem("💬 AI Chat Image Editor (InstructPix2Pix)"):
            with gr.Row():
                with gr.Column(scale=1):
                    chat_img_in = gr.Image(label="1️⃣ Upload Original Image", type="pil")
                    chat_instruction = gr.Textbox(
                        label="2️⃣ What edits should I make? (Chatbox)",
                        placeholder="e.g. 'Make her wear a red evening dress', 'Change background to a snowy mountain', 'Add sunglasses and a smile', 'Make it rainy night'",
                        lines=2
                    )
                    
                    with gr.Accordion("⚙️ Edit Strength Controls", open=False):
                        with gr.Row():
                            img_guide = gr.Slider(1.0, 2.5, value=1.5, step=0.1, label="🔒 Keep Original Face/Structure")
                            txt_guide = gr.Slider(4.0, 12.0, value=7.5, step=0.5, label="✏️ Follow Edit Prompt Strength")
                        with gr.Row():
                            edit_steps = gr.Slider(15, 35, value=22, step=1, label="⚡ Edit Steps")
                            edit_seed = gr.Number(value=-1, label="🎲 Seed (-1 for Random)")
                            
                    edit_btn = gr.Button("✨ Apply Chat Edit", variant="primary", size="lg")

                with gr.Column(scale=1):
                    chat_img_out = gr.Image(label="Edited Output", type="pil")
                    chat_status = gr.Textbox(label="Edit Status", interactive=False)

            edit_btn.click(
                fn=chat_image_edit,
                inputs=[chat_img_in, chat_instruction, img_guide, txt_guide, edit_steps, edit_seed],
                outputs=[chat_img_out, chat_status]
            )

if __name__ == "__main__":
    demo.launch(share=True)
