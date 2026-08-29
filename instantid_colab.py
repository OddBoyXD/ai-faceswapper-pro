import os
import gc
import cv2
import torch
import numpy as np
from PIL import Image
import gradio as gr
import insightface
from insightface.app import FaceAnalysis
from diffusers import (
    StableDiffusionXLInstantIDPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler
)

print("⚡ Initializing InstantID SDXL Engine on GPU...")

# 1. Initialize Fast Face Analysis
app_face = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app_face.prepare(ctx_id=0, det_size=(640, 640))

# 2. Load ControlNet & SDXL InstantID Pipeline
CHECKPOINT = "wangqixun/YamerMIX_v8" # High-realism SDXL checkpoint
INSTANTID_REPO = "InstantX/InstantID"

print("Loading ControlNet Model...")
controlnet = ControlNetModel.from_pretrained(
    INSTANTID_REPO,
    subfolder="ControlNetModel",
    torch_dtype=torch.float16
)

print("Loading SDXL Base Pipeline...")
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    CHECKPOINT,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
pipe.cuda()
pipe.load_ip_adapter_instantid(INSTANTID_REPO, subfolder="ip-adapter.bin")
pipe.enable_attention_slicing()

print("✅ InstantID SDXL Pipeline Ready!")

def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    width, height = image_pil.size
    kps_canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for i, p in enumerate(kps):
        cv2.circle(kps_canvas, (int(p[0]), int(p[1])), 4, color_list[i % len(color_list)], -1)
    return Image.fromarray(kps_canvas)

def run_instantid_swap(
    source_image,
    target_image,
    prompt="photorealistic, 8k uhd, cinematic portrait, highly detailed face, natural skin texture, masterpiece",
    negative_prompt="blurry, bad eyes, disfigured, cartoon, 3d render, illustration, bad anatomy, over-smoothed",
    identity_strength=0.80,
    adapter_strength=0.80,
    num_steps=22,
    guidance_scale=5.0,
    seed=-1,
    progress=gr.Progress(track_tqdm=True)
):
    if source_image is None or target_image is None:
        return None, "❌ Please provide both Source Face (Identity) and Target Image (Pose/Body)."

    # 1. Process Source Image (Face identity embedding)
    src_bgr = cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR)
    src_faces = app_face.get(src_bgr)
    if not src_faces:
        return None, "❌ No face detected in Source Image. Please upload a clear face photo."
    
    src_face = max(src_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    face_emb = src_face.embedding

    # 2. Process Target Image (Pose & Lighting reference)
    tgt_bgr = cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR)
    tgt_faces = app_face.get(tgt_bgr)
    if not tgt_faces:
        return None, "❌ No face detected in Target Image."

    tgt_face = max(tgt_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    
    # Scale target image to standard SDXL aspect ratio (max 1024x1024)
    w, h = target_image.size
    max_dim = max(w, h)
    scale = 1024.0 / max_dim
    new_w, new_h = int(w * scale) // 8 * 8, int(h * scale) // 8 * 8
    target_resized = target_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Scale landmark keypoints
    tgt_kps = tgt_face.kps * scale
    pose_image = draw_kps(target_resized, tgt_kps)

    # Set generator seed
    if seed == -1 or seed is None:
        seed = int(torch.randint(0, 2147483647, (1,)).item())
    generator = torch.Generator("cuda").manual_seed(seed)

    pipe.set_ip_adapter_scale(identity_strength)
    
    # Run SDXL Diffusion Generation
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_embeds=face_emb,
        image=pose_image,
        controlnet_conditioning_scale=adapter_strength,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=new_h,
        width=new_w,
        generator=generator
    ).images[0]

    gc.collect()
    torch.cuda.empty_cache()
    return output, f"✨ 1024x1024 InstantID Generation Complete! (Seed: {seed})"

# ── GRADIO UI ──
with gr.Blocks(title="InstantID 4K Face Swapper Pro (Colab GPU)") as demo:
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="background: linear-gradient(90deg, #f59e0b, #ec4899, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.4rem; font-weight: 900; margin: 0;">🌟 InstantID SDXL • 4K Photorealistic Face Swapper</h1>
        <p style="color: #94a3b8; font-size: 1.05rem; margin-top: 6px;">Hollywood-Grade Diffusion Synthesis • 1024x1024 Native • Natural Skin Pores & Realistic Light Bounces</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1️⃣ Source Face (Identity to Copy)")
            src_input = gr.Image(label="Source Face Photo", type="pil")
            
            gr.Markdown("### 2️⃣ Target Body / Pose / Scene")
            tgt_input = gr.Image(label="Target Image", type="pil")
            
            with gr.Accordion("⚙️ Pro Generation Settings", open=False):
                prompt_input = gr.Textbox(
                    label="Style Prompt",
                    value="photorealistic, 8k uhd, cinematic portrait, highly detailed face, natural skin texture, masterpiece, studio lighting"
                )
                neg_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="blurry, bad eyes, disfigured, cartoon, 3d render, illustration, bad anatomy, over-smoothed, noisy"
                )
                with gr.Row():
                    id_scale = gr.Slider(0.1, 1.0, value=0.80, step=0.05, label="👤 Identity Strength")
                    pose_scale = gr.Slider(0.1, 1.0, value=0.80, step=0.05, label="📐 Pose / Angle Lock")
                with gr.Row():
                    steps = gr.Slider(15, 35, value=22, step=1, label="⚡ Sampling Steps (Speed vs Quality)")
                    cfg = gr.Slider(2.0, 9.0, value=5.0, step=0.5, label="🎯 Prompt Match (CFG)")
                seed_box = gr.Number(value=-1, label="🎲 Seed (-1 for Random)")

            swap_btn = gr.Button("🚀 Generate 4K InstantID Swap", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            out_image = gr.Image(label="1024x1024 Photorealistic Result", type="pil")
            out_status = gr.Textbox(label="Status", interactive=False)
            
    swap_btn.click(
        fn=run_instantid_swap,
        inputs=[src_input, tgt_input, prompt_input, neg_prompt, id_scale, pose_scale, steps, cfg, seed_box],
        outputs=[out_image, out_status]
    )

if __name__ == "__main__":
    demo.launch(share=True)
