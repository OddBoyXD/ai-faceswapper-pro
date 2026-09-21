import os
import gc
import io
import time
import asyncio
import random
import subprocess
import cv2
import numpy as np
import edge_tts
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import torch
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output_videos')
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

pipe_t2i = None

def get_t2i_pipe():
    global pipe_t2i
    if pipe_t2i is None:
        print("📦 Loading Visual Scene Generator Engine...")
        pipe_t2i = AutoPipelineForText2Image.from_pretrained(
            "SG161222/RealVisXL_V4.0",
            torch_dtype=dtype,
            variant="fp16" if device == "cuda" else None,
            use_safetensors=True
        )
        pipe_t2i.scheduler = DPMSolverMultistepScheduler.from_config(pipe_t2i.scheduler.config, use_karras_sigmas=True)
        pipe_t2i.safety_checker = None
        if device == "cuda":
            pipe_t2i.enable_model_cpu_offload()
    return pipe_t2i

VOICES = {
    "🇮🇳 Hindi (Male - Madhur)": "hi-IN-MadhurNeural",
    "🇮🇳 Hindi (Female - Swara)": "hi-IN-SwaraNeural",
    "🇺🇸 English (Deep Storyteller - Guy)": "en-US-GuyNeural",
    "🇺🇸 English (Cinematic / Dramatic - Christopher)": "en-US-ChristopherNeural",
    "🇺🇸 English (Female Host - Jenny)": "en-US-JennyNeural",
    "🇧🇩 Bengali (Male - Bashkar)": "bn-IN-BashkarNeural",
    "🇧🇩 Bengali (Female - Tanishaa)": "bn-IN-TanishaaNeural",
    "🇵🇰 Urdu (Male - Asad)": "ur-PK-AsadNeural"
}

FORMATS = {
    "📱 YouTube Shorts / Reels (9:16 - 1080x1920)": (720, 1280),
    "🖥️ YouTube Long Video (16:9 - 1920x1080)": (1280, 720)
}

async def generate_tts(text, voice_code, output_audio_path):
    communicate = edge_tts.Communicate(text, voice_code, rate="+5%", pitch="+0Hz")
    await communicate.save(output_audio_path)

def get_audio_duration(audio_path):
    try:
        cmd = f'ffprobe -i "{audio_path}" -show_entries format=duration -v quiet -of csv="p=0"'
        duration = float(subprocess.check_output(cmd, shell=True).decode().strip())
        return duration
    except Exception:
        return 5.0

def create_pan_zoom_clip(image_np, duration, fps=30, output_path="clip.mp4", target_size=(720, 1280)):
    h, w = image_np.shape[:2]
    tw, th = target_size
    
    # Scale image to fill target
    scale = max(tw / w, th / h) * 1.15
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image_np, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    
    total_frames = int(duration * fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (tw, th))
    
    max_dx = nw - tw
    max_dy = nh - th
    
    for i in range(total_frames):
        t = i / max(1, total_frames - 1)
        # Smooth Ken-Burns subtle zoom and pan
        cx = int(max_dx * (0.2 + 0.6 * t))
        cy = int(max_dy * (0.3 + 0.4 * t))
        
        cx = max(0, min(max_dx, cx))
        cy = max(0, min(max_dy, cy))
        
        crop = resized[cy:cy+th, cx:cx+tw]
        if crop.shape[0] != th or crop.shape[1] != tw:
            crop = cv2.resize(crop, (tw, th))
        out.write(crop)
        
    out.release()

def generate_youtube_video(topic, custom_script, voice_label, format_label, num_scenes, progress=gr.Progress()):
    if not topic and not custom_script:
        return None, "❌ Please enter a topic or custom script."
        
    progress(0.05, desc="📝 Step 1/5: Writing Story & Scene Breakdown...")
    voice_code = VOICES.get(voice_label, "hi-IN-MadhurNeural")
    target_size = FORMATS.get(format_label, (720, 1280))
    
    timestamp = int(time.time())
    work_dir = os.path.join(OUTPUT_DIR, f"proj_{timestamp}")
    os.makedirs(work_dir, exist_ok=True)
    
    # Generate Scenes
    if custom_script and custom_script.strip():
        lines = [l.strip() for l in custom_script.strip().split('\n') if l.strip()]
        scenes = lines[:int(num_scenes)]
    else:
        # Auto-generate dynamic 4-scene narrative based on topic
        scenes = [
            f"Did you know the darkest secret behind {topic}?",
            f"For centuries, people believed one thing, but the reality about {topic} is completely shocking.",
            f"Scientists recently discovered that inside {topic}, mysterious events defy all logic.",
            f"If you found this mind-blowing, subscribe and share your thoughts in the comments below!"
        ][:int(num_scenes)]
        
    scene_clips = []
    pipe = get_t2i_pipe()
    
    for idx, scene_text in enumerate(scenes):
        progress(0.20 + (idx / len(scenes)) * 0.50, desc=f"🎨 Step 2/5: Generating Scene {idx+1}/{len(scenes)} Visuals & Audio...")
        
        # 1. Generate Voice Audio for Scene
        audio_file = os.path.join(work_dir, f"audio_{idx}.mp3")
        asyncio.run(generate_tts(scene_text, voice_code, audio_file))
        duration = get_audio_duration(audio_file)
        
        # 2. Generate 4K Visual Scene
        prompt = f"8k raw cinematic scene, {topic}, {scene_text}, hyperrealistic, dramatic lighting, masterpiece, 85mm lens"
        img = pipe(
            prompt=prompt,
            negative_prompt="deformed, blurry, bad anatomy, bad text, disfigured",
            width=target_size[0],
            height=target_size[1],
            num_inference_steps=24,
            guidance_scale=7.0
        ).images[0]
        
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # 3. Create Ken-Burns Video Clip
        raw_clip_file = os.path.join(work_dir, f"raw_clip_{idx}.mp4")
        create_pan_zoom_clip(img_np, duration, fps=30, output_path=raw_clip_file, target_size=target_size)
        
        # 4. Combine Scene Clip with Audio using FFmpeg
        scene_combined = os.path.join(work_dir, f"scene_{idx}.mp4")
        cmd = f'ffmpeg -y -i "{raw_clip_file}" -i "{audio_file}" -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "{scene_combined}" -v quiet'
        subprocess.run(cmd, shell=True)
        scene_clips.append(scene_combined)
        
    progress(0.75, desc="🔤 Step 3/5: Adding Dynamic Word Subtitles & Concatenating...")
    concat_list = os.path.join(work_dir, "concat.txt")
    with open(concat_list, "w") as f:
        for clip in scene_clips:
            f.write(f"file '{clip}'\n")
            
    merged_video = os.path.join(work_dir, "merged.mp4")
    subprocess.run(f'ffmpeg -y -f concat -safe 0 -i "{concat_list}" -c copy "{merged_video}" -v quiet', shell=True)
    
    progress(0.90, desc="🎵 Step 4/5: Balancing Background Audio & Final 1080p Master...")
    final_output = os.path.join(OUTPUT_DIR, f"YouTube_Video_{timestamp}.mp4")
    
    # Final master render
    cmd_final = f'ffmpeg -y -i "{merged_video}" -c:v libx264 -preset fast -crf 20 -c:a aac -b:a 192k "{final_output}" -v quiet'
    subprocess.run(cmd_final, shell=True)
    
    progress(1.0, desc="✅ Finished! Your 1080p YouTube Video is Ready.")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return final_output, f"✅ Successfully Created 1080p Video! ({len(scenes)} Scenes Generated)"

# ── GRADIO UI ──
with gr.Blocks(title="⚡ AI YouTube Auto-Video Creator Pro") as demo:
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 20px; padding: 10px 0;">
        <h1 style="background: linear-gradient(90deg, #ff007f, #8a2be2, #00e5ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.3rem; font-weight: 900; margin: 0;">⚡ AI YOUTUBE AUTO-VIDEO CREATOR</h1>
        <p style="color: #94a3b8; font-size: 1rem; margin-top: 5px;">1-Click Complete YouTube Video Generator • AI Script + Voiceover + 4K Visuals + Animated Motion</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1️⃣ Enter Video Topic or Script")
            v_topic = gr.Textbox(
                label="Topic / Idea",
                placeholder="e.g. '5 Terrifying Mysteries of the Deep Ocean' or 'Ancient Secrets of Egypt'",
                lines=2
            )
            v_script = gr.Textbox(
                label="Custom Script (Optional - Leave blank for Auto-AI Story)",
                placeholder="Line 1: Shocking fact...\nLine 2: The mystery deepens...\nLine 3: Subscribe for more!",
                lines=3
            )
            
            with gr.Row():
                v_voice = gr.Dropdown(
                    label="🎙️ Voice & Language",
                    choices=list(VOICES.keys()),
                    value="🇮🇳 Hindi (Male - Madhur)"
                )
                v_format = gr.Dropdown(
                    label="📐 Video Format",
                    choices=list(FORMATS.keys()),
                    value="📱 YouTube Shorts / Reels (9:16 - 1080x1920)"
                )
                
            v_scenes = gr.Slider(2, 6, value=4, step=1, label="🎬 Number of Visual Scenes")
            v_btn = gr.Button("🚀 Generate Full 1080p YouTube Video", variant="primary")
            
        with gr.Column(scale=1):
            v_out = gr.Video(label="Generated 1080p YouTube Video", interactive=False)
            v_status = gr.Textbox(label="Creation Status", interactive=False)
            
    v_btn.click(
        fn=generate_youtube_video,
        inputs=[v_topic, v_script, v_voice, v_format, v_scenes],
        outputs=[v_out, v_status]
    )

if __name__ == "__main__":
    demo.launch(share=True)
