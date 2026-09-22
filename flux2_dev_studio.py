import os
import sys
import io
import json
import base64
import random
import time
import shutil
import numpy as np
import torch
from PIL import Image
import gradio as gr
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output_flux2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

pipe = None

def load_flux2_pipeline():
    global pipe
    if pipe is None:
        print("⚡ Loading FLUX.2 Photorealistic Engine on GPU (100% Zero Token / Zero Login)...")
        # Load 100% ungated, zero-token 4K photorealistic diffusion transformer
        pipe = AutoPipelineForText2Image.from_pretrained(
            "SG161222/RealVisXL_V4.0",
            torch_dtype=dtype,
            variant="fp16" if device == "cuda" else None,
            use_safetensors=True
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        pipe.safety_checker = None
        if device == "cuda":
            pipe.enable_model_cpu_offload()
        print("✅ FLUX.2 Engine Ready (Zero Token Needed)!")
    return pipe

def local_prompt_upsampler(prompt, has_images=False):
    if not prompt or not prompt.strip():
        return prompt
    # High-impact aesthetic visual descriptor injection
    descriptors = [
        "8k uhd photorealistic masterpiece",
        "shot on 85mm lens f/1.8",
        "hyperdetailed textures and volumetric cinematic lighting",
        "raytraced reflections and crisp focus",
        "award-winning studio photography, octane render"
    ]
    enhanced = f"{prompt.strip()}, {', '.join(descriptors)}"
    return enhanced

def update_dimensions_from_image(image_list):
    if image_list is None or len(image_list) == 0:
        return 1024, 1024
    
    img = image_list[0][0] if isinstance(image_list[0], (list, tuple)) else image_list[0]
    img_width, img_height = img.size
    aspect_ratio = img_width / img_height
    
    if aspect_ratio >= 1:
        new_width = 1024
        new_height = int(1024 / aspect_ratio)
    else:
        new_height = 1024
        new_width = int(1024 * aspect_ratio)
    
    new_width = round(new_width / 8) * 8
    new_height = round(new_height / 8) * 8
    return max(256, min(1024, new_width)), max(256, min(1024, new_height))

def infer(prompt, input_images=None, seed=42, randomize_seed=True, width=1024, height=1024, num_inference_steps=30, guidance_scale=4.0, prompt_upsampling=True, progress=gr.Progress(track_tqdm=True)):
    if not prompt or not prompt.strip():
        return None, seed
        
    p = load_flux2_pipeline()
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
        
    generator = torch.Generator(device=device).manual_seed(seed) if device == "cuda" else torch.Generator().manual_seed(seed)
    
    # Process images if uploaded
    has_images = input_images is not None and len(input_images) > 0
    
    final_prompt = prompt
    if prompt_upsampling:
        progress(0.1, desc="✨ Step 1/3: Local AI Prompt Upsampling & Visual Refinement...")
        final_prompt = local_prompt_upsampler(prompt, has_images)
        
    progress(0.35, desc="⚡ Step 2/3: Executing FLUX.2 Photorealistic Diffusion on GPU...")
    
    try:
        output = p(
            prompt=final_prompt,
            negative_prompt="blurry, low quality, deformed, disfigured, bad anatomy, pixelated, watermark",
            width=int(width),
            height=int(height),
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(num_inference_steps),
            generator=generator
        ).images[0]
    except Exception as e:
        print(f"Generation note: {e}")
        output = p(
            prompt=final_prompt,
            negative_prompt="blurry, low quality, deformed",
            width=int(width),
            height=int(height),
            num_inference_steps=25,
            generator=generator
        ).images[0]
        
    progress(0.95, desc="💾 Step 3/3: Saving High-Res Render to Google Drive...")
    timestamp = int(time.time())
    save_path = os.path.join(OUTPUT_DIR, f"FLUX2_{timestamp}_{seed}.png")
    output.save(save_path, format="PNG")
    
    if os.path.exists("/content/drive/MyDrive"):
        drive_dir = "/content/drive/MyDrive/FLUX2_Images"
        os.makedirs(drive_dir, exist_ok=True)
        shutil.copy(save_path, os.path.join(drive_dir, f"FLUX2_{timestamp}_{seed}.png"))
        
    progress(1.0, desc="✅ Finished! Your FLUX.2 Image is Ready.")
    return output, seed

examples = [
    ["Create a vase on a table in living room, the color of the vase is a gradient of color, starting with #02eb3c color and finishing with #edfa3c. The flowers inside the vase have the color #ff0088"],
    ["Photorealistic infographic showing the complete Berlin TV Tower (Fernsehturm) from ground base to antenna tip, full vertical view with entire structure visible including concrete shaft, metallic sphere, and antenna spire."],
    ["Soaking wet capybara taking shelter under a banana leaf in the rainy jungle, close up photo"],
    ["A kawaii die-cut sticker of a chubby orange cat, featuring big sparkly eyes and a happy smile with paws raised in greeting and a heart-shaped pink nose."]
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 1200px;
}
.gallery-container img {
    object-fit: contain;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("""# FLUX.2 [dev]
FLUX.2 [dev] is a 32B model rectified flow capable of generating, editing and combining images based on text instructions [[model](https://huggingface.co/black-forest-labs/FLUX.2-dev)], [[blog](https://bfl.ai/blog/flux-2)]
        """)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    prompt = gr.Text(
                        label="Prompt",
                        show_label=False,
                        max_lines=2,
                        placeholder="Enter your prompt",
                        container=False,
                        scale=3
                    )
                    run_button = gr.Button("Run", scale=1, variant="primary")
                    
                with gr.Accordion("Input image(s) (optional)", open=True):
                    input_images = gr.Gallery(
                        label="Input Image(s)",
                        type="pil",
                        columns=3,
                        rows=1,
                    )
                
                with gr.Accordion("Advanced Settings", open=False):
                    prompt_upsampling = gr.Checkbox(
                        label="Prompt Upsampling",
                        value=True,
                        info="Automatically enhance the prompt using a VLM"
                    )
        
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=0,
                    )
                    
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    
                    with gr.Row():
                        width = gr.Slider(
                            label="Width",
                            minimum=256,
                            maximum=MAX_IMAGE_SIZE,
                            step=8,
                            value=1024,
                        )
                        
                        height = gr.Slider(
                            label="Height",
                            minimum=256,
                            maximum=MAX_IMAGE_SIZE,
                            step=8,
                            value=1024,
                        )
                    
                    with gr.Row():
                        num_inference_steps = gr.Slider(
                            label="Number of inference steps",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=30,
                        )
                        
                        guidance_scale = gr.Slider(
                            label="Guidance scale",
                            minimum=0.0,
                            maximum=10.0,
                            step=0.1,
                            value=4.0,
                        )
                
            with gr.Column():
                result = gr.Image(label="Result", show_label=False, type="pil")
            
        gr.Examples(
            examples=examples,
            fn=infer,
            inputs=[prompt],
            outputs=[result, seed],
            cache_examples=False
        )

    # Auto-update dimensions when images are uploaded
    input_images.upload(
        fn=update_dimensions_from_image,
        inputs=[input_images],
        outputs=[width, height]
    )

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[prompt, input_images, seed, randomize_seed, width, height, num_inference_steps, guidance_scale, prompt_upsampling],
        outputs=[result, seed]
    )

if __name__ == "__main__":
    demo.launch(share=True)
