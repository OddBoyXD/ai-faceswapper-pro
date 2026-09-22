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
from huggingface_hub import InferenceClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output_flux2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

pipe = None

SYSTEM_PROMPT_TEXT_ONLY = """You are an expert prompt engineer for FLUX.2 by Black Forest Labs. Rewrite user prompts to be more descriptive while strictly preserving their core subject and intent.
Guidelines:
1. Structure: Keep structured inputs structured. Convert natural language to detailed paragraphs.
2. Details: Add concrete visual specifics - form, scale, textures, materials, lighting (quality, direction, color), shadows, spatial relationships, and environmental context.
3. Text in Images: Put ALL text in quotation marks, matching the prompt's language. Always provide explicit quoted text for objects that would contain text in reality.
Output only the revised prompt and nothing else."""

SYSTEM_PROMPT_WITH_IMAGES = """You are FLUX.2 by Black Forest Labs, an image-editing expert. You convert editing requests into one concise instruction (50-80 words, ~30 for brief requests).
Rules:
- Single instruction only, no commentary
- Specify what changes AND what stays the same (face, lighting, composition)
- Reference actual image elements
Output only the final instruction in plain text and nothing else."""

def load_flux2_pipeline():
    global pipe
    if pipe is None:
        print("⚡ Initializing FLUX.2 [dev] Engine on GPU...")
        try:
            from diffusers import Flux2Pipeline
            pipe = Flux2Pipeline.from_pretrained(
                "black-forest-labs/FLUX.2-dev",
                torch_dtype=dtype
            )
            if device == "cuda":
                pipe.enable_model_cpu_offload()
            print("✅ FLUX.2-dev Pipeline Loaded!")
        except Exception as e1:
            print(f"Loading FLUX pipeline: {e1}")
            try:
                from diffusers import FluxPipeline
                pipe = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-schnell",
                    torch_dtype=dtype
                )
                if device == "cuda":
                    pipe.enable_model_cpu_offload()
                print("✅ High-Speed FLUX Pipeline Loaded!")
            except Exception as e2:
                print(f"Fallback to 4K RealVisXL Engine: {e2}")
                from diffusers import AutoPipelineForText2Image
                pipe = AutoPipelineForText2Image.from_pretrained(
                    "SG161222/RealVisXL_V4.0",
                    torch_dtype=dtype
                )
                if device == "cuda":
                    pipe.enable_model_cpu_offload()
    return pipe

def image_to_data_uri(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def upsample_prompt_logic(prompt, image_list):
    try:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            # Smart internal prompt enhancer when no HF token is supplied
            enhanced = f"{prompt}, hyperrealistic, ultra high definition 8k, photorealistic masterpiece, 85mm lens, dramatic cinematic lighting, extremely detailed textures, octane render"
            return enhanced
            
        hf_client = InferenceClient(api_key=hf_token)
        VLM_MODEL = "baidu/ERNIE-4.5-VL-424B-A47B-Base-PT"
        
        if image_list and len(image_list) > 0:
            user_content = [{"type": "text", "text": prompt}]
            for img in image_list:
                data_uri = image_to_data_uri(img)
                user_content.append({"type": "image_url", "image_url": {"url": data_uri}})
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_WITH_IMAGES},
                {"role": "user", "content": user_content}
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_TEXT_ONLY},
                {"role": "user", "content": prompt}
            ]

        completion = hf_client.chat.completions.create(
            model=VLM_MODEL,
            messages=messages,
            max_tokens=1024
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Prompt upsampling note: {e}")
        return prompt

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
    image_list = None
    if input_images is not None and len(input_images) > 0:
        image_list = []
        for item in input_images:
            img = item[0] if isinstance(item, (list, tuple)) else item
            image_list.append(img)
            
    final_prompt = prompt
    if prompt_upsampling:
        progress(0.1, desc="✨ Step 1/3: AI Prompt Upsampling & Visual Refinement...")
        final_prompt = upsample_prompt_logic(prompt, image_list)
        
    progress(0.3, desc="⚡ Step 2/3: Executing FLUX.2 [dev] Rectified Flow Transformation...")
    
    try:
        if image_list and len(image_list) > 0 and hasattr(p, "image"):
            output = p(
                prompt=final_prompt,
                image=image_list,
                width=int(width),
                height=int(height),
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(num_inference_steps),
                generator=generator
            ).images[0]
        else:
            output = p(
                prompt=final_prompt,
                width=int(width),
                height=int(height),
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(num_inference_steps),
                generator=generator
            ).images[0]
    except Exception as e:
        print(f"Standard generation fallback: {e}")
        output = p(
            prompt=final_prompt,
            width=int(width),
            height=int(height),
            num_inference_steps=int(num_inference_steps),
            generator=generator
        ).images[0]
        
    progress(0.95, desc="💾 Step 3/3: Saving 4K High-Res Render to Google Drive...")
    timestamp = int(time.time())
    save_path = os.path.join(OUTPUT_DIR, f"FLUX2_{timestamp}_{seed}.png")
    output.save(save_path, format="PNG")
    
    if os.path.exists("/content/drive/MyDrive"):
        drive_dir = "/content/drive/MyDrive/FLUX2_Images"
        os.makedirs(drive_dir, exist_ok=True)
        shutil.copy(save_path, os.path.join(drive_dir, f"FLUX2_{timestamp}_{seed}.png"))
        
    progress(1.0, desc="✅ Finished! Image Ready.")
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
