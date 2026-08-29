import os
import gc
import io
import time
import urllib.request
import cv2
import requests
import numpy as np
from PIL import Image
import onnxruntime as ort
import gradio as gr
import insightface
from insightface.app import FaceAnalysis

# Dynamic base directory (Works everywhere: Colab, Kaggle, Linux VPS, Windows, Mac)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

INSWAPPER_PATH = os.path.join(MODELS_DIR, 'inswapper_128.onnx')
GFPGAN_PATH = os.path.join(MODELS_DIR, 'gfpgan_1.4.onnx')

# ── AUTO-DOWNLOAD MODELS ON STARTUP (IF MISSING) ──
DOWNLOAD_URLS = {
    INSWAPPER_PATH: "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
    GFPGAN_PATH: "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.4.onnx"
}

def ensure_models_downloaded():
    for path, url in DOWNLOAD_URLS.items():
        if not os.path.exists(path) or os.path.getsize(path) < 1000000:
            print(f"Downloading {os.path.basename(path)}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as resp, open(path, 'wb') as f:
                total_size = int(resp.headers.get('Content-Length', 0))
                downloaded = 0
                chunk_size = 1024 * 1024
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded//(1024*1024)}MB / {total_size//(1024*1024)}MB)", end="")
            print(f"\n✅ Successfully downloaded {os.path.basename(path)}")

ensure_models_downloaded()

sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 4
sess_opts.inter_op_num_threads = 4
sess_opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Check if CUDA is available
available_providers = ort.get_available_providers()
if 'CUDAExecutionProvider' in available_providers:
    PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ctx_id = 0
    print("🚀 Running with NVIDIA CUDA GPU Acceleration!")
else:
    PROVIDERS = ['CPUExecutionProvider']
    ctx_id = -1
    print("⚡ Running with Multi-Threaded Parallel CPU Engine")

print("Loading Face Analysis (InsightFace)...")
app_face = FaceAnalysis(name='buffalo_l', providers=PROVIDERS)
app_face.prepare(ctx_id=ctx_id, det_size=(512, 512))

print("Loading InSwapper-128...")
swapper = insightface.model_zoo.get_model(INSWAPPER_PATH, providers=PROVIDERS, session_options=sess_opts)

print("Loading GFPGAN v1.4...")
gfpgan_sess = ort.InferenceSession(GFPGAN_PATH, sess_options=sess_opts, providers=PROVIDERS)

FFHQ_512_KPS = np.array([
    [192.9814, 239.9470],
    [318.9027, 240.3436],
    [256.0000, 314.0409],
    [201.2611, 371.4104],
    [313.0890, 371.1511]
], dtype=np.float32)

def load_image_from_source(uploaded_img, url_str=""):
    if uploaded_img is not None:
        return np.array(uploaded_img)
    if url_str and isinstance(url_str, str) and url_str.strip():
        url = url_str.strip()
        try:
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            if resp.status_code == 200:
                pil_img = Image.open(io.BytesIO(resp.content)).convert('RGB')
                return np.array(pil_img)
        except Exception as e:
            print(f"Error loading URL {url}:", e)
    return None

def fast_hd_enhance(img, face_box):
    x1, y1, x2, y2 = [int(v) for v in face_box]
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 > x1 and y2 > y1:
        face_roi = img[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(face_roi, (0, 0), 1.5)
        sharpened = cv2.addWeighted(face_roi, 1.35, blurred, -0.35, 0)
        img[y1:y2, x1:x2] = sharpened
    return img

def restore_face_gfpgan(img_bgr, face_kps):
    try:
        M, _ = cv2.estimateAffinePartial2D(face_kps[:5].astype(np.float32), FFHQ_512_KPS)
        if M is None:
            return img_bgr
            
        aimg = cv2.warpAffine(img_bgr, M, (512, 512), borderMode=cv2.BORDER_REPLICATE)
        aimg_rgb = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        aimg_norm = (aimg_rgb - 0.5) / 0.5
        aimg_tensor = aimg_norm.transpose(2, 0, 1)[None, ...].astype(np.float32)
        
        out = gfpgan_sess.run(None, {'input': aimg_tensor})[0]
        out_rgb = ((out[0].transpose(1, 2, 0) * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_BGR2RGB)
        
        IM = cv2.invertAffineTransform(M)
        img_h, img_w = img_bgr.shape[:2]
        
        mask = np.ones((512, 512), dtype=np.float32)
        cv2.rectangle(mask, (0, 0), (511, 511), 0, 25)
        mask = cv2.GaussianBlur(mask, (35, 35), 0)
        
        warped_face = cv2.warpAffine(out_bgr, IM, (img_w, img_h), borderMode=cv2.BORDER_REPLICATE)
        warped_mask = cv2.warpAffine(mask, IM, (img_w, img_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_mask = np.clip(warped_mask, 0.0, 1.0)[:, :, None]
        
        result = (img_bgr.astype(np.float32) * (1.0 - warped_mask) + warped_face.astype(np.float32) * warped_mask).clip(0, 255).astype(np.uint8)
        return result
    except Exception as e:
        print("Restoration error:", e)
        return img_bgr

# ── 1. SINGLE FACE SWAP ──
def single_face_swap(source_img=None, source_url="", target_img=None, target_url="", quality_mode="⚡ Lightning Fast (3s)", *args, **kwargs):
    src_np = load_image_from_source(source_img, source_url)
    tgt_np = load_image_from_source(target_img, target_url)
    
    if src_np is None:
        return None, "❌ Please provide Source Face (upload photo or paste image URL)."
    if tgt_np is None:
        return None, "❌ Please provide Target Image (upload photo or paste image URL)."
    
    source_bgr = cv2.cvtColor(src_np, cv2.COLOR_RGB2BGR)
    target_bgr = cv2.cvtColor(tgt_np, cv2.COLOR_RGB2BGR)
    
    source_faces = app_face.get(source_bgr)
    if not source_faces:
        return None, "❌ No face detected in Source Image. Please provide a clear face photo."
    
    target_faces = app_face.get(target_bgr)
    if not target_faces:
        return None, "❌ No face detected in Target Image."
    
    source_face = max(source_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    target_face = max(target_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    
    result = swapper.get(target_bgr.copy(), target_face, source_face, paste_back=True)
    
    if "Ultra" in str(quality_mode) or "GFPGAN" in str(quality_mode):
        result = restore_face_gfpgan(result, target_face.kps)
        msg = "✅ 1:1 Ultra HD Swap Complete!"
    else:
        result = fast_hd_enhance(result, target_face.bbox)
        msg = "⚡ Lightning Fast Swap Complete!"
        
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    gc.collect()
    return result_rgb, msg

# ── 2. TWO-PERSON / GROUP SWAP WITH VISUAL SELECTION ──
def detect_all_group_persons(target_img=None, target_url=""):
    tgt_np = load_image_from_source(target_img, target_url)
    if tgt_np is None:
        return [], gr.Dropdown(choices=[], value=None), gr.Dropdown(choices=[], value=None), "❌ Please upload a group photo first."
        
    target_bgr = cv2.cvtColor(tgt_np, cv2.COLOR_RGB2BGR)
    faces = app_face.get(target_bgr)
    
    if not faces:
        return [], gr.Dropdown(choices=[], value=None), gr.Dropdown(choices=[], value=None), "❌ No faces detected in photo."
        
    sorted_faces = sorted(faces, key=lambda f: f.bbox[0])
    thumbnails = []
    choices = []
    h, w = target_bgr.shape[:2]
    
    for i, face in enumerate(sorted_faces):
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        pad_x = int((x2 - x1) * 0.25)
        pad_y = int((y2 - y1) * 0.25)
        cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        
        crop_bgr = target_bgr[cy1:cy2, cx1:cx2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        
        pos_desc = "Left" if i == 0 else ("Right" if i == len(sorted_faces)-1 else f"Middle #{i+1}")
        label = f"Person #{i+1} ({pos_desc})"
        thumbnails.append((crop_pil, label))
        choices.append(label)
        
    val1 = choices[0] if len(choices) > 0 else None
    val2 = choices[1] if len(choices) > 1 else val1
    
    status_msg = f"✅ Detected {len(sorted_faces)} people! You can now choose who gets Face #1 and who gets Face #2."
    return thumbnails, gr.Dropdown(choices=choices, value=val1), gr.Dropdown(choices=choices, value=val2), status_msg

def two_face_swap_selected(face1_img=None, f1_url="", face2_img=None, f2_url="", target_img=None, target_url="", target_person_1="Person #1", target_person_2="Person #2", quality_mode="⚡ Lightning Fast (3s)", *args, **kwargs):
    tgt_np = load_image_from_source(target_img, target_url)
    if tgt_np is None:
        return None, "❌ Please provide Target Group Image."
        
    f1_np = load_image_from_source(face1_img, f1_url)
    f2_np = load_image_from_source(face2_img, f2_url)
    
    if f1_np is None and f2_np is None:
        return None, "❌ Please provide at least one replacement face."
        
    target_bgr = cv2.cvtColor(tgt_np, cv2.COLOR_RGB2BGR)
    target_faces = app_face.get(target_bgr)
    
    if not target_faces:
        return None, "❌ No faces detected in target image."
        
    sorted_faces = sorted(target_faces, key=lambda f: f.bbox[0])
    
    try:
        idx1 = int(target_person_1.split("#")[1].split()[0].replace(")", "")) - 1
        idx1 = max(0, min(len(sorted_faces)-1, idx1))
    except:
        idx1 = 0
        
    try:
        idx2 = int(target_person_2.split("#")[1].split()[0].replace(")", "")) - 1
        idx2 = max(0, min(len(sorted_faces)-1, idx2))
    except:
        idx2 = 1 if len(sorted_faces) > 1 else 0
        
    tf1 = sorted_faces[idx1]
    tf2 = sorted_faces[idx2]
        
    result = target_bgr.copy()
    swapped_count = 0
    
    if f1_np is not None:
        f1_bgr = cv2.cvtColor(f1_np, cv2.COLOR_RGB2BGR)
        f1_faces = app_face.get(f1_bgr)
        if f1_faces:
            sf1 = max(f1_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            result = swapper.get(result, tf1, sf1, paste_back=True)
            if "Ultra" in str(quality_mode):
                result = restore_face_gfpgan(result, tf1.kps)
            else:
                result = fast_hd_enhance(result, tf1.bbox)
            swapped_count += 1
            
    if f2_np is not None:
        f2_bgr = cv2.cvtColor(f2_np, cv2.COLOR_RGB2BGR)
        f2_faces = app_face.get(f2_bgr)
        if f2_faces:
            sf2 = max(f2_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            result = swapper.get(result, tf2, sf2, paste_back=True)
            if "Ultra" in str(quality_mode):
                result = restore_face_gfpgan(result, tf2.kps)
            else:
                result = fast_hd_enhance(result, tf2.bbox)
            swapped_count += 1
            
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    gc.collect()
    return result_rgb, f"✅ Successfully swapped {swapped_count} selected people in the photo!"

# ── 3. SINGLE-PERSON SELECTOR (3+ PEOPLE) ──
def detect_and_crop_faces(target_img=None, target_url=""):
    tgt_np = load_image_from_source(target_img, target_url)
    if tgt_np is None:
        return [], gr.Radio(choices=[], value=None), "❌ Please upload a target image first."
        
    target_bgr = cv2.cvtColor(tgt_np, cv2.COLOR_RGB2BGR)
    faces = app_face.get(target_bgr)
    
    if not faces:
        return [], gr.Radio(choices=[], value=None), "❌ No faces detected in this photo."
        
    sorted_faces = sorted(faces, key=lambda f: f.bbox[0])
    thumbnails = []
    choices = []
    h, w = target_bgr.shape[:2]
    
    for i, face in enumerate(sorted_faces):
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        pad_x = int((x2 - x1) * 0.25)
        pad_y = int((y2 - y1) * 0.25)
        cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        
        crop_bgr = target_bgr[cy1:cy2, cx1:cx2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        
        label = f"Person #{i+1} (X: {x1})"
        thumbnails.append((crop_pil, label))
        choices.append(label)
        
    status_msg = f"✅ Detected {len(sorted_faces)} people in the photo! Select who to swap below."
    return thumbnails, gr.Radio(choices=choices, value=choices[0]), status_msg

def swap_specific_person(source_img=None, source_url="", target_img=None, target_url="", selected_person="Person #1", quality_mode="⚡ Lightning Fast (3s)", *args, **kwargs):
    src_np = load_image_from_source(source_img, source_url)
    tgt_np = load_image_from_source(target_img, target_url)
    
    if src_np is None:
        return None, "❌ Please upload your Replacement Face."
    if tgt_np is None:
        return None, "❌ Please upload the Target Group Photo."
        
    source_bgr = cv2.cvtColor(src_np, cv2.COLOR_RGB2BGR)
    target_bgr = cv2.cvtColor(tgt_np, cv2.COLOR_RGB2BGR)
    
    source_faces = app_face.get(source_bgr)
    if not source_faces:
        return None, "❌ No face detected in Replacement Face."
    target_faces = app_face.get(target_bgr)
    if not target_faces:
        return None, "❌ No faces detected in Target Image."
        
    source_face = max(source_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    sorted_target_faces = sorted(target_faces, key=lambda f: f.bbox[0])
    
    try:
        idx = int(selected_person.split("#")[1].split()[0]) - 1
        if idx < 0 or idx >= len(sorted_target_faces):
            idx = 0
    except:
        idx = 0
        
    target_face = sorted_target_faces[idx]
    result = swapper.get(target_bgr.copy(), target_face, source_face, paste_back=True)
    
    if "Ultra" in str(quality_mode):
        result = restore_face_gfpgan(result, target_face.kps)
    else:
        result = fast_hd_enhance(result, target_face.bbox)
        
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    gc.collect()
    return result_rgb, f"✅ Successfully replaced {selected_person} in the group photo!"

# ── GRADIO UI ──
with gr.Blocks(title="AI Face Swapper Pro") as demo:
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="background: linear-gradient(90deg, #00e5ff, #8a2be2, #ff007f); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.3rem; font-weight: 800; margin: 0;">⚡ AI FACE SWAPPER PRO</h1>
        <p style="color: #94a3b8; font-size: 1rem; margin-top: 5px;">Visual Person Selector & Multi-Face Swap • 1:1 Pixel Accuracy • Photos & URLs • 100% Free</p>
    </div>
    """)
    
    with gr.Tabs():
        # TAB 1: SINGLE FACE SWAP
        with gr.TabItem("👤 Single Face Swap"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 1️⃣ Source Face (Person to Copy)")
                    s_src = gr.Image(label="Upload Source Photo", type="pil")
                    s_src_url = gr.Textbox(label="...OR Paste Source Image URL", placeholder="https://example.com/face.jpg")
                    
                    gr.Markdown("### 2️⃣ Target Body / Scene")
                    s_tgt = gr.Image(label="Upload Target Image", type="pil")
                    s_tgt_url = gr.Textbox(label="...OR Paste Target Image URL", placeholder="https://example.com/body.jpg")
                    
                    s_mode = gr.Radio(
                        ["⚡ Lightning Fast (3s)", "✨ Ultra 4K GFPGAN (Max Quality)"],
                        value="⚡ Lightning Fast (3s)",
                        label="⚡ Speed / Quality Mode"
                    )
                    s_btn = gr.Button("⚡ Swap Face Now", variant="primary")
                with gr.Column():
                    s_out = gr.Image(label="Swapped Result", type="pil")
                    s_status = gr.Textbox(label="Status", interactive=False)
                    
            s_btn.click(
                fn=single_face_swap,
                inputs=[s_src, s_src_url, s_tgt, s_tgt_url, s_mode],
                outputs=[s_out, s_status]
            )
            
        # TAB 2: TWO-PERSON SWAP WITH VISUAL SELECTION & MAPPING
        with gr.TabItem("👥 2-Person Custom Swap (Choose Target)"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 1️⃣ Upload Group Photo")
                    m_tgt = gr.Image(label="Upload Group Photo (2+ People)", type="pil")
                    m_tgt_url = gr.Textbox(label="...OR Paste Group Image URL", placeholder="https://example.com/group.jpg")
                    m_detect_btn = gr.Button("🔍 Detect & List All People in Photo", variant="secondary")
                    
                    gr.Markdown("### 2️⃣ Detected People in Group")
                    m_gallery = gr.Gallery(label="Detected Face Cards", columns=4, height=130, object_fit="contain")
                    
                    gr.Markdown("### 3️⃣ Assign Replacement Faces")
                    with gr.Row():
                        with gr.Column():
                            m_target_1 = gr.Dropdown(label="🎯 Person to Replace with Face #1", choices=["Person #1 (Left)"], value="Person #1 (Left)")
                            m_f1 = gr.Image(label="👉 New Face #1", type="pil")
                            m_f1_url = gr.Textbox(label="OR URL #1", placeholder="https://example.com/face1.jpg")
                        with gr.Column():
                            m_target_2 = gr.Dropdown(label="🎯 Person to Replace with Face #2", choices=["Person #2 (Right)"], value="Person #2 (Right)")
                            m_f2 = gr.Image(label="👉 New Face #2", type="pil")
                            m_f2_url = gr.Textbox(label="OR URL #2", placeholder="https://example.com/face2.jpg")
                            
                    m_mode = gr.Radio(
                        ["⚡ Lightning Fast (3s)", "✨ Ultra 4K GFPGAN (Max Quality)"],
                        value="⚡ Lightning Fast (3s)",
                        label="Speed Mode"
                    )
                    m_btn = gr.Button("⚡ Swap Both Selected People", variant="primary")
                with gr.Column():
                    m_out = gr.Image(label="Swapped Group Result", type="pil")
                    m_status = gr.Textbox(label="Status", interactive=False)
                    
            m_detect_btn.click(
                fn=detect_all_group_persons,
                inputs=[m_tgt, m_tgt_url],
                outputs=[m_gallery, m_target_1, m_target_2, m_status]
            )
            m_btn.click(
                fn=two_face_swap_selected,
                inputs=[m_f1, m_f1_url, m_f2, m_f2_url, m_tgt, m_tgt_url, m_target_1, m_target_2, m_mode],
                outputs=[m_out, m_status]
            )

        # TAB 3: SINGLE-PERSON SELECTOR (3+ PEOPLE)
        with gr.TabItem("🎯 Swap 1 Specific Person in Large Group"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 1️⃣ Target Group Photo")
                    p_tgt = gr.Image(label="Upload Group Photo (Multiple People)", type="pil")
                    p_tgt_url = gr.Textbox(label="...OR Paste Group Image URL", placeholder="https://example.com/group.jpg")
                    p_detect_btn = gr.Button("🔍 Detect & List All Faces in Photo", variant="secondary")
                    
                    gr.Markdown("### 2️⃣ Detected People in Photo")
                    p_gallery = gr.Gallery(label="Detected Face Thumbnails", columns=4, height=140, object_fit="contain")
                    p_selector = gr.Radio(label="👉 Choose Person to Replace", choices=[])
                    
                    gr.Markdown("### 3️⃣ Your Replacement Face")
                    p_src = gr.Image(label="Upload New Face (To Insert)", type="pil")
                    p_src_url = gr.Textbox(label="...OR Paste New Face URL", placeholder="https://example.com/my_face.jpg")
                    
                    p_mode = gr.Radio(
                        ["⚡ Lightning Fast (3s)", "✨ Ultra 4K GFPGAN (Max Quality)"],
                        value="⚡ Lightning Fast (3s)",
                        label="Speed Mode"
                    )
                    p_btn = gr.Button("⚡ Swap Selected Person", variant="primary")
                with gr.Column():
                    p_out = gr.Image(label="Swapped Group Result", type="pil")
                    p_status = gr.Textbox(label="Detection & Swap Status", interactive=False)
                    
            p_detect_btn.click(
                fn=detect_and_crop_faces,
                inputs=[p_tgt, p_tgt_url],
                outputs=[p_gallery, p_selector, p_status]
            )
            p_btn.click(
                fn=swap_specific_person,
                inputs=[p_src, p_src_url, p_tgt, p_tgt_url, p_selector, p_mode],
                outputs=[p_out, p_status]
            )

if __name__ == "__main__":
    demo.launch(share=True)
