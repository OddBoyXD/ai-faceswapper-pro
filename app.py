import os
import cv2
import numpy as np
import gradio as gr
import insightface
from insightface.app import FaceAnalysis

MODELS_DIR = '/home/ubuntu/faceswapper/models'
INSWAPPER_PATH = os.path.join(MODELS_DIR, 'inswapper_128.onnx')

print("Loading Face Analysis (buffalo_l)...")
app_face = FaceAnalysis(name='buffalo_l')
app_face.prepare(ctx_id=-1, det_size=(640, 640)) # CPU inference

print("Loading InSwapper Model...")
swapper = insightface.model_zoo.get_model(INSWAPPER_PATH, download=False, download_zip=False)

def enhance_face_detail(img, face_box):
    # Gentle unsharp mask for 4K crisp clarity
    x1, y1, x2, y2 = [int(v) for v in face_box]
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 > x1 and y2 > y1:
        face_roi = img[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(face_roi, (0, 0), 1.5)
        sharpened = cv2.addWeighted(face_roi, 1.4, blurred, -0.4, 0)
        img[y1:y2, x1:x2] = sharpened
    return img

def single_face_swap(source_img, target_img, enhance):
    if source_img is None or target_img is None:
        return None, "Please upload both Source Face and Target Image."
    
    source_bgr = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
    target_bgr = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    
    source_faces = app_face.get(source_bgr)
    if not source_faces:
        return None, "❌ No face detected in Source Image. Please upload a clear face photo."
    
    target_faces = app_face.get(target_bgr)
    if not target_faces:
        return None, "❌ No face detected in Target Image."
    
    source_face = max(source_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    
    result = target_bgr.copy()
    target_face = max(target_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    result = swapper.get(result, target_face, source_face, paste_back=True)
    
    if enhance:
        result = enhance_face_detail(result, target_face.bbox)
        
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result_rgb, f"✅ Successfully swapped face! ({result_rgb.shape[1]}x{result_rgb.shape[0]})"

def two_face_swap(face1_img, face2_img, target_img, swap_order, enhance):
    if target_img is None:
        return None, "Please upload the Target Image."
    if face1_img is None and face2_img is None:
        return None, "Please upload at least one replacement face."
        
    target_bgr = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    target_faces = app_face.get(target_bgr)
    
    if len(target_faces) < 2:
        return None, f"❌ Detected only {len(target_faces)} face in the target image. Need at least 2 faces."
        
    # Sort target faces from left to right (X coordinate)
    target_faces = sorted(target_faces, key=lambda f: f.bbox[0])
    
    if swap_order == "Face 1 ➔ Left Person, Face 2 ➔ Right Person":
        tf1, tf2 = target_faces[0], target_faces[1]
    else:
        tf1, tf2 = target_faces[1], target_faces[0]
        
    result = target_bgr.copy()
    swapped_count = 0
    
    if face1_img is not None:
        f1_bgr = cv2.cvtColor(np.array(face1_img), cv2.COLOR_RGB2BGR)
        f1_faces = app_face.get(f1_bgr)
        if f1_faces:
            sf1 = max(f1_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            result = swapper.get(result, tf1, sf1, paste_back=True)
            if enhance:
                result = enhance_face_detail(result, tf1.bbox)
            swapped_count += 1
            
    if face2_img is not None:
        f2_bgr = cv2.cvtColor(np.array(face2_img), cv2.COLOR_RGB2BGR)
        f2_faces = app_face.get(f2_bgr)
        if f2_faces:
            sf2 = max(f2_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            result = swapper.get(result, tf2, sf2, paste_back=True)
            if enhance:
                result = enhance_face_detail(result, tf2.bbox)
            swapped_count += 1
            
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result_rgb, f"✅ Successfully swapped {swapped_count} faces simultaneously!"

def swap_all_faces(source_img, target_img, enhance):
    if source_img is None or target_img is None:
        return None, "Please upload both Source and Target images."
    source_bgr = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
    target_bgr = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    
    source_faces = app_face.get(source_bgr)
    if not source_faces:
        return None, "❌ No face detected in Source Image."
    target_faces = app_face.get(target_bgr)
    if not target_faces:
        return None, "❌ No faces detected in Target Image."
        
    sf = max(source_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    result = target_bgr.copy()
    
    for tf in target_faces:
        result = swapper.get(result, tf, sf, paste_back=True)
        if enhance:
            result = enhance_face_detail(result, tf.bbox)
            
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result_rgb, f"✅ Swapped all {len(target_faces)} faces in the photo!"

with gr.Blocks(title="AI Face Swapper Pro") as demo:
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="background: linear-gradient(90deg, #00e5ff, #8a2be2, #ff007f); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.2rem; font-weight: 800; margin: 0;">⚡ AI FACE SWAPPER PRO</h1>
        <p style="color: #94a3b8; font-size: 1rem; margin-top: 5px;">High-Accuracy 1:1 Face Replacement & Multi-Face Swap • Zero Watermark • 100% Free</p>
    </div>
    """)
    
    with gr.Tabs():
        with gr.TabItem("👤 Single Face Swap"):
            with gr.Row():
                with gr.Column():
                    s_src = gr.Image(label="1. Source Face (Your Photo)", type="pil")
                    s_tgt = gr.Image(label="2. Target Body / Image", type="pil")
                    s_enh = gr.Checkbox(label="✨ HD Face Enhancement & Clarity", value=True)
                    s_btn = gr.Button("⚡ Swap Face Now", variant="primary")
                with gr.Column():
                    s_out = gr.Image(label="Swapped Result", type="pil")
                    s_status = gr.Textbox(label="Status", interactive=False)
                    
            s_btn.click(fn=single_face_swap, inputs=[s_src, s_tgt, s_enh], outputs=[s_out, s_status])
            
        with gr.TabItem("👥 2-Face Group Swap"):
            with gr.Row():
                with gr.Column():
                    m_tgt = gr.Image(label="1. Main Photo (With 2 People)", type="pil")
                    with gr.Row():
                        m_f1 = gr.Image(label="Face #1 (Replacement)", type="pil")
                        m_f2 = gr.Image(label="Face #2 (Replacement)", type="pil")
                    m_order = gr.Radio(
                        ["Face 1 ➔ Left Person, Face 2 ➔ Right Person", "Face 1 ➔ Right Person, Face 2 ➔ Left Person"],
                        value="Face 1 ➔ Left Person, Face 2 ➔ Right Person",
                        label="Mapping Position"
                    )
                    m_enh = gr.Checkbox(label="✨ HD Face Enhancement", value=True)
                    m_btn = gr.Button("⚡ Swap Both Faces", variant="primary")
                with gr.Column():
                    m_out = gr.Image(label="Swapped Group Result", type="pil")
                    m_status = gr.Textbox(label="Status", interactive=False)
                    
            m_btn.click(fn=two_face_swap, inputs=[m_f1, m_f2, m_tgt, m_order, m_enh], outputs=[m_out, m_status])

        with gr.TabItem("🎭 Swap All Faces with One Person"):
            with gr.Row():
                with gr.Column():
                    a_src = gr.Image(label="Source Face", type="pil")
                    a_tgt = gr.Image(label="Group Photo", type="pil")
                    a_enh = gr.Checkbox(label="✨ HD Face Enhancement", value=True)
                    a_btn = gr.Button("⚡ Swap Every Face", variant="primary")
                with gr.Column():
                    a_out = gr.Image(label="Result", type="pil")
                    a_status = gr.Textbox(label="Status", interactive=False)
                    
            a_btn.click(fn=swap_all_faces, inputs=[a_src, a_tgt, a_enh], outputs=[a_out, a_status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

import time
while True:
    time.sleep(3600)
