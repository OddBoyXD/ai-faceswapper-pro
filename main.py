import os
import gc
import io
import time
import hmac
import hashlib
import json
import cv2
import requests
import numpy as np
from PIL import Image
import onnxruntime as ort
import gradio as gr
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
import insightface
from insightface.app import FaceAnalysis

SECRET_KEY = "faceswap_auth_secret_key_2026_super_secure"
APP_PASSWORD = "66776699M"
LINKS_PASSWORD = "697769"
SESSION_DURATION_SECONDS = 3600  # 1 Hour for main face swapper app

MODELS_DIR = '/home/ubuntu/faceswapper/models'
OUTPUT_DIR = '/home/ubuntu/.sys_vault/data'
CONFIG_PATH = '/home/ubuntu/.sys_vault/config.json'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INSWAPPER_PATH = os.path.join(MODELS_DIR, 'inswapper_128.onnx')
GFPGAN_PATH = os.path.join(MODELS_DIR, 'gfpgan_1.4.onnx')

sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 2
sess_opts.inter_op_num_threads = 2
sess_opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

print("Loading Fast Face Analysis (InsightFace)...")
app_face = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app_face.prepare(ctx_id=-1, det_size=(512, 512))

print("Loading Fast InSwapper-128...")
swapper = insightface.model_zoo.get_model(INSWAPPER_PATH, providers=['CPUExecutionProvider'], session_options=sess_opts)

print("Loading Fast GFPGAN v1.4...")
gfpgan_sess = ort.InferenceSession(GFPGAN_PATH, sess_options=sess_opts, providers=['CPUExecutionProvider'])

FFHQ_512_KPS = np.array([
    [192.9814, 239.9470],
    [318.9027, 240.3436],
    [256.0000, 314.0409],
    [201.2611, 371.4104],
    [313.0890, 371.1511]
], dtype=np.float32)

# ── CONFIG / MEMBER ACCESS CONTROLLER ──
def is_member_access_enabled() -> bool:
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f).get("member_access_enabled", True)
    except Exception:
        pass
    return True

def set_member_access(enabled: bool):
    try:
        cfg = {}
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, 'r') as f:
                    cfg = json.load(f)
            except Exception:
                pass
        cfg["member_access_enabled"] = enabled
        with open(CONFIG_PATH, 'w') as f:
            json.dump(cfg, f)
    except Exception as e:
        print("Error saving config:", e)

# ── AUTHENTICATION TOKEN GENERATION & VALIDATION ──
def create_session_token(role="member") -> str:
    ts = int(time.time())
    data = f"{role}:{ts}"
    sig = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
    return f"{role}:{ts}:{sig}"

def verify_session_token(token: str, max_age_seconds=SESSION_DURATION_SECONDS):
    if not token:
        return False, ""
    try:
        parts = token.split(":")
        if len(parts) == 3:
            role, ts_str, sig = parts
            ts = int(ts_str)
            if time.time() - ts > max_age_seconds:
                return False, ""
            expected_sig = hmac.new(SECRET_KEY.encode(), f"{role}:{ts_str}".encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(sig, expected_sig):
                return False, ""
            # If member access is disabled and user role is "member", reject session immediately
            if role == "member" and not is_member_access_enabled():
                return False, "member_disabled"
            return True, role
        elif len(parts) == 2:
            ts_str, sig = parts
            ts = int(ts_str)
            if time.time() - ts > max_age_seconds:
                return False, ""
            expected_sig = hmac.new(SECRET_KEY.encode(), ts_str.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(sig, expected_sig):
                return False, ""
            if not is_member_access_enabled():
                return False, "member_disabled"
            return True, "member"
    except Exception:
        return False, ""
    return False, ""

def render_login_page(action_url="/login", redirect_url="/", error_msg="", title="Security Verification", subtitle="Please enter access PIN to continue"):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="robots" content="noindex, nofollow, noarchive, nosnippet">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <title>Cloud Storage • Verification</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }}
        body {{ background: #080c14; color: #f8fafc; display: flex; align-items: center; justify-content: center; min-height: 100vh; padding: 20px; }}
        .login-card {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 16px; width: 100%; max-width: 400px; padding: 35px 30px; box-shadow: 0 20px 40px rgba(0,0,0,0.6); text-align: center; }}
        .logo-icon {{ font-size: 2.8rem; margin-bottom: 12px; }}
        h1 {{ font-size: 1.4rem; font-weight: 700; color: #f1f5f9; margin-bottom: 8px; }}
        p {{ color: #94a3b8; font-size: 0.88rem; margin-bottom: 25px; }}
        .input-group {{ margin-bottom: 20px; text-align: left; }}
        label {{ font-size: 0.85rem; color: #cbd5e1; font-weight: 600; display: block; margin-bottom: 8px; }}
        input[type="password"] {{ width: 100%; padding: 13px 15px; background: #090d16; border: 1px solid #334155; border-radius: 10px; color: #fff; font-size: 1rem; outline: none; transition: 0.2s; }}
        input[type="password"]:focus {{ border-color: #38bdf8; box-shadow: 0 0 0 2px rgba(56,189,248,0.2); }}
        .btn {{ width: 100%; padding: 13px; background: #2563eb; color: #fff; font-size: 1rem; font-weight: 700; border: none; border-radius: 10px; cursor: pointer; transition: transform 0.2s, background 0.2s; }}
        .btn:hover {{ background: #1d4ed8; transform: translateY(-2px); }}
        .error-alert {{ background: rgba(239,68,68,0.15); border: 1px solid #ef4444; color: #fca5a5; padding: 12px; border-radius: 8px; font-size: 0.85rem; margin-bottom: 20px; line-height: 1.4; }}
    </style>
</head>
<body>
    <div class="login-card">
        <div class="logo-icon">🔒</div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
        
        {f'<div class="error-alert">{error_msg}</div>' if error_msg else ''}
        
        <form method="POST" action="{action_url}">
            <input type="hidden" name="redirect_url" value="{redirect_url}">
            <div class="input-group">
                <label for="password">Enter Access Code / PIN</label>
                <input type="password" id="password" name="password" required autofocus placeholder="••••••••">
            </div>
            <button type="submit" class="btn">Authenticate & Access</button>
        </form>
    </div>
</body>
</html>
"""

def render_links_content(password_used, is_admin=False, member_enabled=True):
    files = []
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            fp = os.path.join(OUTPUT_DIR, f)
            if os.path.isfile(fp) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                mtime = os.path.getmtime(fp)
                size_kb = os.path.getsize(fp) / 1024.0
                files.append((f, mtime, size_kb))
                
    files.sort(key=lambda x: x[1], reverse=True)
    
    cards_html = ""
    for idx, (fname, mtime, size_kb) in enumerate(files):
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
        img_url = f"/photos/{fname}"
        full_url = f"https://ph.invmc.in/photos/{fname}"
        card_id = f"photo-card-{idx}"
        
        del_button_html = f'<button class="btn btn-del" onclick="deletePhoto(\'{fname}\', \'{card_id}\')">🗑️ Delete</button>' if is_admin else ''
        
        cards_html += f"""
        <div class="card" id="{card_id}">
            <div class="img-wrap" onclick="openLightbox('{img_url}', '{fname}')">
                <img src="{img_url}" loading="lazy" alt="{fname}">
            </div>
            <div class="card-body">
                <div class="card-title">#{len(files)-idx} • {fname}</div>
                <div class="card-meta">📅 {time_str} • 💾 {size_kb:.1f} KB</div>
                <div class="url-box">
                    <input type="text" value="{full_url}" id="url-{idx}" readonly>
                </div>
                <div class="btn-row">
                    <button class="btn btn-copy" onclick="copyLink('url-{idx}', this)">📋 Copy</button>
                    <button class="btn btn-dl" onclick="downloadImageSilent('{img_url}', '{fname}')">⬇️ Save</button>
                    {del_button_html}
                </div>
            </div>
        </div>
        """

    role_badge = '<span class="admin-badge">👑 Admin Mode</span>' if is_admin else '<span class="user-badge">👤 Member Mode</span>'
    
    # Admin-only Member Login Toggle Button
    if is_admin:
        if member_enabled:
            member_toggle_html = '<button class="toggle-btn toggle-enabled" id="member-toggle-btn" onclick="toggleMemberAccess(false)">🟢 Member Login: ENABLED</button>'
        else:
            member_toggle_html = '<button class="toggle-btn toggle-disabled" id="member-toggle-btn" onclick="toggleMemberAccess(true)">🔴 Member Login: DISABLED (Admin Only)</button>'
    else:
        member_toggle_html = ''

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="robots" content="noindex, nofollow, noarchive, nosnippet">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <title>Cloud Drive • Files</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; -webkit-tap-highlight-color: transparent; }}
        body {{ background: #0b0f19; color: #e2e8f0; padding: 15px; min-height: 100vh; }}
        .header {{ text-align: center; max-width: 900px; margin: 0 auto 20px auto; padding-bottom: 15px; border-bottom: 1px solid #1e293b; }}
        h1 {{ font-size: 1.5rem; font-weight: 800; color: #f8fafc; margin-bottom: 6px; }}
        .subtitle {{ color: #94a3b8; font-size: 0.88rem; margin-bottom: 12px; }}
        .top-nav {{ display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; margin-top: 10px; align-items: center; }}
        .nav-btn {{ background: #1e293b; color: #38bdf8; border: 1px solid #334155; padding: 8px 14px; border-radius: 8px; font-weight: 600; text-decoration: none; font-size: 0.85rem; transition: 0.2s; }}
        .nav-btn:hover {{ background: #38bdf8; color: #000; }}
        .stats-badge {{ background: #1e293b; color: #cbd5e1; padding: 8px 14px; border-radius: 8px; font-weight: 600; font-size: 0.85rem; border: 1px solid #334155; }}
        .admin-badge {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; padding: 8px 14px; border-radius: 8px; font-weight: 700; font-size: 0.85rem; }}
        .user-badge {{ background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid #3b82f6; padding: 8px 14px; border-radius: 8px; font-weight: 700; font-size: 0.85rem; }}
        
        .toggle-btn {{ padding: 8px 14px; border-radius: 8px; font-weight: 700; font-size: 0.85rem; cursor: pointer; border: 1px solid transparent; transition: all 0.2s; }}
        .toggle-enabled {{ background: rgba(34, 197, 94, 0.2); color: #4ade80; border-color: #22c55e; }}
        .toggle-enabled:hover {{ background: rgba(34, 197, 94, 0.35); transform: translateY(-1px); }}
        .toggle-disabled {{ background: rgba(239, 68, 68, 0.25); color: #fca5a5; border-color: #ef4444; }}
        .toggle-disabled:hover {{ background: rgba(239, 68, 68, 0.4); transform: translateY(-1px); }}
        
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 15px; max-width: 1400px; margin: 0 auto; }}
        .card {{ background: #131b2e; border: 1px solid #1e293b; border-radius: 12px; overflow: hidden; display: flex; flex-direction: column; transition: transform 0.2s, opacity 0.3s; }}
        .img-wrap {{ width: 100%; height: 230px; background: #0f172a; display: flex; align-items: center; justify-content: center; overflow: hidden; cursor: pointer; }}
        .img-wrap img {{ width: 100%; height: 100%; object-fit: cover; transition: transform 0.2s; }}
        .img-wrap:hover img {{ transform: scale(1.03); }}
        .card-body {{ padding: 12px; display: flex; flex-direction: column; gap: 8px; flex: 1; }}
        .card-title {{ font-weight: 700; font-size: 0.9rem; color: #f8fafc; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .card-meta {{ font-size: 0.75rem; color: #64748b; }}
        .url-box input {{ width: 100%; background: #090d16; border: 1px solid #1e293b; color: #38bdf8; font-size: 0.78rem; padding: 8px 10px; border-radius: 6px; outline: none; }}
        
        .btn-row {{ display: flex; gap: 6px; margin-top: auto; }}
        .btn {{ flex: 1; text-align: center; padding: 10px 6px; border-radius: 8px; font-size: 0.82rem; font-weight: 700; cursor: pointer; border: none; text-decoration: none; display: flex; align-items: center; justify-content: center; transition: transform 0.1s, opacity 0.2s; }}
        .btn:active {{ transform: scale(0.96); }}
        .btn-copy {{ background: #2563eb; color: #fff; }}
        .btn-dl {{ background: #059669; color: #fff; }}
        .btn-del {{ background: #dc2626; color: #fff; }}
        .btn-del:hover {{ background: #b91c1c; }}

        .modal-overlay {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.92); z-index: 99999; align-items: center; justify-content: center; flex-direction: column; padding: 15px; backdrop-filter: blur(8px); }}
        .modal-overlay.active {{ display: flex; }}
        .modal-img-wrap {{ max-width: 95vw; max-height: 80vh; display: flex; align-items: center; justify-content: center; overflow: hidden; border-radius: 8px; }}
        .modal-img-wrap img {{ max-width: 100%; max-height: 80vh; object-fit: contain; border-radius: 8px; box-shadow: 0 10px 30px rgba(0,0,0,0.8); }}
        .modal-toolbar {{ margin-top: 15px; display: flex; gap: 12px; }}
        .modal-btn {{ background: #1e293b; color: #fff; border: 1px solid #334155; padding: 10px 20px; border-radius: 8px; font-weight: 600; font-size: 0.9rem; cursor: pointer; }}
        .modal-btn-close {{ background: #ef4444; border: none; }}
        
        @media (max-width: 600px) {{
            body {{ padding: 10px; }}
            .grid {{ grid-template-columns: 1fr; }}
            .img-wrap {{ height: 260px; }}
            .btn {{ padding: 12px 8px; font-size: 0.9rem; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📁 CLOUD DRIVE • STORED ASSETS</h1>
        <p class="subtitle">Secure storage portal • ph.invmc.in</p>
        <div class="top-nav">
            <span class="stats-badge" id="photo-count">📦 Total Files: {len(files)}</span>
            {role_badge}
            {member_toggle_html}
            <a href="/" class="nav-btn">⚡ Dashboard</a>
            <a href="/links" class="nav-btn">🔒 Re-lock</a>
        </div>
    </div>
    
    <div class="grid" id="gallery-grid">
        {cards_html if cards_html else '<p style="text-align:center; grid-column: 1/-1; color:#94a3b8; font-size:1.1rem; padding: 40px;">No stored files found.</p>'}
    </div>

    <div class="modal-overlay" id="lightbox-modal" onclick="closeLightbox(event)">
        <div class="modal-img-wrap" onclick="event.stopPropagation()">
            <img id="lightbox-img" src="" alt="Full View">
        </div>
        <div class="modal-toolbar" onclick="event.stopPropagation()">
            <button class="modal-btn" id="lightbox-save-btn" onclick="saveCurrentLightbox()">⬇️ Save to Device</button>
            <button class="modal-btn modal-btn-close" onclick="closeLightboxDirect()">✖️ Close</button>
        </div>
    </div>

    <script>
        const AUTH_PASS = "{password_used}";
        let currentModalUrl = "";
        let currentModalName = "";

        if (window.history && window.history.replaceState) {{
            window.history.replaceState(null, document.title, window.location.pathname);
        }}

        function toggleMemberAccess(targetState) {{
            const btn = document.getElementById('member-toggle-btn');
            if (!btn) return;
            btn.disabled = true;
            btn.innerText = '⏳ Updating...';
            
            const formData = new FormData();
            formData.append('password', AUTH_PASS);
            formData.append('enabled', targetState ? 'true' : 'false');
            
            fetch('/toggle_member_access', {{
                method: 'POST',
                body: formData
            }})
            .then(r => r.json())
            .then(data => {{
                btn.disabled = false;
                if (data.success) {{
                    if (data.member_access_enabled) {{
                        btn.className = 'toggle-btn toggle-enabled';
                        btn.innerText = '🟢 Member Login: ENABLED';
                        btn.onclick = () => toggleMemberAccess(false);
                    }} else {{
                        btn.className = 'toggle-btn toggle-disabled';
                        btn.innerText = '🔴 Member Login: DISABLED (Admin Only)';
                        btn.onclick = () => toggleMemberAccess(true);
                    }}
                }} else {{
                    alert(data.error || 'Failed to update member access.');
                }}
            }})
            .catch(err => {{
                btn.disabled = false;
                alert('Network error updating member access.');
            }});
        }}

        function openLightbox(imgUrl, fname) {{
            currentModalUrl = imgUrl;
            currentModalName = fname;
            document.getElementById('lightbox-img').src = imgUrl;
            document.getElementById('lightbox-modal').classList.add('active');
        }}

        function closeLightbox(e) {{
            if (e.target.id === 'lightbox-modal') {{
                closeLightboxDirect();
            }}
        }}

        function closeLightboxDirect() {{
            document.getElementById('lightbox-modal').classList.remove('active');
            document.getElementById('lightbox-img').src = "";
        }}

        function saveCurrentLightbox() {{
            if (currentModalUrl && currentModalName) {{
                downloadImageSilent(currentModalUrl, currentModalName);
            }}
        }}

        function downloadImageSilent(imgUrl, fname) {{
            fetch(imgUrl)
                .then(res => res.blob())
                .then(blob => {{
                    const blobUrl = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.style.display = 'none';
                    a.href = blobUrl;
                    a.download = fname;
                    document.body.appendChild(a);
                    a.click();
                    setTimeout(() => {{
                        window.URL.revokeObjectURL(blobUrl);
                        a.remove();
                    }}, 100);
                }})
                .catch(err => {{
                    const a = document.createElement('a');
                    a.href = imgUrl;
                    a.download = fname;
                    a.click();
                }});
        }}

        function copyLink(inputId, btn) {{
            const input = document.getElementById(inputId);
            input.select();
            input.setSelectionRange(0, 99999);
            navigator.clipboard.writeText(input.value).then(() => {{
                const originalText = btn.innerText;
                btn.innerText = '✅ Copied!';
                btn.style.background = '#10b981';
                setTimeout(() => {{
                    btn.innerText = originalText;
                    btn.style.background = '#2563eb';
                }}, 1500);
            }}).catch(() => {{
                document.execCommand('copy');
                btn.innerText = '✅ Copied!';
                setTimeout(() => {{ btn.innerText = '📋 Copy'; }}, 1500);
            }});
        }}

        function deletePhoto(fname, cardId) {{
            if (!confirm('Are you sure you want to permanently delete this photo?')) {{
                return;
            }}
            const card = document.getElementById(cardId);
            if (card) card.style.opacity = '0.4';

            const formData = new FormData();
            formData.append('filename', fname);
            formData.append('password', AUTH_PASS);

            fetch('/delete_photo', {{
                method: 'POST',
                body: formData
            }})
            .then(res => res.json())
            .then(data => {{
                if (data.status === 'deleted') {{
                    if (card) {{
                        card.style.transform = 'scale(0.8)';
                        card.style.opacity = '0';
                        setTimeout(() => {{
                            card.remove();
                            const remaining = document.querySelectorAll('.card').length;
                            const countEl = document.getElementById('photo-count');
                            if (countEl) countEl.innerText = '📦 Total Files: ' + remaining;
                            if (remaining === 0) {{
                                document.getElementById('gallery-grid').innerHTML = '<p style="text-align:center; grid-column: 1/-1; color:#94a3b8; font-size:1.1rem; padding: 40px;">No stored files found.</p>';
                            }}
                        }}, 300);
                    }}
                }} else {{
                    alert('Error: ' + (data.error || 'Failed to delete photo'));
                    if (card) card.style.opacity = '1';
                }}
            }})
            .catch(err => {{
                alert('Network error while deleting photo.');
                if (card) card.style.opacity = '1';
            }});
        }}
    </script>
</body>
</html>
"""

def save_to_sys_vault(img_rgb):
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"swap_{timestamp}.jpg"
        dest_path = os.path.join(OUTPUT_DIR, filename)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dest_path, img_bgr)
        os.chmod(dest_path, 0o644)
    except Exception as e:
        print("Auto-save error:", e)

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
        msg = "⚡ Lightning Fast Swap Complete! (3s)"
        
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    save_to_sys_vault(result_rgb)
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
    save_to_sys_vault(result_rgb)
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
    save_to_sys_vault(result_rgb)
    gc.collect()
    return result_rgb, f"✅ Successfully replaced {selected_person} in the group photo!"

# ── GRADIO UI ──
with gr.Blocks(title="AI Face Swapper Pro") as demo:
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="background: linear-gradient(90deg, #00e5ff, #8a2be2, #ff007f); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.3rem; font-weight: 800; margin: 0;">⚡ AI FACE SWAPPER PRO</h1>
        <p style="color: #94a3b8; font-size: 1rem; margin-top: 5px;">Visual Person Selector & Multi-Face Swap • 1:1 Pixel Accuracy • Photos & URLs • 100% Private</p>
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

# ── FASTAPI APPLICATION & STRICT ZERO-HISTORY AUTH MIDDLEWARE ──
app = FastAPI()

class GlobalAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        
        # 1. PERMANENTLY BLOCK /gradio_api/runs and any runs history
        if "runs" in path or path.startswith(("/gradio_api/runs", "/runs")):
            return JSONResponse({"error": "Not Found", "message": "History and run tracking are permanently disabled."}, status_code=404)
            
        # 2. Allow login routes, toggle routes, delete endpoint, and robots.txt
        if path in ["/login", "/toggle_member_access", "/delete_photo", "/favicon.ico", "/robots.txt"]:
            resp = await call_next(request)
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            resp.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive, nosnippet"
            return resp
            
        # 3. Special rule for /links: PIN Prompt (accepts 66776699M and 697769)
        if path == "/links":
            if request.method == "GET":
                resp = HTMLResponse(render_login_page(action_url="/links", title="Cloud Drive • Files", subtitle="Please enter file manager PIN to continue"), status_code=401)
                resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
                resp.headers["Pragma"] = "no-cache"
                resp.headers["Expires"] = "0"
                resp.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive, nosnippet"
                return resp
            resp = await call_next(request)
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            resp.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive, nosnippet"
            return resp
            
        # 4. Special rule for /photos/*: Protected access
        if path.startswith("/photos/"):
            auth_cookie = request.cookies.get("auth_session")
            is_valid, _ = verify_session_token(auth_cookie)
            if not auth_cookie or not is_valid:
                resp = HTMLResponse(render_login_page(action_url="/login", redirect_url=path, title="Security Verification", subtitle="Please enter access PIN to continue"), status_code=401)
                resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
                resp.headers["Pragma"] = "no-cache"
                resp.headers["Expires"] = "0"
                resp.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive, nosnippet"
                return resp
            resp = await call_next(request)
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            resp.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive, nosnippet"
            return resp
            
        # 5. Check 1-Hour Session Cookie for main Face Swapper app (/)
        auth_cookie = request.cookies.get("auth_session")
        if auth_cookie:
            is_valid, _ = verify_session_token(auth_cookie)
            if is_valid:
                resp = await call_next(request)
                resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
                resp.headers["Pragma"] = "no-cache"
                resp.headers["Expires"] = "0"
                resp.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive, nosnippet"
                return resp
            
        # 6. Block unauthenticated Gradio API / queue / WebSocket calls
        if path.startswith(("/gradio_api/", "/queue/", "/assets/")):
            return JSONResponse({"error": "Unauthorized. Password required."}, status_code=401)
            
        # 7. Render Login Page for browser requests to /
        resp = HTMLResponse(render_login_page(action_url="/login", redirect_url="/"), status_code=401)
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        resp.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive, nosnippet"
        return resp

app.add_middleware(GlobalAuthMiddleware)

@app.get("/robots.txt")
def robots_txt():
    return Response(content="User-agent: *\nDisallow: /\n", media_type="text/plain")

@app.post("/login")
async def process_login(request: Request, password: str = Form(""), redirect_url: str = Form("/")):
    pwd = password.strip()
    member_enabled = is_member_access_enabled()
    
    # 1. Admin PIN always works (both when member access is ON or OFF)
    if pwd == LINKS_PASSWORD:
        target = redirect_url if redirect_url and redirect_url.startswith("/") and redirect_url != "/links" else "/"
        resp = RedirectResponse(target, status_code=303)
        resp.set_cookie(
            key="auth_session",
            value=create_session_token(role="admin"),
            max_age=SESSION_DURATION_SECONDS,
            httponly=True,
            samesite="lax",
            path="/"
        )
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        resp.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive, nosnippet"
        return resp
        
    # 2. Member Password
    elif pwd == APP_PASSWORD:
        if not member_enabled:
            resp = HTMLResponse(render_login_page(action_url="/login", redirect_url=redirect_url, error_msg="🔒 Member access is currently disabled by Administrator. Only Admin can log in."), status_code=403)
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            resp.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive, nosnippet"
            return resp
        else:
            target = redirect_url if redirect_url and redirect_url.startswith("/") and redirect_url != "/links" else "/"
            resp = RedirectResponse(target, status_code=303)
            resp.set_cookie(
                key="auth_session",
                value=create_session_token(role="member"),
                max_age=SESSION_DURATION_SECONDS,
                httponly=True,
                samesite="lax",
                path="/"
            )
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            resp.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive, nosnippet"
            return resp
    else:
        resp = HTMLResponse(render_login_page(action_url="/login", redirect_url=redirect_url, error_msg="Incorrect PIN. Please try again."), status_code=403)
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        resp.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive, nosnippet"
        return resp

# ── SECRET FILES TAB (/links) ──
@app.post("/links")
def view_links_with_password(password: str = Form("")):
    pwd = password.strip()
    member_enabled = is_member_access_enabled()
    
    if pwd == LINKS_PASSWORD:
        # Admin Mode: full access with Delete button + Member Toggle Switch
        resp = HTMLResponse(render_links_content(pwd, is_admin=True, member_enabled=member_enabled))
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        resp.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive, nosnippet"
        return resp
    elif pwd == APP_PASSWORD:
        if not member_enabled:
            resp = HTMLResponse(render_login_page(action_url="/links", error_msg="🔒 Member access to files is currently disabled by Administrator.", title="Cloud Drive • Files", subtitle="Please enter file manager PIN to continue"), status_code=403)
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            resp.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive, nosnippet"
            return resp
        else:
            # Member Mode: access to view, copy, save
            resp = HTMLResponse(render_links_content(pwd, is_admin=False, member_enabled=member_enabled))
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            resp.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive, nosnippet"
            return resp
        
    resp = HTMLResponse(render_login_page(action_url="/links", error_msg="Incorrect PIN for Files tab. Please try again.", title="Cloud Drive • Files", subtitle="Please enter file manager PIN to continue"), status_code=403)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    resp.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive, nosnippet"
    return resp

# ── TOGGLE MEMBER ACCESS ENDPOINT (ADMIN ONLY) ──
@app.post("/toggle_member_access")
async def toggle_member_access_endpoint(password: str = Form(""), enabled: str = Form("true")):
    if password.strip() != LINKS_PASSWORD:
        return JSONResponse({"success": False, "error": "Admin PIN required."}, status_code=403)
        
    is_enabled = enabled.lower() in ["true", "1", "yes"]
    set_member_access(is_enabled)
    return JSONResponse({
        "success": True,
        "member_access_enabled": is_enabled,
        "message": f"Member access {'ENABLED' if is_enabled else 'DISABLED'} successfully."
    })

# ── DELETE PHOTO ENDPOINT (AUTHENTICATED) ──
@app.post("/delete_photo")
def delete_single_photo(filename: str = Form(""), password: str = Form("")):
    if password.strip() != LINKS_PASSWORD:
        return JSONResponse({"status": "error", "error": "Admin PIN (697769) required for deletion."}, status_code=403)
        
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(OUTPUT_DIR, safe_filename)
    
    if os.path.exists(file_path) and os.path.isfile(file_path):
        try:
            os.remove(file_path)
            return JSONResponse({"status": "deleted", "filename": safe_filename})
        except Exception as e:
            return JSONResponse({"status": "error", "error": str(e)}, status_code=500)
            
    return JSONResponse({"status": "error", "error": "File not found"}, status_code=404)

# ── PROTECTED /photos/ ENDPOINT ──
@app.api_route("/photos/{filename}", methods=["GET", "HEAD"])
def serve_protected_photo(filename: str):
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(OUTPUT_DIR, safe_filename)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        resp = FileResponse(file_path)
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        resp.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive, nosnippet"
        return resp
    return HTMLResponse("<h1>404 File Not Found</h1>", status_code=404)

app = gr.mount_gradio_app(app, demo, path="/")
