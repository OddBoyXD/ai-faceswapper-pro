# ⚡ AI Face Swapper Pro & Ultra-Realistic AI Studio • Complete A-to-Z Guide

An enterprise-grade, high-accuracy AI Face Swapper & Ultra-Realistic Creation Suite powered by **InsightFace (InSwapper-128)**, **GFPGAN v1.4 (Ultra HD)**, **RealVisXL V4.0 (4K Photorealism)**, **InstructPix2Pix (Chatbot Image Editor)**, **InstantID SDXL**, and **FastAPI / Gradio**. Features visual multi-person selector, zero-browser-history stealth camouflage, role-based dual encrypted vaults, and dynamic admin access controls.

---

## 📋 Table of Contents
1. [🌟 Features & Capabilities](#-features--capabilities)
2. [🔑 Access Passwords & Roles](#-access-passwords--roles)
3. [🖥️ Option 1: Full A-to-Z Installation on Any Linux VPS (Ubuntu / Debian)](#️-option-1-full-a-to-z-installation-on-any-linux-vps-ubuntu--debian)
4. [🎨 Option 2: Ultra-Realistic 4K Generator & Chat Image Editor on Google Colab](#-option-2-ultra-realistic-4k-generator--chat-image-editor-on-google-colab)
5. [🌟 Option 3: Hollywood-Grade InstantID (SDXL 1024x1024) on Google Colab](#-option-3-hollywood-grade-instantid-sdxl-1024x1024-on-google-colab)
6. [⚡ Option 4: Fast GPU Face Swap (InSwapper + GFPGAN) on Google Colab](#-option-4-fast-gpu-face-swap-inswapper--gfpgan-on-google-colab)
7. [🚀 Option 5: Free GPU on Kaggle (30h Free GPU Weekly)](#-option-5-free-gpu-on-kaggle-30h-free-gpu-weekly)
8. [☁️ Option 6: 100% Free 24/7 Hosting on Hugging Face Spaces](#️-option-6-100-free-247-hosting-on-hugging-face-spaces-16-gb-ram)
9. [💻 Option 7: Local Setup on Windows / Mac PC](#-option-7-local-setup-on-windows--mac-pc)
10. [📁 Project Architecture & File Structure](#-project-architecture--file-structure)
11. [🛠️ Useful Maintenance Commands](#️-useful-maintenance-commands)

---

## 🌟 Features & Capabilities

* **🎨 4K Photorealistic Text-to-Image (RealVisXL V4.0 • Colab GPU)**: Authentic skin pores, natural eye reflections, DSLR depth-of-field, studio lighting simulation, and 100% uncensored generation.
* **💬 Conversational Chat Image Editor (InstructPix2Pix • Colab GPU)**: Upload any photo and simply type your edits in plain English (e.g. *"make him wear a leather jacket"*, *"change hair to blonde"*, *"put on a beach at sunset"*).
* **🌟 InstantID SDXL 1024x1024 (Colab GPU)**: Native diffusion-based face synthesis with 10/10 Hollywood photorealism.
* **👤 1:1 Single Face Swap**: Instant sub-second swapping using InSwapper-128 + Ultra HD facial restoration with GFPGAN.
* **👥 2-Person Custom Swap**: Automatic face detection with thumbnail preview cards, allowing custom mapping of who gets which replacement face.
* **🎯 Group Person Selector (3+ People)**: Crop preview selector to pick exactly 1 person to swap out of a large group.
* **👑 Dynamic Member Access Control**: Admin can toggle member login ON/OFF live from `/links` with instant session invalidation.
* **🕵️ Zero-History Stealth Camouflage**: Disguised titles (`Cloud Storage`), `Cache-Control: no-store`, and unindexed headers.
* **📁 Hidden System Vault**: Processed photos auto-saved inside a hidden directory (`~/.sys_vault/data/`).

---

## 🔑 Access Passwords & Roles

| Location | Password / PIN | Role | Permissions |
| :--- | :--- | :--- | :--- |
| **Main App (`/`)** | `66776699M` | **Member** | Full access to Face Swapper (1-Hour session cookie). |
| **Main App (`/`)** | `697769` | 👑 **Admin** | Full access (Works even when Member access is disabled). |
| **Files Vault (`/links`)** | `697769` | 👑 **Admin** | View all photos, Copy Link, Save Image, 🗑️ **Delete Photo**, and **🟢/🔴 Member Login Toggle**. |
| **Files Vault (`/links`)** | `66776699M` | 👤 **Member** | View all photos, Copy Link, and Save Image *(Delete & Toggle hidden)*. |

---

## 🖥️ Option 1: Full A-to-Z Installation on Any Linux VPS (Ubuntu / Debian)

Follow these exact step-by-step commands on a fresh server:

### Step 1: Update System & Install Required Packages
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git ffmpeg libsm6 libxext6 libgl1 nodejs npm nginx ufw
```

### Step 2: Install Node.js Process Manager (PM2)
```bash
sudo npm install -g pm2
```

### Step 3: Clone Your Private Repository (with Pre-authenticated API Token)
```bash
cd /home/ubuntu   # or your home directory
git clone https://OddBoyXdxd69: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@github.com/OddBoyXdxd69/ai-faceswapper-pro.git faceswapper
cd faceswapper
```

### Step 4: Create Virtual Environment & Install Python Packages
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Download AI Models Automatically
```bash
python download_models.py
```

### Step 6: Start 24/7 Background Process with PM2
```bash
pm2 start "venv/bin/uvicorn main:app --host 127.0.0.1 --port 7860" --name faceswapper
pm2 save
pm2 startup
```

### Step 7: Configure Nginx Reverse Proxy (for Port 80 / 443 & Domain)
Create Nginx configuration:
```bash
sudo nano /etc/nginx/sites-available/faceswapper
```
Paste the following configuration (replace `yourdomain.com` with your domain):
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    client_max_body_size 100M;

    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
}
```
Enable the site and restart Nginx:
```bash
sudo ln -sf /etc/nginx/sites-available/faceswapper /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## 🎨 Option 2: Ultra-Realistic 4K Generator & Chat Image Editor on Google Colab

> [!TIP]
> **Features:**  
> 1. **4K Photorealistic Generator (RealVisXL V4.0):** Generates lifelike humans, portraits, and scenes with DSLR lighting.  
> 2. **AI Chat Image Editor (InstructPix2Pix):** Upload any image and tell the AI what to change in conversational chat!

Paste this into a **Google Colab** code cell:

```python
# 1. Step out and clone fresh repository with Auth Token
%cd /content
!rm -rf /content/faceswapper
!git clone https://OddBoyXdxd69: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@github.com/OddBoyXdxd69/ai-faceswapper-pro.git /content/faceswapper
%cd /content/faceswapper

# 2. Install AI Studio Dependencies
!pip install -q diffusers transformers accelerate gradio opencv-python-headless pillow invisible-watermark

# 3. Launch Ultra-Realistic AI Studio with Free Share Link
!python colab_ai_studio.py
```

---

## 🌟 Option 3: Hollywood-Grade InstantID (SDXL 1024x1024) on Google Colab

```python
# 1. Step out and clone repository with Token
%cd /content
!rm -rf /content/faceswapper
!git clone https://OddBoyXdxd69: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@github.com/OddBoyXdxd69/ai-faceswapper-pro.git /content/faceswapper
%cd /content/faceswapper

# 2. Install InstantID Dependencies
!pip install -q diffusers transformers accelerate insightface gradio opencv-python-headless pillow requests

# 3. Launch InstantID 4K Web App
!python instantid_colab.py
```

---

## ⚡ Option 4: Fast GPU Face Swap (InSwapper + GFPGAN) on Google Colab

```python
# 1. Step out and clone repository with Token
%cd /content
!rm -rf /content/faceswapper
!git clone https://OddBoyXdxd69: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@github.com/OddBoyXdxd69/ai-faceswapper-pro.git /content/faceswapper
%cd /content/faceswapper

# 2. Install Dependencies
!pip install -q fastapi uvicorn gradio insightface onnxruntime-gpu opencv-python-headless pillow requests

# 3. Download AI Models
!python download_models.py

# 4. Launch with Free Share Link
import sys
if 'app' in sys.modules:
    del sys.modules['app']
import app
app.demo.launch(share=True)
```

---

## 🚀 Option 5: Free GPU on Kaggle (30h Free GPU Weekly)

> [!IMPORTANT]
> In Kaggle sidebar: Turn **Internet ON** and set **Accelerator to GPU T4 x2**.

```python
# 1. Clone Private Repository to Kaggle Working Directory
!git clone https://OddBoyXdxd69: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@github.com/OddBoyXdxd69/ai-faceswapper-pro.git /kaggle/working/faceswapper
%cd /kaggle/working/faceswapper

# 2. Install Dependencies
!pip install -q fastapi uvicorn gradio insightface onnxruntime-gpu opencv-python-headless pillow requests

# 3. Download AI Models
!python download_models.py

# 4. Launch with Free Public Gradio Share Link
import app
app.demo.launch(share=True)
```

---

## ☁️ Option 6: 100% Free 24/7 Hosting on Hugging Face Spaces (16 GB RAM)

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. **Space Name**: `ai-faceswapper`
3. **SDK**: Select **Gradio**.
4. **Hardware**: Choose **Free (2 vCPU • 16 GB RAM • 50GB Disk)**.
5. Connect your GitHub repository `https://OddBoyXdxd69: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@github.com/OddBoyXdxd69/ai-faceswapper-pro.git`.

---

## 💻 Option 7: Local Setup on Windows / Mac PC

### Windows (PowerShell):
```powershell
git clone https://OddBoyXdxd69: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@github.com/OddBoyXdxd69/ai-faceswapper-pro.git faceswapper
cd faceswapper
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python download_models.py
python main.py
```
Open browser at: `http://localhost:7860`

### macOS / Linux:
```bash
git clone https://OddBoyXdxd69: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@github.com/OddBoyXdxd69/ai-faceswapper-pro.git faceswapper
cd faceswapper
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 download_models.py
python3 main.py
```
Open browser at: `http://localhost:7860`

---

## 📁 Project Architecture & File Structure

```
ai-faceswapper-pro/
├── main.py                   # Main FastAPI & Gradio Server for Linux VPS (Dual Auth + Stealth)
├── colab_ai_studio.py        # 4K Photorealistic Generator & Conversational Image Editor Chatbot
├── ai_generator_colab.ipynb  # 1-Click AI Studio Colab Notebook
├── instantid_colab.py        # Hollywood-Grade 1024x1024 InstantID SDXL engine (Google Colab GPU)
├── instantid_colab.ipynb     # 1-Click InstantID Colab Notebook
├── app.py                    # Fast InSwapper + GFPGAN runner
├── ai_faceswapper_colab.ipynb# 1-Click Classic Fast Face Swap Colab Notebook
├── download_models.py        # Automated downloader for InSwapper & GFPGAN ONNX models
├── run_server.py             # Process orchestrator script
├── requirements.txt          # Complete Python dependencies
├── .gitignore                # Ignores large binaries, caches, and secret data
└── README.md                 # Complete Documentation
```

---

## 🛠️ Useful Maintenance Commands

```bash
# Check running service status
pm2 status

# View live application logs
pm2 logs faceswapper

# Restart server after changes
pm2 restart faceswapper

# Check memory and swap usage
free -h

# Check storage space
df -h /

# List stored files inside hidden vault
ls -lah /home/ubuntu/.sys_vault/data/
```
