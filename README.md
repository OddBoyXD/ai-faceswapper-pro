# ⚡ AI Face Swapper Pro & Uncensored AI Creation Suite • Complete Guide

An enterprise-grade, high-accuracy AI Face Swapper & Uncensored Creation Suite powered by **InsightFace (InSwapper-128)**, **GFPGAN v1.4 (Ultra HD)**, **Black Forest Labs (FLUX.1 [dev] Max 12B)**, **Stable Diffusion XL (SDXL Base 1.0)**, and **FastAPI / Gradio**. Features visual multi-person selector, zero-browser-history stealth camouflage, role-based dual encrypted vaults, and dynamic admin access controls.

---

## 📋 Table of Contents
1. [🌟 Features & Capabilities](#-features--capabilities)
2. [🔑 Access Passwords & Roles](#-access-passwords--roles)
3. [🖥️ Option 1: Full 1-Click Setup on Any Linux VPS (Ubuntu / Debian)](#️-option-1-full-1-click-setup-on-any-linux-vps-ubuntu--debian)
4. [🎨 Option 2: Uncensored SDXL Studio (Image-to-Image & 4K Generator) on Google Colab](#-option-2-uncensored-sdxl-studio-image-to-image--4k-generator-on-google-colab)
5. [🌟 Option 3: FLUX.1 [dev] Max Version (Flagship 12B Flow Transformer) on Google Colab](#-option-3-flux1-dev-max-version-flagship-12b-flow-transformer-on-google-colab)
6. [⚡ Option 4: Fast GPU Face Swap (InSwapper + GFPGAN) on Google Colab](#-option-4-fast-gpu-face-swap-inswapper--gfpgan-on-google-colab)
7. [🚀 Option 5: Free GPU on Kaggle (30h Free GPU Weekly)](#-option-5-free-gpu-on-kaggle-30h-free-gpu-weekly)
8. [☁️ Option 6: 100% Free 24/7 Hosting on Hugging Face Spaces](#️-option-6-100-free-247-hosting-on-hugging-face-spaces-16-gb-ram)
9. [💻 Option 7: Local Setup on Windows / Mac PC](#-option-7-local-setup-on-windows--mac-pc)
10. [📁 Project Architecture & File Structure](#-project-architecture--file-structure)
11. [🛠️ Useful Maintenance Commands](#️-useful-maintenance-commands)

---

## 🌟 Features & Capabilities

* **🌟 FLUX.1 [dev] Max (Black Forest Labs • Colab GPU)**: The flagship 12B parameter guidance-distilled flow transformer for the absolute peak of photorealism, perfect human anatomy/hands, complex lighting, and sharp typography.
* **🖼️ Reference Image-to-Image Editor (SDXL • Colab GPU)**: Upload any reference photo and transform clothing, hairstyles, lighting, and backgrounds in ~2.5 seconds with adjustable transformation strength. 100% uncensored.
* **🎨 4K Photorealistic Text-to-Image (SDXL • Colab GPU)**: Sub-3 second generation with cinematic depth-of-field, realistic skin textures, and custom aspect ratios (1:1, 9:16, 16:9, 4:5).
* **👤 1:1 Single Face Swap (VPS & Colab)**: Instant sub-second swapping using InSwapper-128 + Ultra HD facial restoration with GFPGAN.
* **👥 2-Person Custom Swap**: Automatic face detection with thumbnail preview cards, allowing custom mapping of who gets which replacement face.
* **🎯 Group Person Selector (3+ People)**: Crop preview selector to pick exactly 1 person to swap out of a large group.
* **👑 Dynamic Member Access Control**: Admin can toggle member login ON/OFF live from `/links` with instant session invalidation.
* **🚦 1-at-a-Time Concurrency Queue**: Queues concurrent users with live position/ETA counters to guarantee zero server lag.
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

## 🖥️ Option 1: Full 1-Click Setup on Any Linux VPS (Ubuntu / Debian)

### 🚀 **Option 1A: The Single 1-Line Command (Full Auto Setup)**
Copy and paste this single command into your fresh VPS terminal. It updates the server, installs dependencies, clones your private repository, downloads AI models, and starts the 24/7 PM2 production server automatically:

```bash
sudo apt update -y && sudo apt install -y python3 python3-pip python3-venv git ffmpeg libsm6 libxext6 libgl1 nodejs npm nginx ufw && sudo npm install -g pm2 && cd ~ && rm -rf faceswapper && git clone https://OddBoyXdxd69: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@github.com/OddBoyXdxd69/ai-faceswapper-pro.git faceswapper && cd faceswapper && python3 -m venv venv && source venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt && python download_models.py && pm2 delete faceswapper 2>/dev/null || true && pm2 start "venv/bin/uvicorn main:app --host 127.0.0.1 --port 7860" --name faceswapper && pm2 save
```

---

### 🛠️ **Option 1B: Step-by-Step Manual Setup (If Preferred)**

#### Step 1: Update System & Install Dependencies
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git ffmpeg libsm6 libxext6 libgl1 nodejs npm nginx ufw
sudo npm install -g pm2
```

#### Step 2: Clone Private Repository
```bash
cd ~
git clone https://OddBoyXdxd69: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@github.com/OddBoyXdxd69/ai-faceswapper-pro.git faceswapper
cd faceswapper
```

#### Step 3: Set Up Python Virtual Environment & Install Packages
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python download_models.py
```

#### Step 4: Start 24/7 PM2 Server
```bash
pm2 start "venv/bin/uvicorn main:app --host 127.0.0.1 --port 7860" --name faceswapper
pm2 save
pm2 startup
```

#### Step 5: Configure Nginx Reverse Proxy (Domain / Port 80 & 443)
```bash
sudo nano /etc/nginx/sites-available/faceswapper
```
Paste configuration (replace `yourdomain.com` with your actual domain):
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
Enable and restart Nginx:
```bash
sudo ln -sf /etc/nginx/sites-available/faceswapper /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## 🎨 Option 2: Uncensored SDXL Studio (Image-to-Image & 4K Generator) on Google Colab

> [!TIP]
> **Recommended for Fast Creation & Reference Editing:**  
> * 🖼️ **Reference Image-to-Image:** Upload any photo and change clothes, hair, background, or lighting in ~2.5 seconds!  
> * 🎨 **4K Text-to-Image:** Sub-3 second generation with 100% uncensored freedom.  
> * ⚡ **Lightweight & Fast:** Downloads in **~25 seconds** (~6.6 GB) and runs with zero memory lag on Free T4 GPU.

Paste this into a **Google Colab** code cell:

```python
# 1. Step out and clone fresh repository with Auth Token
%cd /content
!rm -rf /content/faceswapper
!git clone https://OddBoyXdxd69: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@github.com/OddBoyXdxd69/ai-faceswapper-pro.git /content/faceswapper
%cd /content/faceswapper

# 2. Install SDXL Dependencies
!pip install -q diffusers transformers accelerate gradio opencv-python-headless pillow invisible-watermark

# 3. Launch Uncensored SDXL Studio with Free Public Live Link (Starts in 25s!)
!python sdxl_img2img_colab.py
```

---

## 🌟 Option 3: FLUX.1 [dev] Max Version (Flagship 12B Flow Transformer) on Google Colab

> [!IMPORTANT]
> **The #1 Highest Quality AI Image Generator in the World:**  
> * 💎 **Flagship 12B Flow Transformer:** Peak photorealism, perfect human anatomy, lifelike skin textures, and flawless typography.  
> * ⚡ **Optimized for Colab GPU:** Uses official 4-bit NF4 quantized weights (~6.8 GB) to download in **~25s** and run smoothly within T4 GPU VRAM without crashing.

Paste this into a **Google Colab** code cell:

```python
# 1. Step out and clone fresh repository with Auth Token
%cd /content
!rm -rf /content/faceswapper
!git clone https://OddBoyXdxd69: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@github.com/OddBoyXdxd69/ai-faceswapper-pro.git /content/faceswapper
%cd /content/faceswapper

# 2. Install FLUX [dev] Max Dependencies
!pip install -q diffusers transformers accelerate gradio sentencepiece protobuf torch torchvision bitsandbytes huggingface_hub

# 3. Launch FLUX.1 [dev] Max Studio with Free Public Live Link
!python flux_max_colab.py
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

# 4. Launch with Free Public Live Link
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
├── main.py                   # Main FastAPI & Gradio Server for Linux VPS (Dual Auth + Stealth + Queue)
├── setup_vps.sh              # 1-Click Automated Setup script for Linux VPS
├── flux_max_colab.py         # Flagship FLUX.1 [dev] Max 12B Studio (Google Colab GPU)
├── flux_max_colab.ipynb      # 1-Click FLUX.1 [dev] Max Colab Notebook
├── sdxl_img2img_colab.py     # Uncensored SDXL Image-to-Image & 4K Generator (Google Colab GPU)
├── sdxl_colab.ipynb          # 1-Click SDXL Colab Notebook
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
