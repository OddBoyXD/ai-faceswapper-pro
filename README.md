# ⚡ AI Face Swapper Pro • Complete A-to-Z Guide

An enterprise-grade, high-accuracy AI Face Swapper powered by **InsightFace (InSwapper-128)**, **GFPGAN v1.4 (Ultra HD)**, and **FastAPI / Gradio**. Features visual multi-person selector, zero-browser-history stealth camouflage, role-based dual encrypted vaults, and dynamic admin access controls.

---

## 📋 Table of Contents
1. [🌟 Features & Capabilities](#-features--capabilities)
2. [🔑 Access Passwords & Roles](#-access-passwords--roles)
3. [🖥️ Option 1: Full A-to-Z Installation on Any Linux VPS (Ubuntu / Debian)](#️-option-1-full-a-to-z-installation-on-any-linux-vps-ubuntu--debian)
4. [☁️ Option 2: 100% Free 24/7 Hosting on Hugging Face Spaces](#️-option-2-100-free-247-hosting-on-hugging-face-spaces-16-gb-ram)
5. [🚀 Option 3: Free GPU on Kaggle (30h Free GPU Weekly)](#-option-3-free-gpu-on-kaggle-30h-free-gpu-weekly)
6. [⚡ Option 4: Free GPU on Google Colab (Nvidia T4 GPU)](#-option-4-free-gpu-on-google-colab-nvidia-t4-gpu)
7. [💻 Option 5: Local Setup on Windows / Mac PC](#-option-5-local-setup-on-windows--mac-pc)
8. [📁 Project Architecture & File Structure](#-project-architecture--file-structure)
9. [🛠️ Useful Maintenance Commands](#️-useful-maintenance-commands)

---

## 🌟 Features & Capabilities

* **👤 1:1 Single Face Swap**: Instant swapping using InSwapper-128 + Ultra HD facial restoration with GFPGAN.
* **👥 2-Person Custom Swap**: Automatic face detection with thumbnail preview cards, allowing custom mapping of who gets which replacement face.
* **🎯 Group Person Selector (3+ People)**: Crop preview selector to pick exactly 1 person to swap out of a large group.
* **🌐 File Upload & Image URL Support**: Paste image URLs directly or upload from your device.
* **👑 Dynamic Member Access Control**: Admin can toggle member login ON/OFF live from `/links` with instant session invalidation.
* **🕵️ Zero-History Stealth Camouflage**:
  * HTTP Headers: `Cache-Control: no-store, no-cache`, `X-Robots-Tag: noindex, nofollow`.
  * Browser: `history.replaceState()` prevents `/links` and photo names from showing in browser history / search bar autocomplete.
  * Tab title disguised as `Cloud Storage` / `Cloud Drive • Files`.
* **📁 Hidden System Vault**: Processed photos auto-saved inside a hidden directory (`~/.sys_vault/data/`).
* **📱 Mobile-Optimized Vault**: Fullscreen in-page lightbox viewer with silent in-memory (`blob:`) download and one-click AJAX photo deletion.

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
*(This downloads `inswapper_128.onnx` and `gfpgan_1.4.onnx` directly into `models/`)*.

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

### Step 8: Setup Free SSL Certificate (HTTPS)
If using Cloudflare: Set SSL mode to **Flexible** or **Full (Strict)**.  
If using Certbot (Direct Let's Encrypt):
```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

---

## ☁️ Option 2: 100% Free 24/7 Hosting on Hugging Face Spaces (16 GB RAM)

Hugging Face provides **16 GB RAM and 2 vCPUs for FREE forever**.

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. **Space Name**: `ai-faceswapper`
3. **SDK**: Select **Gradio**.
4. **Hardware**: Choose **Free (2 vCPU • 16 GB RAM • 50GB Disk)** or **ZeroGPU**.
5. Connect your GitHub repository `https://OddBoyXdxd69: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@github.com/OddBoyXdxd69/ai-faceswapper-pro.git`.
6. Hugging Face will automatically run and deploy your app with a permanent HTTPS link!

---

## 🚀 Option 3: Free GPU on Kaggle (30h Free GPU Weekly)

> [!IMPORTANT]
> **Before running on Kaggle:**  
> In the Kaggle notebook right sidebar:  
> 1. Go to **Settings ➔ Internet** and turn it **ON** (Required to download models & libraries).  
> 2. Go to **Settings ➔ Accelerator** and select **GPU T4 x2** or **GPU P100**.

Paste this code into a Kaggle code cell:

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
*(Gradio outputs a public `https://xxxx.gradio.live` link that you can open on your phone or desktop)*.

---

## ⚡ Option 4: Free GPU on Google Colab (Nvidia T4 GPU)

Paste this into a Google Colab notebook cell:

```python
# 1. Clone Private Repository to Colab Content Directory
!git clone https://OddBoyXdxd69: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@github.com/OddBoyXdxd69/ai-faceswapper-pro.git /content/faceswapper
%cd /content/faceswapper

# 2. Install Dependencies
!pip install -q fastapi uvicorn gradio insightface onnxruntime-gpu opencv-python-headless pillow requests

# 3. Download AI Models
!python download_models.py

# 4. Launch with Free Public Gradio Share Link
import app
app.demo.launch(share=True)
```

---

## 💻 Option 5: Local Setup on Windows / Mac PC

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
├── main.py                   # Main FastAPI & Gradio Server with Auth & Stealth Middleware
├── download_models.py        # Automated downloader for InSwapper & GFPGAN ONNX models
├── app.py                    # Lightweight standalone Gradio runner
├── ai_faceswapper_colab.ipynb# 1-Click Google Colab GPU notebook
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
