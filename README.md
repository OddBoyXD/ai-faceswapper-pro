# ⚡ AI Face Swapper Pro • Complete A-to-Z Guide

An enterprise-grade, high-accuracy AI Face Swapper powered by **InsightFace (InSwapper-128)**, **GFPGAN v1.4 (Ultra HD)**, and **FastAPI / Gradio**. Features visual multi-person selector, zero-browser-history stealth camouflage, and role-based dual encrypted vaults.

---

## 📋 Table of Contents
1. [🌟 Features & Capabilities](#-features--capabilities)
2. [🔑 Access Passwords & Roles](#-access-passwords--roles)
3. [🖥️ Option 1: Full A-to-Z Installation on Any Linux VPS (Ubuntu / Debian)](#️-option-1-full-a-to-z-installation-on-any-linux-vps-ubuntu--debian)
4. [☁️ Option 2: 100% Free 24/7 Hosting on Hugging Face Spaces](#️-option-2-100-free-247-hosting-on-hugging-face-spaces-16-gb-ram)
5. [🚀 Option 3: Free GPU Execution on Google Colab / Kaggle](#-option-3-free-gpu-execution-on-google-colab--kaggle)
6. [💻 Option 4: Local Setup on Windows / Mac PC](#-option-4-local-setup-on-windows--mac-pc)
7. [📁 Project Architecture & File Structure](#-project-architecture--file-structure)
8. [🛠️ Useful Maintenance Commands](#️-useful-maintenance-commands)

---

## 🌟 Features & Capabilities

* **👤 1:1 Single Face Swap**: Instant swapping using InSwapper-128 + Ultra HD facial restoration with GFPGAN.
* **👥 2-Person Custom Swap**: Automatic face detection with thumbnail preview cards, allowing custom mapping of who gets which replacement face.
* **🎯 Group Person Selector (3+ People)**: Crop preview selector to pick exactly 1 person to swap out of a large group.
* **🌐 File Upload & Image URL Support**: Paste image URLs directly or upload from your device.
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
| **Main App (`/`)** | `66776699M` | **User** | Full access to Face Swapper (1-Hour auto-expiring cookie). |
| **Files Vault (`/links`)** | `697769` | 👑 **Admin** | View all photos, Copy Link, Save Image, and 🗑️ **Delete Photo**. |
| **Files Vault (`/links`)** | `66776699M` | 👤 **Member** | View all photos, Copy Link, and Save Image *(Delete hidden)*. |

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

### Step 3: Clone Your Private Repository
```bash
cd /home/ubuntu   # or your home directory
git clone https://github.com/OddBoyXdxd69/ai-faceswapper-pro.git faceswapper
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
3. **License**: `MIT` or `OpenRAIL`
4. **SDK**: Select **Gradio** or **Docker**.
5. **Hardware**: Choose **Free (2 vCPU • 16 GB RAM • 50GB Disk)**.
6. Connect your GitHub repository `OddBoyXdxd69/ai-faceswapper-pro`.
7. Hugging Face will automatically run `download_models.py` and deploy your app with a permanent HTTPS link!

---

## 🚀 Option 3: Free GPU Execution on Google Colab / Kaggle

Run on a **Free Nvidia T4 GPU (16 GB VRAM)** with sub-second swap speed:

```python
# In a Google Colab / Kaggle Notebook cell:
!git clone https://github.com/OddBoyXdxd69/ai-faceswapper-pro.git
%cd ai-faceswapper-pro
!pip install -r requirements.txt
!python download_models.py
!python main.py
```
*(Enable `share=True` in Gradio blocks to get a public 72-hour `gradio.live` link).*

---

## 💻 Option 4: Local Setup on Windows / Mac PC

### Windows (PowerShell):
```powershell
git clone https://github.com/OddBoyXdxd69/ai-faceswapper-pro.git
cd ai-faceswapper-pro
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python download_models.py
python main.py
```
Open browser at: `http://localhost:7860`

### macOS / Linux:
```bash
git clone https://github.com/OddBoyXdxd69/ai-faceswapper-pro.git
cd ai-faceswapper-pro
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
├── main.py              # Main FastAPI & Gradio Server with Auth & Middleware
├── download_models.py   # Automated downloader for InSwapper & GFPGAN ONNX models
├── app.py               # Lightweight standalone Gradio runner
├── run_server.py        # Process orchestrator script
├── requirements.txt     # Complete Python dependencies
├── .gitignore           # Ignores large binaries, caches, and secret data
└── README.md            # Complete Documentation
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
