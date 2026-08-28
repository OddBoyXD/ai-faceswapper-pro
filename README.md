# ⚡ AI Face Swapper Pro

Advanced AI Face Swapper with 1:1 Pixel Accuracy, 2-Person Custom Swapping, Multi-Person Visual Selector, Zero-History Session Camouflage, and Encrypted Vault Management.

---

## 🚀 Quick Start (Local or Any Linux VPS)

### 1. Install Dependencies
```bash
git clone https://github.com/OddBoyXdxd69/ai-faceswapper-pro.git
cd ai-faceswapper-pro
pip install -r requirements.txt
```

### 2. Download AI Models
```bash
python download_models.py
```

### 3. Run Server
```bash
python main.py
# Or with uvicorn / pm2:
# pm2 start "uvicorn main:app --host 0.0.0.0 --port 7860" --name faceswapper
```

---

## 🔑 Access Credentials

* **Main Face Swapper App (`/`)**: `66776699M` (1-Hour auto-expiring session cookie).
* **Admin Files & Photo Vault (`/links`)**: `697769` (Unlocks View, Copy, Save, and 🗑️ Delete buttons).
* **Member Files View (`/links`)**: `66776699M` (View, Copy, Save with Delete buttons hidden).

---

## ☁️ 100% Free Hosting Options When VPS Expires

### Option 1: Hugging Face Spaces (Recommended - 100% Free 24/7)
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and create a **New Space**.
2. Select **Gradio** or **Docker** SDK (Free CPU tier: **16 GB RAM, 2 vCPUs**).
3. Connect your GitHub repository `OddBoyXdxd69/ai-faceswapper-pro`.
4. Your app runs 24/7 for free with a public HTTPS link!

### Option 2: Google Colab / Kaggle (Free GPU T4)
* Open Google Colab (Free T4 GPU runtime).
* Clone this repository, run `pip install -r requirements.txt`, download models, and run with Gradio public share link (`demo.launch(share=True)`).

### Option 3: Oracle Cloud Always-Free Tier
* Oracle offers **Always Free 4 ARM vCPUs & 24 GB RAM** VPS instances forever.

---

## 🛡️ Key Features & Architecture
* **Single Face Swap**: High-speed 1:1 face swapping with Ultra GFPGAN restoration.
* **2-Person Swap**: Visual crop detection allowing custom source-to-target person mapping.
* **Group Face Swap**: Visual face selection for large multi-person images.
* **Stealth Camouflage**: Browser history prevention (`no-store`, `no-cache`, `history.replaceState()`, disguised titles `Cloud Storage`).
* **Hidden Vault**: Output images stored in hidden directory (`~/.sys_vault/data`).
