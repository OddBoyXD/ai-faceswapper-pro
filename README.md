<div align="center">

# ⚡ AI Face Swapper Pro & Ultra 4K Creation Suite

[![Google Colab Ultra](https://img.shields.io/badge/Google%20Colab-Launch%20Ultra%204K-f9ab00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/OddBoyXD/ai-faceswapper-pro/blob/main/AI_FaceSwapper_Ultra.ipynb)
[![Google Colab Standard](https://img.shields.io/badge/Google%20Colab-Launch%20Standard-blue?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/OddBoyXD/ai-faceswapper-pro/blob/main/AI_FaceSwapper_Pro.ipynb)
[![FLUX.1 Max](https://img.shields.io/badge/Black%20Forest%20Labs-FLUX.1%2012B-8a2be2?style=for-the-badge&logo=huggingface&logoColor=white)](https://colab.research.google.com/github/OddBoyXD/ai-faceswapper-pro/blob/main/flux_max_colab.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<p align="center">
  <b>A Studio-Grade, Photorealistic AI Face Cloning & Multi-Person Replacement Suite</b><br>
  Powered by <b>InsightFace (InSwapper-128)</b> • <b>GFPGAN v1.4 HD</b> • <b>CodeFormer</b> • <b>FLUX.1 [dev]</b> • <b>SDXL 1.0</b>
</p>

---

[🚀 Quick Start (Google Colab)](#-1-click-google-colab-notebooks) • [✨ Key Features](#-features--capabilities) • [💎 Ultra vs Standard](#-ultra-4k-vs-standard-comparison) • [🖥️ VPS Deployment](#-vps-installation) • [📜 FAQ & Hair Explanation](#-frequently-asked-questions)

</div>

---

## 🚀 1-Click Google Colab Notebooks

Run free on NVIDIA T4 GPU with zero local installation:

| Notebook | Link | Best For | Output Quality |
| :--- | :---: | :--- | :---: |
| 💎 **AI FaceSwapper Ultra** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OddBoyXD/ai-faceswapper-pro/blob/main/AI_FaceSwapper_Ultra.ipynb) | 4K Skin Pores, Natural Eyes, Zero Blur, 100% Outfit/Body Preservation | **4K Ultra HD** |
| ⚡ **AI FaceSwapper Pro** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OddBoyXD/ai-faceswapper-pro/blob/main/AI_FaceSwapper_Pro.ipynb) | Lightning Fast Swapping (1–2 seconds) | **High Definition** |
| 🌟 **FLUX.1 [dev] Max Studio** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OddBoyXD/ai-faceswapper-pro/blob/main/flux_max_colab.ipynb) | 12B Flagship Text-to-Image Generation (Hands, Anatomy, Lighting) | **8K Studio Quality** |
| 🎨 **SDXL Img2Img Studio** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OddBoyXD/ai-faceswapper-pro/blob/main/sdxl_colab.ipynb) | Reference Image-to-Image Style & Outfit Transformation | **Photorealistic** |

---

## 🌟 Features & Capabilities

* 👤 **1:1 Single Face Swap**: High-accuracy facial feature transfer with real-time pose and gaze tracking.
* 👥 **Multi-Person Custom Swap**: Automatic group face detection with interactive thumbnail selector to assign custom replacement faces.
* 👗 **100% Body & Outfit Preservation**: Clothes, suits, dresses, jewelry, body posture, and background remain completely untouched.
* 💎 **Ultra 4K Neural Restoration**: Multi-scale GFPGAN v1.4 and CodeFormer integration for authentic micro-skin pores and sharp eyelashes.
* 💡 **Dynamic Lighting Harmonization**: Adapts facial skin tone, contrast, and highlights to match ambient scene lighting.
* 🌐 **Instant Free Sharing**: 1-Click Gradio Live URL, Localtunnel, or Ngrok tunnel.

---

## 📊 Ultra 4K vs Standard Comparison

```
┌──────────────────────────┬─────────────────────────┬─────────────────────────┐
│ Feature                  │ ⚡ Standard Swapper     │ 💎 Ultra 4K Studio      │
├──────────────────────────┼─────────────────────────┼─────────────────────────┤
│ Inference Speed          │ ⚡ 1–2 Seconds          │ 🚀 3–4 Seconds          │
│ Skin Micro-Pores         │ Standard Smoothing      │ 💎 Real 4K Pores        │
│ Eyelash & Teeth Clarity  │ Good                    │ 🔍 Ultra-Sharp HD       │
│ Lighting Harmonization   │ Basic                   │ 💡 Dynamic LAB Matching │
│ Target Clothes & Body    │ 100% Preserved          │ 100% Preserved          │
│ Target Background        │ 100% Preserved          │ 100% Preserved          │
└──────────────────────────┴─────────────────────────┴─────────────────────────┘
```

---

## 🖥️ VPS Installation (Ubuntu 20.04 / 22.04 / 24.04)

To deploy on a remote cloud VPS (AWS, DigitalOcean, Hetzner):

```bash
# Clone the repository
git clone https://github.com/OddBoyXD/ai-faceswapper-pro.git
cd ai-faceswapper-pro

# Run the automated setup script
chmod +x setup_vps.sh
./setup_vps.sh

# Download models & launch
python3 download_models_ultra.py
python3 app_ultra.py
```

---

## ❓ Frequently Asked Questions

<details>
<summary><b>Q: Does this model change the hair or only the face?</b></summary>
<br>
<b>A:</b> By design in facial transfer architectures (InsightFace / InSwapper / FaceFusion), the model replaces the <b>facial features (eyes, nose, mouth, cheeks, jawline)</b> while intentionally keeping the target photo's <b>hair, body, and clothing</b> intact. 

This ensures that:
1. The head fits naturally on the target person's neck, collar, and posture.
2. The clothing, background, and hairstyle of the original scene are 100% preserved.

To change or generate completely new hairstyles from scratch, use the included <b>FLUX.1 [dev] Max</b> or <b>SDXL Studio</b> notebooks.
</details>

<details>
<summary><b>Q: How do I save my generated photos permanently on Google Colab?</b></summary>
<br>
<b>A:</b> Run the first cell <code>Mount Google Drive (Optional)</code> in the notebook. All outputs will automatically save to your Google Drive in the folder <code>/MyDrive/FaceSwapper_Ultra_Outputs</code>.
</details>

---

## ⚖️ Ethical Usage & Disclaimer

This software is provided for creative, educational, artistic, and research applications only. Users are strictly responsible for complying with local laws, respecting privacy rights, and obtaining appropriate consent before using personal photographs. The authors assume no liability for misuse.

---

<div align="center">
  <b>Developed by OddBoyXD • Free & Open-Source</b>
</div>
