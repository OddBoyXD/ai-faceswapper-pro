#!/bin/bash
export DEBIAN_FRONTEND=noninteractive
set -e

echo "=========================================================="
echo "🚀 1-CLICK UBUNTU VPS INSTALLER: AI FACE SWAPPER PRO"
echo "=========================================================="

# 1. Update Ubuntu & Install Dependencies
echo "📦 [1/5] Updating Ubuntu packages & installing dependencies..."
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv git ffmpeg libsm6 libxext6 libgl1 nodejs npm nginx ufw

# 2. Install Node.js PM2 Process Manager
echo "⚡ [2/5] Installing PM2 process manager..."
sudo npm install -g pm2

# 3. Clone Private Repository
WORK_DIR="/home/ubuntu/faceswapper"
if [ ! -d "/home/ubuntu" ]; then
    WORK_DIR="$HOME/faceswapper"
fi

echo "📥 [3/5] Cloning private repository into $WORK_DIR..."
rm -rf "$WORK_DIR"
git clone https://OddBoyXdxd69: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@github.com/OddBoyXdxd69/ai-faceswapper-pro.git "$WORK_DIR"
cd "$WORK_DIR"

# 4. Virtual Environment & Models
echo "🐍 [4/5] Setting up Python virtual environment & packages..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "🧠 [5/5] Downloading AI face swapping & GFPGAN models..."
python download_models.py

# 5. Launch 24/7 PM2 Server
echo "🚀 Starting 24/7 PM2 service on Port 7860..."
pm2 delete faceswapper 2>/dev/null || true
pm2 start "venv/bin/uvicorn main:app --host 127.0.0.1 --port 7860" --name faceswapper
pm2 save

echo "=========================================================="
echo "🎉 SUCCESS! AI Face Swapper Pro is LIVE on Ubuntu VPS!"
echo "🌐 Local Port: http://127.0.0.1:7860"
echo "🔑 Member PIN: 66776699M | Admin PIN: 697769"
echo "=========================================================="
