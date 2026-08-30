#!/bin/bash
set -e

echo "🚀 Starting 1-Click Automated AI Face Swapper Pro Installation..."

# 1. Update System & Install System Dependencies
echo "📦 Installing system dependencies..."
sudo apt update -y
sudo apt install -y python3 python3-pip python3-venv git ffmpeg libsm6 libxext6 libgl1 nodejs npm nginx ufw

# 2. Install PM2
echo "⚡ Installing PM2 Process Manager..."
sudo npm install -g pm2

# 3. Clone / Update Repository
TARGET_DIR="$HOME/faceswapper"
if [ ! -d "$TARGET_DIR" ]; then
    echo "📥 Cloning private repository..."
    git clone https://OddBoyXdxd69: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@github.com/OddBoyXdxd69/ai-faceswapper-pro.git "$TARGET_DIR"
fi

cd "$TARGET_DIR"

# 4. Virtual Environment & Python Packages
echo "🐍 Setting up Python Virtual Environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 5. Download AI Models
echo "🧠 Downloading AI Face Swap & GFPGAN Models..."
python download_models.py

# 6. Start / Restart PM2 Service
echo "🚀 Launching 24/7 Production Server on Port 7860..."
pm2 delete faceswapper 2>/dev/null || true
pm2 start "venv/bin/uvicorn main:app --host 127.0.0.1 --port 7860" --name faceswapper
pm2 save

echo "=========================================================="
echo "🎉 SUCCESS! AI Face Swapper Pro is LIVE & Running 24/7!"
echo "📍 Local Port: http://127.0.0.1:7860"
echo "🔑 Member PIN: 66776699M | Admin PIN: 697769"
echo "=========================================================="
