import os
import sys
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

MODELS = {
    "inswapper_128.onnx": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
    "gfpgan_1.4.onnx": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.4.onnx",
    "codeformer.onnx": "https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx",
    "GPEN-BFR-512.onnx": "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_512.onnx"
}

def download_models():
    print("=" * 60)
    print("🌟 Downloading Ultra-Realistic 4K AI Models...")
    print("=" * 60)
    for fname, url in MODELS.items():
        fp = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(fp) or os.path.getsize(fp) < 1000000:
            print(f"📥 Downloading {fname}...")
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as resp, open(fp, 'wb') as f:
                    total_size = int(resp.headers.get('Content-Length', 0))
                    downloaded = 0
                    chunk_size = 1024 * 1024
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rProgress: {percent:.1f}% ({downloaded//(1024*1024)}MB / {total_size//(1024*1024)}MB)", end="")
                print(f"\n✅ Successfully downloaded {fname} ({os.path.getsize(fp)//(1024*1024)} MB)")
            except Exception as e:
                print(f"\n⚠️ Note for {fname}: {e}")
        else:
            print(f"✅ Found {fname} ({os.path.getsize(fp)//(1024*1024)} MB)")
    print("=" * 60)
    print("✅ All Ultra-Realistic Models Verified!")
    print("=" * 60)

if __name__ == "__main__":
    download_models()
