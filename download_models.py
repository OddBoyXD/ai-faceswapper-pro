import os
import sys
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

MODELS = {
    "inswapper_128.onnx": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
    "gfpgan_1.4.onnx": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.4.onnx"
}

def download_models():
    for fname, url in MODELS.items():
        fp = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(fp):
            print(f"Downloading {fname}...")
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
                print(f"\n✅ Successfully downloaded {fname}")
            except Exception as e:
                print(f"\n❌ Error downloading {fname}: {e}")
        else:
            print(f"✅ Found {fname} ({os.path.getsize(fp)//(1024*1024)} MB)")

if __name__ == "__main__":
    download_models()
