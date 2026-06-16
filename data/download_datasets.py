"""
Dataset Download Helper
Downloads all required datasets for Driver Monitoring System
Run this script once to set up your data directory
"""

import os
import urllib.request
import zipfile


def download_file(url: str, dest: str):
    print(f"Downloading {os.path.basename(dest)}...")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def _progress(block, block_size, total):
    pct = min(block * block_size / total * 100, 100)
    print(f"\r  {pct:5.1f}%", end="", flush=True)


def setup_dirs():
    dirs = ["data/raw", "data/processed", "models/weights"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ Directories created")


def download_yolo_base():
    """YOLOv8n base weights — auto-downloaded by ultralytics on first run"""
    print("\n📦 YOLOv8n base weights:")
    print("  These are auto-downloaded when you first run the app.")
    print("  File: yolov8n.pt (~6MB)")


def instructions_roboflow():
    print("\n📦 Cigarette Detection Dataset (Roboflow):")
    print("  1. Go to: https://universe.roboflow.com")
    print("  2. Search: 'cigarette detection'")
    print("  3. Choose a dataset with 1000+ images")
    print("  4. Export as YOLOv8 format")
    print("  5. Download and place in: data/datasets/cigarette/")
    print("  OR use the Colab notebook: notebooks/YOLOv8_Training.ipynb")


def instructions_mrl():
    print("\n📦 MRL Eye Dataset:")
    print("  1. Go to: http://mrl.cs.vsb.cz/eyedataset")
    print("  2. Download the dataset (~3GB)")
    print("  3. Extract to: data/raw/mrl_eye/")
    print("  Note: Used for evaluation only — EAR algorithm needs no training!")


if __name__ == "__main__":
    print("=" * 55)
    print("  Driver Monitoring System — Dataset Setup")
    print("  Cairo University | FCAI | AI Dept | 2026")
    print("=" * 55)

    setup_dirs()
    download_yolo_base()
    instructions_roboflow()
    instructions_mrl()

    print("\n" + "=" * 55)
    print("✅ Setup complete!")
    print("\nNext steps:")
    print("  1. Open notebooks/YOLOv8_Training.ipynb in Google Colab")
    print("  2. Run all cells to train the cigarette detector")
    print("  3. Download cigarette.pt and place in models/weights/")
    print("  4. Run: streamlit run app/streamlit_app.py")
    print("=" * 55)
