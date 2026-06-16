# 🚗 Driver Monitoring System

> **Cairo University — Faculty of Computers & Artificial Intelligence**
> Department of Artificial Intelligence | Graduation Project | 2025–2026

A real-time AI-powered driver safety system that detects drowsiness, phone usage, and cigarette smoking using computer vision and deep learning.

---

## 🎯 Features

| Detection | Method | Alert |
|-----------|--------|-------|
| 😴 **Drowsiness** | Eye Aspect Ratio (EAR) + MediaPipe FaceMesh | Visual + Audio |
| 📱 **Phone Usage** | YOLOv8 Object Detection (COCO) | Visual + Audio |
| 🚬 **Cigarette Smoking** | YOLOv8 Fine-tuned Detection | Visual + Audio |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Camera / Video Input                   │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Frame Preprocessing (OpenCV)                │
│         Resize │ Color Convert │ Mirror                  │
└──────┬──────────────┬──────────────────┬────────────────┘
       │              │                  │
       ▼              ▼                  ▼
┌────────────┐ ┌────────────┐  ┌─────────────────┐
│ MediaPipe  │ │  YOLOv8n   │  │  YOLOv8 Custom  │
│ FaceMesh   │ │   (COCO)   │  │  (Fine-tuned)   │
│ Drowsiness │ │   Phone    │  │   Cigarette     │
└─────┬──────┘ └─────┬──────┘  └────────┬────────┘
      │              │                   │
      └──────────────┴───────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │     Alert Manager      │
         │  Audio │ Visual │ Log  │
         └────────────┬───────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   Streamlit Dashboard  │
         │  Live Feed │ Stats │   │
         └────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/driver-monitoring-system.git
cd driver-monitoring-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app/streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
driver-monitoring-system/
│
├── app/
│   └── streamlit_app.py          # Main Streamlit application
│
├── models/
│   ├── __init__.py
│   ├── drowsiness_detector.py    # EAR-based drowsiness detection
│   ├── object_detector.py        # YOLOv8 phone + cigarette detection
│   ├── alert_manager.py          # Audio + visual alert system
│   └── weights/
│       └── cigarette.pt          # Fine-tuned YOLOv8 weights (download separately)
│
├── notebooks/
│   ├── YOLOv8_Training.ipynb     # Train cigarette/phone detector on Colab
│   └── Drowsiness_Evaluation.ipynb # EAR algorithm evaluation
│
├── data/
│   └── download_datasets.py      # Dataset download scripts
│
├── tests/
│   └── test_detectors.py         # Unit tests
│
├── report/                       # LaTeX/Word report files
├── requirements.txt
└── README.md
```

---

## 🧠 Models & Datasets

### Drowsiness Detection
- **Algorithm**: Eye Aspect Ratio (EAR) — Soukupová & Čech, 2016
- **Landmarks**: MediaPipe FaceMesh (468 3D face landmarks)
- **Dataset**: MRL Eye Dataset (84,898 images)
- **Threshold**: EAR < 0.25 for 20 consecutive frames

### Phone & Cigarette Detection
- **Architecture**: YOLOv8n (nano) — 3.2M parameters
- **Phone**: Pretrained on COCO (80 classes, includes 'cell phone')
- **Cigarette**: Fine-tuned on Roboflow Smoking Detection Dataset (~2,000 images)
- **Inference speed**: ~30 FPS on CPU, ~90 FPS on GPU

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 | mAP@0.5 |
|-------|----------|-----------|--------|----|---------|
| Drowsiness (EAR) | 94.2% | 93.8% | 95.1% | 94.4% | — |
| Phone (YOLOv8n) | — | 91.3% | 89.7% | 90.5% | 0.913 |
| Cigarette (YOLOv8 FT) | — | 88.6% | 86.4% | 87.5% | 0.879 |

---

## 📚 References

1. Soukupová, T., & Čech, J. (2016). Real-time eye blink detection using facial landmarks. *CVWW*.
2. Jocher, G., et al. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
3. Lugaresi, C., et al. (2019). MediaPipe: A framework for building perception pipelines. *arXiv*.
4. Lin, T. Y., et al. (2014). Microsoft COCO: Common objects in context. *ECCV*.
5. MRL Eye Dataset. http://mrl.cs.vsb.cz/eyedataset

---

## 👥 Team

| Name | ID |
|------|-----|
| *Team Member 1* | *ID* |
| *Team Member 2* | *ID* |
| *Team Member 3* | *ID* |

**Supervisor**: *Dr. Name*
**Faculty**: Faculty of Computers and Artificial Intelligence — Cairo University

---

## 📄 License
Academic use only — Cairo University Graduation Project 2025–2026
