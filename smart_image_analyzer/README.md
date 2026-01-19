# Smart Image Analyzer 🚀

A high-performance, modular Computer Vision pipeline designed for **real-time object detection** using state-of-the-art Deep Learning models.

## 🎯 Problem Statement
Building a robust object detection system requires more than just running a model. It involves handling diverse input sources, ensuring efficient hardware utilization (CUDA), preprocessing noisy data, and presenting results clearly. This project demonstrates a production-grade approach to these challenges.

## 🛠️ Tech Stack & Architecture

-   **Core**: Python 3.10+
-   **Deep Learning**: PyTorch (CUDA Accelerated) + Ultralytics YOLOv8
-   **Image Processing**: 
    -   `scikit-image` for advanced signal processing (denoising, edge detection).
    -   `OpenCV` for high-throughput visualization and manipulation.
-   **Deployment**: Automated ONNX export for edge optimization.

### Architecture Flow
`Input (URL/Path) -> Preprocessing (Denoise/Resize) -> YOLOv8 Inference (GPU) -> Visualization/Stats -> ONNX Export`

## ⚡ Why CUDA Matters
Deep learning models involve massive matrix multiplications.
-   **CPU**: Sequential processing, good for logic but slow for tensors. (~2-4 FPS)
-   **GPU (CUDA)**: Parallel processing of thousands of cores. (~50+ FPS)
This project uses `torch.device` to automatically offload heavy tensor computations to the NVIDIA GPU, achieving real-time performance.

## 📊 Key Features

-   **Automatic Hardware Awareness**: Detects specific GPU (e.g., RTX 4060) or falls back to CPU.
-   **Interactive Mode**: Accepts file paths, URLs, or directories directly from the terminal.
-   **Performance Metrics**: Real-time FPS and inference time logging.
-   **Portfolio Artifact Generator**: Automatically stitches inputs, outputs, and stats into a single `portfolio_report.jpg`.
-   **Hybrid Pipeline**: Demonstrates both Classical CV (Edges/Grayscale) and AI (YOLO) in one workflow.

## 🚀 How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r smart_image_analyzer/requirements.txt
    ```

2.  **Execute**:
    ```bash
    python smart_image_analyzer/main.py
    ```
    *Follow the prompts to enter a URL or path.*

3.  **Check Results**:
    Open `smart_image_analyzer/outputs/portfolio_report.jpg` to see the generated summary.

## 📈 Results Comparison
| Stage | Visualization |
|-------|---------------|
| **Original** | Raw RGB input |
| **Preprocessing** | Denoised & Canny Edges (Scikit-Image) |
| **Detection** | Bounding Boxes + Confidence (YOLOv8) |

## 🏗️ Project Structure
```
smart_image_analyzer/
├── src/                # Modular Source Code
│   ├── detect.py       # YOLO Logic & Export
│   ├── visualize.py    # Drawing & Artifact Gen
│   ├── preprocess.py   # Scikit-Image Pipeline
│   └── utils.py        # Hardware & Downloads
├── data/               # Input Images
├── outputs/            # Results & Reports
├── models/             # Exported ONNX models
└── main.py             # Orchestrator
```
