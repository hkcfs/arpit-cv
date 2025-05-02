# Retail Image Annotation Proof of Concept (PoC)

This project implements a computer vision pipeline to analyze retail store images, performing object detection, classification, OCR, and entity linking to generate structured JSON annotations.

**Features:**

- Object Detection (Stores, POSM, Products, SKUs, Brand Ambassadors) using YOLOv8.
- Text Detection and Recognition (OCR) using PaddleOCR.
- Hierarchical Entity Linking based on spatial relationships and heuristics.
- Output generation in a COCO-like JSON format.

**Setup:**

1.  Clone the repository.
2.  Install dependencies: `pip install -r requirements.txt`
3.  Download or train your object detection model weights (e.g., `models/object_detection/yolov8s_retail.pt`). PaddleOCR models are often downloaded automatically on first run.
4.  Place images in `data/images/`.

**Usage:**

Run the main pipeline script:
```bash
python src/pipeline/full_pipeline.py --image_path data/images/your_image.jpg --output_path data/processed/output.json
```