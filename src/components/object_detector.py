from ultralytics import YOLO
import cv2
import numpy as np
from src.config import config
from src.utils.bbox_utils import xyxy_to_xywh # YOLO output is xyxy

class ObjectDetector:
    def __init__(self, model_path=config.YOLO_MODEL_PATH):
        # Load a pre-trained YOLO model
        # Specify device='cpu' here
        self.model = YOLO(model_path)
        self.device = 'cpu' # Explicitly set device to CPU
        print("YOLO Model Names (ID to Name):", self.model.names)
        print("Config Category Mapping (ID to Name):", config.CATEGORY_MAPPING)
        print("Config Class to Category ID Mapping (Name to ID):", config.CLASS_TO_CATEGORY_ID)

        # Map model class IDs to our category IDs
        self.model_class_to_category_id = {}
        for model_class_id, class_name in self.model.names.items():
             # Assuming your trained YOLO model class names match the CATEGORY_MAPPING values
             if class_name in config.CLASS_TO_CATEGORY_ID:
                 self.model_class_to_category_id[model_class_id] = config.CLASS_TO_CATEGORY_ID[class_name]
             # Special handling if YOLO detects 'text_region' and OCR handles recognition
             elif class_name == 'text_region' and 'text' in config.CLASS_TO_CATEGORY_ID:
                  self.model_class_to_category_id[model_class_id] = config.CLASS_TO_CATEGORY_ID['text']


    def detect_objects(self, img):
        """Performs object detection on the image."""
        # YOLO expects RGB image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run inference
        # Pass the device='cpu' argument to the predict method
        results = self.model.predict(img_rgb,
                                     conf=config.DETECTION_CONFIDENCE_THRESHOLD,
                                     iou=config.NMS_THRESHOLD,
                                     device=self.device, # Use the configured device
                                     verbose=False) # Set verbose=True for debugging


        detected_objects = []
        annotation_counter = 1 # Start IDs for detected objects

        # Process results
        for r in results:
            boxes = r.boxes # Boxes object for bounding box outputs
            for box in boxes:
                # box.xyxy: bounding box coordinates in [x1, y1, x2, y2] format
                # box.conf: confidence score
                # box.cls: class index
                bbox_xyxy = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                # Convert bbox to xywh format
                bbox_xywh = xyxy_to_xywh(bbox_xyxy)

                # Map model class ID to our defined category ID
                category_id = self.model_class_to_category_id.get(class_id)

                if category_id is not None:
                     # Create a preliminary annotation object
                     obj_annotation = {
                         "id": annotation_counter, # Assign a unique ID
                         "category_id": category_id,
                         "bbox": bbox_xywh,
                         "confidence": confidence, # Keep confidence for filtering/debugging
                         "attributes": {} # Attributes will be populated later
                     }
                     detected_objects.append(obj_annotation)
                     annotation_counter += 1

        return detected_objects