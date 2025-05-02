from paddleocr import PaddleOCR # No need for build_model unless using custom arch
import cv2
from src.config import config
from src.utils.bbox_utils import xywh_to_xyxy, xyxy_to_xywh # PaddleOCR output might be polygon or xyxy

class OCRProcessor:
    def __init__(self):
        # Initialize PaddleOCR
        # Set use_gpu=False to explicitly use the CPU
        # lang='en' for English, add more languages as needed in a list
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False) # set use_gpu=False

    def process_image(self, img):
        """Performs text detection and recognition on the image."""
        # PaddleOCR expects BGR image
        img_bgr = img # Assuming image_loader provides BGR

        # Run OCR
        # results is a list of lines, each line contains (bbox, (text, score))
        # No need to pass device here, it's configured during initialization
        ocr_results = self.ocr.ocr(img_bgr, det=True, rec=True, cls=True)

        detected_texts = []
        annotation_counter = 1 # Start IDs for text annotations

        if ocr_results and ocr_results[0]: # Check if results are not empty
            for line in ocr_results[0]: # Iterate through detected lines
                if line is not None:
                    bbox_polygon = line[0] # Bbox is a list of 4 points (polygon)
                    text, confidence = line[1]

                    # Convert polygon to bounding box [x, y, w, h] for consistency
                    x_coords = [p[0] for p in bbox_polygon]
                    y_coords = [p[1] for p in bbox_polygon]
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)
                    bbox_xywh = xyxy_to_xywh([x1, y1, x2, y2])

                    if confidence >= config.OCR_CONFIDENCE_THRESHOLD:
                        # Create a preliminary annotation object for text
                        text_annotation = {
                            "id": annotation_counter, # Assign a unique ID
                            "category_id": config.CLASS_TO_CATEGORY_ID.get('text'), # Map to 'text' category
                            "bbox": bbox_xywh,
                            "confidence": confidence, # Keep confidence
                            "attributes": {
                                "text_content": text,
                                "language": 'en' # Or detect language if needed
                            }
                        }
                        detected_texts.append(text_annotation)
                        annotation_counter += 1

        return detected_texts