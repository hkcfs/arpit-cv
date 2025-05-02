import argparse
import cv2 # For debugging/visualization if needed
from src.components.image_loader import ImageLoader
from src.components.object_detector import ObjectDetector
from src.components.ocr_processor import OCRProcessor
from src.components.attribute_extractor import AttributeExtractor
from src.components.linking_logic import EntityLinker
from src.components.json_generator import JSONGenerator
from src.config import config # Import config for settings like environment check

class RetailAnnotationPipeline:
    def __init__(self):
        self.image_loader = ImageLoader()
        self.object_detector = ObjectDetector()
        self.ocr_processor = OCRProcessor()
        self.attribute_extractor = AttributeExtractor()
        self.entity_linker = EntityLinker()
        self.json_generator = JSONGenerator()

    def run(self, image_path, output_path):
        """Runs the full annotation pipeline for a single image."""
        print(f"Processing image: {image_path}")

        # 1. Load and preprocess image + get metadata
        img = self.image_loader.load_image(image_path)
        img_processed = self.image_loader.preprocess_image(img)
        image_metadata = self.image_loader.get_image_metadata(image_path)

        # Determine overall environment (simple heuristic for PoC)
        # Run a quick detection pass to check for store_facade/interior
        # Or use a separate simple classifier
        temp_detections = self.object_detector.detect_objects(img_processed) # Might process twice, optimize later
        has_facade = any(d['category_id'] == config.CLASS_TO_CATEGORY_ID.get('store_facade') for d in temp_detections)
        has_interior = any(d['category_id'] == config.CLASS_TO_CATEGORY_ID.get('store_interior') for d in temp_detections)

        if has_facade and not has_interior:
             image_metadata['environment'] = 'outdoor'
        elif has_interior and not has_facade:
             image_metadata['environment'] = 'indoor'
        elif has_facade and has_interior:
            # Decide based on which is more dominant or centered
            image_metadata['environment'] = 'mixed' # Or 'semi-outdoor' if applicable
        else:
            image_metadata['environment'] = 'unknown'


        # 2. Perform Object Detection
        detected_objects = self.object_detector.detect_objects(img_processed)
        print(f"Detected {len(detected_objects)} objects.")

        # 3. Perform OCR (on original image or processed one depending on OCR needs)
        detected_texts = self.ocr_processor.process_image(img) # OCR often prefers original resolution/colors
        print(f"Detected {len(detected_texts)} text regions.")

        # Combine detected objects and texts into a single list for processing
        all_entities = detected_objects + detected_texts

        # Assign initial attributes (e.g., text content from OCR)
        # This step is partially done in OCRProcessor, but AttributeExtractor can refine
        all_entities = self.attribute_extractor.extract_attributes(all_entities)


        # 4. Perform Entity Linking and refine attributes based on links
        print("Performing entity linking...")
        linked_entities = self.entity_linker.perform_linking(all_entities)
        print("Linking complete.")

        # 5. Generate JSON Output
        json_output = self.json_generator.generate_json(image_path, linked_entities, image_metadata)

        # 6. Save JSON Output
        self.json_generator.save_json(json_output, output_path)

        print("Pipeline finished.")
        # Optional: Visualize results
        # self.visualize_results(img, linked_entities)

    # Optional: Add visualization method
    # def visualize_results(self, img, entities):
    #     """Draws bounding boxes and labels on the image."""
    #     img_display = img.copy()
    #     # Draw bboxes and labels from entities list
    #     # Use entity category and attributes for text
    #     # cv2.rectangle, cv2.putText
    #     pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retail Image Annotation Pipeline PoC")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input retail image.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output JSON file.")

    args = parser.parse_args()

    pipeline = RetailAnnotationPipeline()
    pipeline.run(args.image_path, args.output_path)
