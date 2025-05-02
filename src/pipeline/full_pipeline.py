import argparse
import cv2
import os
from src.components.image_loader import ImageLoader
from src.components.object_detector import ObjectDetector
from src.components.ocr_processor import OCRProcessor
from src.components.attribute_extractor import AttributeExtractor
from src.components.linking_logic import EntityLinker
from src.components.json_generator import JSONGenerator
from src.config import config
from src.utils.bbox_utils import xywh_to_xyxy
from src.utils.annotation_utils import get_category_name_from_id

class RetailAnnotationPipeline:
    def __init__(self):
        self.image_loader = ImageLoader()
        self.object_detector = ObjectDetector()
        self.ocr_processor = OCRProcessor()
        self.attribute_extractor = AttributeExtractor()
        self.entity_linker = EntityLinker()
        self.json_generator = JSONGenerator()

    def run(self, image_path, output_path_json): # Renamed output_path to output_path_json
        """Runs the full annotation pipeline for a single image."""
        print(f"Processing image: {image_path}")

        # 1. Load and preprocess image + get metadata
        img = self.image_loader.load_image(image_path)
        img_processed = self.image_loader.preprocess_image(img)
        image_metadata = self.image_loader.get_image_metadata(image_path)

        # Determine overall environment (simple heuristic for PoC)
        # Use a copy here too if subsequent steps modify the image (though they shouldn't in this framework)
        temp_detections = self.object_detector.detect_objects(img_processed.copy()) # Use a copy

        # Let's simplify environment detection for this PoC to rely on the main detection pass later
        image_metadata['environment'] = 'unknown' # Default, updated after main detection


        # 2. Perform Object Detection
        # Use a copy here too if subsequent steps modify the image (though they shouldn't in this framework)
        detected_objects = self.object_detector.detect_objects(img_processed.copy())
        print(f"Detected {len(detected_objects)} objects.")

        # 3. Perform OCR (on original image or processed one depending on OCR needs)
        # OCR often prefers original image quality
        detected_texts = self.ocr_processor.process_image(img.copy()) # Use a copy
        print(f"Detected {len(detected_texts)} text regions.")

        # Combine detected objects and texts into a single list for processing
        all_entities = detected_objects + detected_texts

        # Assign initial attributes (e.g., text content from OCR)
        # This step is partially done in OCRProcessor, but AttributeExtractor can refine
        all_entities = self.attribute_extractor.extract_attributes(all_entities)


        # 4. Perform Entity Linking and refine attributes based on links
        print("Performing entity linking...")
        # The linking logic modifies the list in place, so just pass all_entities
        linked_entities = self.entity_linker.perform_linking(all_entities)
        print("Linking complete.")

        # --- Add Environment Classification based on linked_entities ---
        has_facade = any(e.get('category_id') == config.CLASS_TO_CATEGORY_ID.get('store_facade') for e in linked_entities)
        has_interior = any(e.get('category_id') == config.CLASS_TO_CATEGORY_ID.get('store_interior') for e in linked_entities)

        if has_facade and not has_interior:
             image_metadata['environment'] = 'outdoor'
        elif has_interior and not has_facade:
             image_metadata['environment'] = 'indoor'
        elif has_facade and has_interior:
            image_metadata['environment'] = 'semi-outdoor' # Or 'mixed'
        # else remains 'unknown' if neither is detected


        # 5. Generate JSON Output
        json_output = self.json_generator.generate_json(image_path, linked_entities, image_metadata)

        # 6. Save JSON Output
        self.json_generator.save_json(json_output, output_path_json)

        # 7. Visualize and Save Results Image
        # Determine the output path for the annotated image
        output_dir = os.path.dirname(output_path_json)
        image_filename = os.path.basename(image_path)
        image_name, image_ext = os.path.splitext(image_filename)
        output_image_filename = f"{image_name}_annotated{image_ext}"
        output_image_path = os.path.join(output_dir, output_image_filename)

        self.visualize_results(img, linked_entities, output_image_path)


        print("Pipeline finished.")


    def visualize_results(self, img, entities, output_image_path): # Added output_image_path parameter
        """Draws bounding boxes, labels, and attributes on the image and saves it."""
        img_display = img.copy()
        height, width = img_display.shape[:2]

        # Define text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = config.VISUALIZATION_FONT_SCALE
        font_thickness = config.VISUALIZATION_FONT_THICKNESS
        line_thickness = config.VISUALIZATION_LINE_THICKNESS

        for entity in entities:
            bbox_xywh = entity['bbox']
            bbox_xyxy = xywh_to_xyxy(bbox_xywh)
            category_id = entity.get('category_id') # Use .get in case it's null/missing
            annotation_id = entity['id']
            attributes = entity['attributes']

            # Get category name and color
            category_name = get_category_name_from_id(category_id, config)
            color = config.VISUALIZATION_COLORS.get(category_id, (200, 200, 200)) # Default to grey

            # Draw bounding box
            p1 = (int(bbox_xyxy[0]), int(bbox_xyxy[1]))
            p2 = (int(bbox_xyxy[2]), int(bbox_xyxy[3]))
            cv2.rectangle(img_display, p1, p2, color, line_thickness)

            # Prepare text label
            label_lines = []
            # Line 1: Category Name and ID
            label_lines.append(f"{category_name} ({annotation_id})")

            # Add key attributes based on category
            if category_name == 'text':
                # Display text content (potentially truncated if very long)
                text_content = attributes.get('text_content', 'N/A')
                # Simple truncation for display
                if len(text_content) > 40:
                    text_content = text_content[:37] + "..."
                label_lines.append(f"Txt: \"{text_content}\"")
                text_type = attributes.get('text_type', 'Unknown')
                if text_type != 'Unknown':
                    label_lines.append(f"Type: {text_type}")
                linked_entity_id = attributes.get('linked_entity')
                if linked_entity_id is not None:
                     label_lines.append(f"-> {linked_entity_id}")


            elif category_name in ['store_facade', 'store_interior']:
                 store_name = attributes.get('store_name', 'Unknown Store')
                 if store_name != 'Unknown Store':
                     label_lines.append(f"Name: {store_name}")
                 store_type = attributes.get('store_type', 'Unknown')
                 if store_type != 'Unknown':
                     label_lines.append(f"Type: {store_type}")

            elif category_name == 'posm':
                 posm_type = attributes.get('posm_type', 'Unknown')
                 if posm_type != 'Unknown':
                     label_lines.append(f"Type: {posm_type}")
                 brand_name = attributes.get('brand_name', 'Unknown Brand')
                 if brand_name != 'Unknown Brand':
                     label_lines.append(f"Brand: {brand_name}")
                 tagline_text = attributes.get('tagline_text')
                 if tagline_text: # Only display if populated
                      # Simple truncation
                      if len(tagline_text) > 30:
                           tagline_text = tagline_text[:27] + "..."
                      label_lines.append(f"Tag: \"{tagline_text}\"")
                 parent_store_id = attributes.get('parent_store_id')
                 if parent_store_id is not None:
                      label_lines.append(f"-> Store: {parent_store_id}")


            elif category_name == 'product':
                 brand_name = attributes.get('brand_name', 'Unknown Brand')
                 if brand_name != 'Unknown Brand':
                      label_lines.append(f"Brand: {brand_name}")
                 product_category = attributes.get('product_category', 'Unknown')
                 if product_category != 'Unknown':
                     label_lines.append(f"Cat: {product_category}")
                 parent_store_id = attributes.get('parent_store_id')
                 if parent_store_id is not None:
                      label_lines.append(f"-> Store: {parent_store_id}")

            elif category_name == 'sku':
                 sku_model = attributes.get('sku_model', 'Unknown SKU')
                 if sku_model != 'Unknown SKU':
                      # Simple truncation
                      if len(sku_model) > 25:
                           sku_model = sku_model[:22] + "..."
                      label_lines.append(f"Model: {sku_model}")
                 price = attributes.get('price')
                 if price:
                      label_lines.append(f"Price: {price}")
                 parent_product_id = attributes.get('parent_product_id')
                 if parent_product_id is not None:
                      label_lines.append(f"-> Prod: {parent_product_id}")


            elif category_name == 'brand_ambassador':
                 ambassador_name = attributes.get('ambassador_name', 'Unknown_Celebrity')
                 if ambassador_name != 'Unknown_Celebrity':
                     label_lines.append(f"Name: {ambassador_name}")
                 brand_association = attributes.get('brand_association', 'Unknown Brand')
                 if brand_association != 'Unknown Brand':
                     label_lines.append(f"Brand: {brand_association}")
                 associated_posm_id = attributes.get('associated_posm_id')
                 if associated_posm_id is not None:
                      label_lines.append(f"-> POSM: {associated_posm_id}")


            # Determine text position (top-left corner of the box, slightly above)
            text_x = int(bbox_xyxy[0])
            text_y = int(bbox_xyxy[1]) - 5 # 5 pixels above the box

            # Draw each line of text
            for i, line in enumerate(label_lines):
                # Get text size to calculate position for multiple lines
                (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, font_thickness)
                # Adjust y position for each line
                line_y = text_y - (len(label_lines) - 1 - i) * (text_height + baseline)

                # Ensure text is not drawn outside the image boundary (especially at the top)
                line_y = max(line_y, text_height + baseline + 5) # Ensure y is at least text height + baseline + a little padding

                # Ensure text is not drawn outside the image boundary (especially on the left)
                text_x = max(text_x, 5) # Ensure x is at least 5 pixels from the left edge


                cv2.putText(img_display, line, (text_x, line_y), font, font_scale, color, font_thickness, cv2.LINE_AA)

        # --- Save the annotated image ---
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True) # Ensure output directory exists
        cv2.imwrite(output_image_path, img_display)
        print(f"Annotated image saved to {output_image_path}")

        # Removed cv2.imshow, waitKey, destroyAllWindows
        # cv2.imshow("Retail Annotation Results", img_display_resized if height > display_height else img_display)
        # print("Displaying results. Press any key to close.")
        # cv2.waitKey(0) # Wait indefinitely for a key press
        # cv2.destroyAllWindows() # Close all OpenCV windows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retail Image Annotation Pipeline PoC")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input retail image.")
    parser.add_argument("--output_path_json", type=str, required=True, help="Path to save the output JSON file.") # Renamed argument

    args = parser.parse_args()

    pipeline = RetailAnnotationPipeline()
    pipeline.run(args.image_path, args.output_path_json) # Pass the JSON output path