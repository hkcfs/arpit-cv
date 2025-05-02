import json
from src.config import config
import datetime # To generate current datetime if needed
import os # To get image file name

class JSONGenerator:
    def generate_json(self, image_path, processed_entities, image_metadata=None):
        """Generates the final JSON output."""
        # Ensure category IDs in entities match the JSON_CATEGORIES list
        # Filter out entities that don't belong to a valid JSON category if necessary

        # Assign a unique image ID (can be based on filename hash, database ID, etc.)
        # For PoC, let's use a simple hash or index
        image_id = hash(image_path) % 1000000 # Simple hash as ID

        json_output = {
            "info": {
                "description": config.JSON_INFO_DESCRIPTION
            },
            "images": [
                {
                    "id": image_id,
                    "file_name": os.path.basename(image_path),
                    "environment": image_metadata.get('environment', config.DEFAULT_ATTRIBUTES['environment']), # Set environment in main pipeline
                    "gps": image_metadata.get('gps', None),
                    "datetime": image_metadata.get('datetime', datetime.datetime.now().isoformat()) # Use detected or current time
                }
            ],
            "categories": config.JSON_CATEGORIES, # Use the predefined list from config
            "annotations": []
        }

        # Populate annotations
        # Ensure annotation IDs are unique across all objects
        annotation_counter = 1 # Re-sequence IDs for the final JSON if needed, or use the IDs assigned during detection/OCR
        final_annotation_id_map = {} # Map internal processing IDs to final JSON IDs

        for entity in processed_entities:
            # Create the annotation object for JSON
            # Only include attributes that are not None and are part of the expected schema for that category
            json_attributes = {}
            # This part requires carefully mapping attributes from the processed entity
            # to the final JSON schema based on category.
            # Iterate through default attributes or a defined schema for each category
            # For simplicity in this placeholder, we'll just copy and clean up

            # Clean up None values and potentially filter attributes
            for key, value in entity['attributes'].items():
                 # Only include attributes defined in the JSON schema for this category if you have a strict schema validator
                 # For this PoC, let's include all non-None attributes
                 if value is not None:
                      json_attributes[key] = value

            # Ensure linking attributes use the *final* JSON annotation IDs
            # This requires a second pass or mapping IDs. For simplicity now,
            # we'll assume the IDs assigned earlier are the final ones, but
            # a robust system needs to manage ID spaces carefully.
            # If re-sequencing IDs:
            # Map parent IDs:
            # if 'parent_product_id' in json_attributes:
            #     json_attributes['parent_product_id'] = final_annotation_id_map.get(json_attributes['parent_product_id'], None)
            # ... similar for other linking attributes

            json_annotation = {
                "id": entity['id'], # Using the ID assigned during detection/OCR for now
                "image_id": image_id,
                "category_id": entity['category_id'],
                "bbox": entity['bbox'],
                "attributes": json_attributes
            }
            json_output['annotations'].append(json_annotation)

            # if re-sequencing: final_annotation_id_map[entity['id']] = annotation_counter; annotation_counter += 1


        return json_output

    def save_json(self, json_data, output_path):
        """Saves the generated JSON data to a file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON output saved to {output_path}")
