# Helper functions for managing the annotation data structure before JSON generation
import json # Might be needed later, good practice to import if functions interact with json structures

def create_annotation_object(annotation_id, image_id, category_id, bbox_xywh, attributes=None):
    """Creates a basic annotation dictionary."""
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox_xywh,
        "attributes": attributes if attributes is not None else {}
    }
    return annotation

def add_attribute_to_annotation(annotation, key, value):
    """Adds or updates an attribute in an annotation dictionary."""
    annotation['attributes'][key] = value

def get_annotations_by_category(annotations_list, category_name, config):
    """Filters a list of annotations by category name."""
    category_id = config.CLASS_TO_CATEGORY_ID.get(category_name)
    if category_id is None:
        # Handle case where category_name might not be in the mapping
        return []
    return [ann for ann in annotations_list if ann.get('category_id') == category_id] # Use .get for robustness

def get_annotation_by_id(annotations_list, annotation_id):
    """Finds an annotation by its ID."""
    for ann in annotations_list:
        if ann['id'] == annotation_id:
            return ann
    return None

# Add this function:
def get_category_name_from_id(category_id, config):
    """Looks up the category name from its ID using the config."""
    # Ensure category_id is not None and is in the expected range if needed
    if category_id is None:
        return "Unknown Category"

    # Iterate through the categories list in config to find the name
    for cat in config.JSON_CATEGORIES:
        if cat['id'] == category_id:
            return cat['name']

    # Return a default if the ID is not found in the categories list
    return "Unknown Category ID" # More specific unknown message