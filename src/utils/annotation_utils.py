# Helper functions for managing the annotation data structure before JSON generation

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
        return []
    return [ann for ann in annotations_list if ann['category_id'] == category_id]

def get_annotation_by_id(annotations_list, annotation_id):
    """Finds an annotation by its ID."""
    for ann in annotations_list:
        if ann['id'] == annotation_id:
            return ann
    return None
