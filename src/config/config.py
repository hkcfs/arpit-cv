# Model Paths
YOLO_MODEL_PATH = 'models/object_detection/best.pt' # Path to your trained YOLO model
# PaddleOCR models are usually downloaded automatically, but can be configured

# Confidence Thresholds
DETECTION_CONFIDENCE_THRESHOLD = 0.3
OCR_CONFIDENCE_THRESHOLD = 0.3

# NMS Threshold
NMS_THRESHOLD = 0.4

# Class Mapping (Update with your actual training classes and IDs)
# Map category ID from JSON spec to model output class names/IDs
CATEGORY_MAPPING = {
    1: 'store_facade',
    2: 'posm',
    3: 'product',
    4: 'sku',
    5: 'brand_ambassador',
    6: 'text_region', # YOLO might detect text regions generally, then OCR reads. Or OCR handles detection directly.
    7: 'store_interior'
}

# Reverse mapping for easier lookup
CLASS_TO_CATEGORY_ID = {v: k for k, v in CATEGORY_MAPPING.items()}

# Linking Logic Thresholds (Adjust based on experiments)
IOU_THRESHOLD_CONTAINMENT = 0.7 # IoU threshold for 'contained within'
PROXIMITY_THRESHOLD_PIXELS = 20 # Max pixel distance for 'near'
# Define priority rules for linking (e.g., Text inside POSM > Text near Product)

# JSON Output Settings
JSON_INFO_DESCRIPTION = "Comprehensive Retail & Advertising Dataset PoC Output"
# Define the full list of categories for the JSON output (ID and name)
JSON_CATEGORIES = [
    {"id": 1, "name": "store_facade"},
    {"id": 2, "name": "posm"},
    {"id": 3, "name": "product"},
    {"id": 4, "name": "sku"},
    {"id": 5, "name": "brand_ambassador"},
    {"id": 6, "name": "text"},
    {"id": 7, "name": "store_interior"}
]

# Attribute Mapping and Extraction Rules
# Define how attributes are extracted based on category and linked text
ATTRIBUTE_RULES = {
    # Example rule: For 'store_facade', find linked text with type 'store_name_text'
    'store_facade': {
        'store_name': {'source': 'linked_text', 'text_type': 'store_name_text'},
        # Define extraction for other store attributes...
    },
     'posm': {
        'brand_name': {'source': 'linked_text', 'text_type': ['brand_name_text', 'brand_slogan']},
        'tagline_text': {'source': 'linked_text', 'text_type': ['brand_slogan', 'offer_text']},
        # Define extraction for other POSM attributes...
     },
     'sku': {
         'price': {'source': 'linked_text', 'text_type': 'price_text'},
         'specifications': {'source': 'linked_text', 'text_type': 'spec_text'},
         # Define extraction for other SKU attributes...
     },
     # Define rules for product, brand_ambassador, text...
}

# Default attribute values if not found
DEFAULT_ATTRIBUTES = {
    'store_id': None,
    'store_name': 'Unknown Store',
    'store_type': 'Unknown',
    'chain_status': 'Unknown',
    'location': None,
    'environment': 'Unknown',
    'posm_type': 'Unknown',
    'brand_name': 'Unknown Brand',
    'size_dimensions': 'Unknown',
    'color_scheme': 'Unknown',
    'illumination': 'Unknown',
    'condition': 'Unknown',
    'tagline_text': None,
    'product_id': None,
    'product_category': 'Unknown',
    'color': 'Unknown',
    'promo_offer': None,
    'sku_model': 'Unknown SKU',
    'variant': 'Unknown',
    'price': None,
    'specifications': None,
    'ambassador_name': 'Unknown_Celebrity',
    'brand_association': 'Unknown Brand',
    'text_content': '',
    'language': 'Unknown',
    'text_type': 'Unknown',
    'linked_entity': None, # Linking attribute
    'parent_store_id': None, # Linking attribute
    'parent_product_id': None, # Linking attribute
    'associated_posm_id': None, # Linking attribute
}

# Visualization Settings
VISUALIZATION_COLORS = {
    1: (0, 255, 0),   # Green for store_facade (BGR format)
    2: (255, 0, 0),   # Blue for posm
    3: (0, 0, 255),   # Red for product
    4: (0, 255, 255), # Yellow for sku
    5: (255, 0, 255), # Magenta for brand_ambassador
    6: (255, 165, 0), # Orange for text
    7: (128, 0, 128)  # Purple for store_interior
    # Add more colors if you map other YOLO/OCR classes temporarily
}

VISUALIZATION_FONT_SCALE = 0.5
VISUALIZATION_FONT_THICKNESS = 1
VISUALIZATION_LINE_THICKNESS = 2