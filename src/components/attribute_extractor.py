from src.config import config
# Potentially import NLP libraries like spaCy or NLTK for text analysis

class AttributeExtractor:
    def extract_attributes(self, detected_entities):
        """
        Extracts and populates initial attributes based on category and model output.
        Does NOT handle linking yet.
        """
        processed_entities = []

        for entity in detected_entities:
            # Start with default attributes for the category
            attributes = config.DEFAULT_ATTRIBUTES.copy()

            # Populate attributes based on category and available information
            category_id = entity['category_id']
            bbox = entity['bbox']
            confidence = entity.get('confidence', 1.0) # Use confidence if available

            # --- Basic Attribute Population based on Category ---
            if category_id == config.CLASS_TO_CATEGORY_ID.get('store_facade') or category_id == config.CLASS_TO_CATEGORY_ID.get('store_interior'):
                # Attributes like store_type, chain_status ideally come from the model's classification
                # For PoC, you might map detected class name directly or use simple rules
                attributes['store_type'] = 'Grocery' # Placeholder
                attributes['chain_status'] = 'Independent' # Placeholder
                # environment will be set in the main pipeline logic based on dominant category

            elif category_id == config.CLASS_TO_CATEGORY_ID.get('posm'):
                attributes['posm_type'] = 'billboard' # Placeholder - get from model output if classified
                # brand_name will be populated after linking text
                # size_dimensions, color_scheme, illumination, condition - require specific logic (manual, other models, or simple rules)
                attributes['size_dimensions'] = 'Unknown'
                attributes['color_scheme'] = 'Unknown'
                attributes['illumination'] = 'Unknown'
                attributes['condition'] = 'Unknown'
                 # tagline_text will be populated after linking text


            elif category_id == config.CLASS_TO_CATEGORY_ID.get('product'):
                attributes['product_category'] = 'beverage' # Placeholder - get from model output
                # brand_name will be populated after linking text or using visual features
                attributes['color'] = 'Unknown' # Requires color analysis or linked text
                # promo_offer requires linked text analysis
                attributes['promo_offer'] = None


            elif category_id == config.CLASS_TO_CATEGORY_ID.get('sku'):
                attributes['sku_model'] = 'Unknown SKU' # Placeholder - get from model output or linked text
                attributes['variant'] = 'Unknown' # Requires linked text
                # price, specifications require linked text

            elif category_id == config.CLASS_TO_CATEGORY_ID.get('brand_ambassador'):
                 attributes['ambassador_name'] = 'Unknown_Celebrity' # Requires face recognition/classification
                 attributes['brand_association'] = 'Unknown Brand' # Requires linking to POSM/Product and analyzing text


            elif category_id == config.CLASS_TO_CATEGORY_ID.get('text'):
                 # text_content and language already extracted by OCRProcessor
                 attributes['text_content'] = entity['attributes'].get('text_content', '')
                 attributes['language'] = entity['attributes'].get('language', 'Unknown')
                 # text_type and linked_entity will be populated during linking


            # Update entity with populated attributes
            entity['attributes'] = attributes
            processed_entities.append(entity)

        return processed_entities

    def classify_text_type(self, text_content, linked_entity_category=None):
        """
        Heuristic or rule-based classification of text type.
        Can use regex, keyword matching, or look at the linked entity category.
        """
        text_content_lower = text_content.lower()

        if '$' in text_content or 'rs.' in text_content_lower or '£' in text_content:
            return 'price_text'
        elif 'off' in text_content_lower or '%' in text_content or 'deal' in text_content_lower:
             return 'offer_text'
        elif linked_entity_category == 'store_facade' or linked_entity_category == 'store_interior':
             # More checks might be needed to confirm it's the main name
             return 'store_name_text'
        elif linked_entity_category == 'posm':
             # Could be slogan or offer depending on content
             if len(text_content) > 10: # Simple heuristic
                  return 'brand_slogan'
             else:
                  return 'offer_text' # Could be short offer text on POSM
        elif linked_entity_category == 'sku' or linked_entity_category == 'product':
             # Check for common specification keywords or number patterns
             if 'mp camera' in text_content_lower or 'gb' in text_content_lower or 'sugar-free' in text_content_lower:
                 return 'spec_text'
             else:
                 return 'unknown_text' # Or try to match with known product/sku names

        # Add more rules based on expected text types
        return 'unknown_text' # Default if no rule matches

    # You might add methods here to extract color, analyze image patches etc.

# ... (previous functions) ...

def get_category_name_from_id(category_id, config):
    """Looks up the category name from its ID using the config."""
    for cat in config.JSON_CATEGORIES:
        if cat['id'] == category_id:
            return cat['name']
    return "Unknown Category"
