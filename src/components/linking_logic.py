from src.config import config
from src.utils.bbox_utils import is_contained, calculate_distance, calculate_iou
from src.utils.annotation_utils import get_annotations_by_category, get_annotation_by_id
from src.components.attribute_extractor import AttributeExtractor
import numpy as np

class EntityLinker:
    def __init__(self):
        pass # Initialize linking specific parameters if needed

    def perform_linking(self, entities):
        """
        Performs entity linking based on spatial relationships and entity types.
        Modifies entities list in place by adding linking attributes.
        """
        # Create lookup tables for easier access
        entities_by_category = {
             cat_name: get_annotations_by_category(entities, cat_name, config)
             for cat_name in config.CLASS_TO_CATEGORY_ID.keys() # Use keys for names
        }
        entity_lookup = {entity['id']: entity for entity in entities}


        # --- Linking SKUs to Products ---
        skus = entities_by_category.get('sku', [])
        products = entities_by_category.get('product', [])
        for sku in skus:
            best_parent_product_id = None
            highest_iou = 0
            for product in products:
                # Check if the SKU is contained within the product bbox
                iou = calculate_iou(sku['bbox'], product['bbox'])
                # A high IoU overlap is a strong indicator of containment/association
                if iou > highest_iou and is_contained(sku['bbox'], product['bbox'], config.IOU_THRESHOLD_CONTAINMENT):
                    highest_iou = iou
                    best_parent_product_id = product['id']

            if best_parent_product_id is not None:
                sku['attributes']['parent_product_id'] = best_parent_product_id

        # --- Linking Brand Ambassadors to POSM ---
        brand_ambassadors = entities_by_category.get('brand_ambassador', [])
        posms = entities_by_category.get('posm', [])
        for ambassador in brand_ambassadors:
            best_associated_posm_id = None
            highest_iou = 0
            for posm in posms:
                 # Check if the ambassador is on the POSM (high IoU overlap)
                 iou = calculate_iou(ambassador['bbox'], posm['bbox'])
                 if iou > highest_iou and iou > config.IOU_THRESHOLD_CONTAINMENT: # Require significant overlap
                     highest_iou = iou
                     best_associated_posm_id = posm['id']

            if best_associated_posm_id is not None:
                 ambassador['attributes']['associated_posm_id'] = best_associated_posm_id
                 # Maybe inherit brand_name from the POSM if linked?
                 # linked_posm = entity_lookup.get(best_associated_posm_id)
                 # if linked_posm and 'brand_name' in linked_posm['attributes']:
                 #      ambassador['attributes']['brand_association'] = linked_posm['attributes']['brand_name']


        # --- Linking Text to Entities (Store, POSM, Product, SKU) ---
        texts = entities_by_category.get('text', [])
        linkable_entities = [e for e in entities if e['category_id'] in [
            config.CLASS_TO_CATEGORY_ID.get('store_facade'),
            config.CLASS_TO_CATEGORY_ID.get('store_interior'),
            config.CLASS_TO_CATEGORY_ID.get('posm'),
            config.CLASS_TO_CATEGORY_ID.get('product'),
            config.CLASS_TO_CATEGORY_ID.get('sku')
        ]]

        attribute_extractor = AttributeExtractor() # Need this for text type classification

        for text_ann in texts:
            best_linked_entity_id = None
            min_distance = float('inf')
            best_iou = 0
            best_linked_category_name = None

            # Strategy: Prioritize containment, then proximity, then content matching
            # 1. Check for containment (e.g., text inside a button on a POSM)
            for entity in linkable_entities:
                iou = calculate_iou(text_ann['bbox'], entity['bbox'])
                if iou > best_iou and iou > config.IOU_THRESHOLD_CONTAINMENT:
                    best_iou = iou
                    best_linked_entity_id = entity['id']
                    # Get the category name of the potential parent for text type classification
                    best_linked_category_name = [name for name, id in config.CLASS_TO_CATEGORY_ID.items() if id == entity['category_id']][0] # Get name from ID


            # 2. If not contained, check proximity to the closest entity
            if best_linked_entity_id is None:
                 for entity in linkable_entities:
                     distance = calculate_distance(text_ann['bbox'], entity['bbox'])
                     if distance < min_distance and distance < config.PROXIMITY_THRESHOLD_PIXELS:
                          min_distance = distance
                          best_linked_entity_id = entity['id']
                          best_linked_category_name = [name for name, id in config.CLASS_TO_CATEGORY_ID.items() if id == entity['category_id']][0]

            # Assign the best linked entity ID
            if best_linked_entity_id is not None:
                text_ann['attributes']['linked_entity'] = best_linked_entity_id

                # 3. Classify text type based on content and linked entity category
                text_content = text_ann['attributes']['text_content']
                text_ann['attributes']['text_type'] = attribute_extractor.classify_text_type(
                    text_content,
                    linked_entity_category=best_linked_category_name # Pass parent category hint
                )

                # 4. Propagate attributes FROM text TO parent entity based on text type
                # Example: If text_type is 'store_name_text' and linked to a store_facade,
                # update the store_facade's 'store_name' attribute.
                linked_entity = entity_lookup.get(best_linked_entity_id)
                if linked_entity:
                    text_type = text_ann['attributes'].get('text_type')
                    text_content = text_ann['attributes'].get('text_content')
                    parent_category_name = [name for name, id in config.CLASS_TO_CATEGORY_ID.items() if id == linked_entity['category_id']][0]

                    if text_type == 'store_name_text' and parent_category_name in ['store_facade', 'store_interior']:
                        linked_entity['attributes']['store_name'] = text_content
                    elif text_type in ['brand_slogan', 'offer_text'] and parent_category_name == 'posm':
                        # Decide whether this text is the main tagline or a separate offer text
                        # Simple: Append or overwrite
                        if 'tagline_text' not in linked_entity['attributes'] or not linked_entity['attributes']['tagline_text']:
                             linked_entity['attributes']['tagline_text'] = text_content
                         # More complex logic needed here for multiple texts on one POSM
                    elif text_type == 'price_text' and parent_category_name in ['sku', 'product']:
                         # Update price attribute on the parent SKU/Product
                         linked_entity['attributes']['price'] = text_content
                    elif text_type == 'spec_text' and parent_category_name in ['sku', 'product']:
                         # Update specifications attribute on the parent SKU/Product
                         # Could combine multiple spec texts
                         current_specs = linked_entity['attributes'].get('specifications')
                         if current_specs:
                             linked_entity['attributes']['specifications'] = f"{current_specs}, {text_content}"
                         else:
                             linked_entity['attributes']['specifications'] = text_content
                    # Add more rules for linking brand names from text to POSM/Product etc.


        # --- Linking POSM/Product/Ambassador to Store ---
        stores = entities_by_category.get('store_facade', []) + entities_by_category.get('store_interior', [])
        top_level_entities = posms + products + brand_ambassadors

        for entity in top_level_entities:
            best_parent_store_id = None
            highest_overlap = 0 # Using overlap percentage or area might be better than IoU if store bbox is huge

            # Find the store that contains or is closest to the entity
            for store in stores:
                 # Check for containment first
                 if is_contained(entity['bbox'], store['bbox'], iou_threshold=0.1): # Lower threshold for larger store bbox
                     best_parent_store_id = store['id']
                     break # Assume it belongs to the containing store

            # If not contained in any, link to the closest store (if within a reasonable distance)
            if best_parent_store_id is None:
                 min_distance = float('inf')
                 for store in stores:
                      distance = calculate_distance(entity['bbox'], store['bbox'])
                      # You might add a max distance threshold here
                      if distance < min_distance: # and distance < SOME_MAX_DISTANCE:
                          min_distance = distance
                          best_parent_store_id = store['id']

            if best_parent_store_id is not None:
                entity['attributes']['parent_store_id'] = best_parent_store_id


        # --- Final Attribute Consolidation (e.g., Brand Name) ---
        # Decide which source is most reliable for brand name (model classification, linked text)
        # Iterate through POSMs and Products again
        for entity in posms + products:
             linked_brand_name = None
             # 1. Check linked text first (often most accurate if present)
             linked_texts = [t for t in texts if t['attributes'].get('linked_entity') == entity['id'] and t['attributes'].get('text_type') in ['brand_name_text', 'brand_slogan']]
             if linked_texts:
                  # Pick one? Combine? Simple: take the content of the first linked text
                  linked_brand_name = linked_texts[0]['attributes']['text_content']

             # 2. If no linked text, use model classification
             if linked_brand_name is None and entity['category_id'] in [config.CLASS_TO_CATEGORY_ID['posm'], config.CLASS_TO_CATEGORY_ID['product']]:
                 # Assuming model classification directly provides brand name or a class that maps to it
                 # This requires your YOLO model training to include brand classes or have a mapping
                 model_classified_brand = "BrandFromModel" # Placeholder: Replace with actual extraction from entity/model output
                 if model_classified_brand != "Unknown": # Or check confidence
                     linked_brand_name = model_classified_brand

             # Assign the determined brand name
             if linked_brand_name:
                 entity['attributes']['brand_name'] = linked_brand_name


        # The entities list has now been modified in place with linking attributes.
        return entities # Return the modified list

    # Potential future methods:
    # def perform_vlm_linking(self, img, entities):
    #    """Use a VLM like LLaVA to refine linking based on visual/textual context."""
    #    # This would be significantly more complex, involving prompts to the VLM
    #    # and parsing its unstructured (or semi-structured) output.
    #    pass
