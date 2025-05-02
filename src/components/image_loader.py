import cv2

class ImageLoader:
    def load_image(self, image_path):
        """Loads an image from a file path using OpenCV."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        return img

    def preprocess_image(self, img):
        """Placeholder for image preprocessing (resizing, normalization, etc.)."""
        # Example: Resize for consistent model input size
        # img = cv2.resize(img, (640, 640))
        # Example: Convert color space
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # YOLOv8 expects RGB

        # Depending on your chosen models, add necessary preprocessing here.
        # YOLOv8 handles normalization internally. PaddleOCR might need specific formatting.
        return img

    def get_image_metadata(self, image_path):
        """Extracts metadata (like GPS, datetime) if available in image file."""
        metadata = {}
        # Implement logic to read EXIF data or other metadata formats
        # For PoC, you might hardcode or pass this separately if not in image
        # Example: Reading GPS from EXIF (requires libraries like exifread)
        # import exifread
        # try:
        #     with open(image_path, 'rb') as f:
        #         tags = exifread.process_file(f)
        #         if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
        #             # Need to convert EXIF format to degrees
        #             # metadata['gps'] = {'lat': ..., 'lng': ...}
        #         if 'EXIF DateTimeOriginal' in tags:
        #              metadata['datetime'] = str(tags['EXIF DateTimeOriginal']) # Needs ISO format conversion
        # except Exception as e:
        #     print(f"Could not read metadata from {image_path}: {e}")

        return metadata
