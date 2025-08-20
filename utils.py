import numpy as np
from PIL import Image
import base64
import io

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Basic image preprocessing without OpenCV"""
    # Simple grayscale conversion using PIL instead of OpenCV
    if len(image.shape) == 3:
        # Convert to PIL Image for processing
        pil_image = Image.fromarray(image)
        # Convert to grayscale
        gray_image = pil_image.convert('L')
        return np.array(gray_image)
    return image

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def bytes_to_image(file_bytes: bytes) -> Image.Image:
    """Convert bytes to PIL Image"""
    return Image.open(io.BytesIO(file_bytes))

def create_bbox(x: int, y: int, w: int, h: int):
    """Create BoundingBox from coordinates and dimensions"""
    # Import here to avoid circular import
    from schemas import BoundingBox
    return BoundingBox(x1=x, y1=y, x2=x+w, y2=y+h)
