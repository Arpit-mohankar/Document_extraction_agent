import cv2
import numpy as np
from PIL import Image
import base64
import io

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Basic image preprocessing for better OCR"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Threshold
    thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    return thresh

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
