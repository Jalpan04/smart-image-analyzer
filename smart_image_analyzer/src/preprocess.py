import cv2
import numpy as np
from skimage import color, restoration, feature, img_as_ubyte

def load_image(image_path):
    """
    Loads an image using OpenCV.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        numpy.ndarray: Loaded image in BGR format (OpenCV default).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    return img

def resize_image(image, target_size=(640, 640)):
    """
    Resizes image using OpenCV.
    
    Args:
        image (numpy.ndarray): Input image.
        target_size (tuple): Target size (width, height).
        
    Returns:
        numpy.ndarray: Resized image.
    """
    return cv2.resize(image, target_size)

def preprocess_pipeline(image_bgr):
    """
    Demonstrates Scikit-image preprocessing steps.
    
    Args:
        image_bgr (numpy.ndarray): Input image in BGR format.
        
    Returns:
        dict: Dictionary containing processed image stages.
    """
    # Convert to RGB for skimage compatibility
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # 1. Grayscale Conversion (scikit-image)
    gray_image = color.rgb2gray(image_rgb)
    
    # 2. Denoising (scikit-image)
    # Using Total Variation denoising as an example of advanced denoising
    denoised_image = restoration.denoise_tv_chambolle(gray_image, weight=0.1)
    
    # 3. Edge Detection (scikit-image)
    # Using Canny edge detector
    edges = feature.canny(denoised_image, sigma=2.0)
    
    return {
        "original_rgb": image_rgb,
        "grayscale": img_as_ubyte(gray_image),
        "denoised": img_as_ubyte(denoised_image),
        "edges": img_as_ubyte(edges)
    }
