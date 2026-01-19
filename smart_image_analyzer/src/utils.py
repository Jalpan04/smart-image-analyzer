import torch
import logging
import sys

def setup_logging():
    """Sets up the logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def get_device():
    """
    Automatically detects and returns the GPU device if available, else CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print device info as required
    print("-" * 30)
    print(f"Device Setup")
    print(f"Detected Device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
    else:
        print("CUDA is NOT available. Using CPU.")
    print("-" * 30)
    
    return device

def download_image(url, save_dir="smart_image_analyzer/data"):
    """
    Downloads an image from a URL.
    """
    import urllib.request
    import os
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    filename = url.split('/')[-1]
    # Handle potential query params in url
    if '?' in filename:
        filename = filename.split('?')[0]
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        filename = "downloaded_image.jpg"
        
    save_path = os.path.join(save_dir, filename)
    
    print(f"Downloading image from {url}...")
    try:
        # User agent might be needed for some sites
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, save_path)
        print(f"Saved to {save_path}")
        return save_path
    except Exception as e:
        print(f"Failed to download: {e}")
        return None
