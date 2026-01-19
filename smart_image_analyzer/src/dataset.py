import os
from ultralytics.utils.downloads import download
from pathlib import Path

def download_coco128(download_dir='smart_image_analyzer/data'):
    """
    Downloads the COCO128 dataset (128 images, labels).
    
    Args:
        download_dir (str): Directory where data should be stored.
    
    Returns:
        str: Path to the dataset directory.
    """
    # URL for COCO128 zip
    url = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip'
    
    save_dir = Path(download_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading COCO128 dataset to {save_dir}...")
    # Ultralytics helper or just manual unzip. 
    # The 'download' helper handles zip extraction automatically.
    download(url, dir=save_dir, unzip=True)
    
    dataset_path = save_dir / 'coco128'
    print(f"Dataset downloaded to {dataset_path}")
    return str(dataset_path)
