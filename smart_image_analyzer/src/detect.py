from ultralytics import YOLO
import torch

def load_model(model_name='yolov8n.pt'):
    """
    Loads the YOLO model.
    """
    return YOLO(model_name)

def detect_objects(image, model, device):
    """
    Runs object detection on the image using the specified device.
    
    Args:
        image (numpy.ndarray): Input image.
        model (YOLO): Loaded YOLO model.
        device (torch.device): Device to run inference on.
        
    Returns:
        list: List of detection results (ultralytics.engine.results.Results).
    """
    # Ultralytics handles device as a string or torch.device
    print(f"Running inference on {device}...")
    results = model(image, device=device)
    return results

def export_to_onnx(model, save_dir='smart_image_analyzer/models'):
    """
    Exports the model to ONNX format.
    """
    print("Exporting model to ONNX...")
    # Ultralytics export saves to the same dir as the model usually, 
    # but we will try to direct or move it if needed. 
    # For simplicity, we let it default and then move or just let user know.
    # Actually, model.export() returns the path to the exported file.
    path = model.export(format="onnx")
    print(f"Model exported to: {path}")
    return path

def print_model_details(model):
    """
    Prints details about the model, including the class names (COCO).
    """
    print(f"Model: {model.ckpt_path}")
    print(f"Classes ({len(model.names)}):")
    # Print first few and last few to be concise, or all if user wants.
    # For 80 classes, printing all might be verbose but useful for "Dataset Awareness".
    # Let's print them in a nice formatted way.
    names = model.names
    name_list = [f"{k}: {v}" for k, v in names.items()]
    print(", ".join(name_list))

