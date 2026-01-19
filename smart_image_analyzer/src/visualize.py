import cv2
import os

def draw_detections(image, results):
    """
    Draws bounding boxes and labels on the image.
    
    Args:
        image (numpy.ndarray): Original image.
        results (list): List of YOLO results.
        
    Returns:
        numpy.ndarray: Annotated image.
    """
    # Create a copy to avoid modifying the original
    annotated_img = image.copy()
    
    # YOLO v8 results list - usually one result per image
    result = results[0]
    
    # Iterate over detections
    for box in result.boxes:
        # Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Get confidence and class
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{result.names[cls]} {conf:.2f}"
        
        # Draw bounding box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated_img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(annotated_img, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
    return annotated_img

def save_image(image, output_dir, filename="detected.jpg"):
    """
    Saves the image to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")
    return output_path

def save_preprocessing_stages(stages, output_dir):
    """
    Saves preprocessing stages (e.g. edges, grayscale) to disk.
    
    Args:
        stages (dict): Dictionary of images.
        output_dir (str): Directory to save them.
    """
    for name, img in stages.items():
        # Skip original rgb if you want, or save it too
        if name == "original_rgb":
            # Convert back to BGR for saving with OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        filename = f"stage_{name}.jpg"
        save_image(img, output_dir, filename)


def create_portfolio_artifact(original_img, processed_img, stats_text, output_path):
    """
    Creates a portfolio-ready image combining:
    1. Original Image
    2. Processed Image
    3. Stats Panel (FPS, Device, Count)
    """
    import numpy as np

    # Ensure consistent height
    h1, w1 = original_img.shape[:2]
    h2, w2 = processed_img.shape[:2]
    target_h = max(h1, h2)
    
    # Resize keeping aspect ratio roughly or just pad? 
    # Let's simple resize for height match
    if h1 != target_h:
        original_img = cv2.resize(original_img, (int(w1 * target_h / h1), target_h))
    if h2 != target_h:
        processed_img = cv2.resize(processed_img, (int(w2 * target_h / h2), target_h))
        
    # Create Panel for text
    panel_w = 400
    panel = np.zeros((target_h, panel_w, 3), dtype=np.uint8)
    
    # Basic Font Config
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 50
    line_height = 40
    
    # Title
    cv2.putText(panel, "Smart Analyzer", (20, y_offset), font, 1.0, (0, 255, 0), 2)
    y_offset += 60
    
    # Stats
    for line in stats_text.split('\n'):
        if not line.strip(): continue
        cv2.putText(panel, line.strip(), (20, y_offset), font, 0.6, (200, 200, 200), 1)
        y_offset += line_height

    # Combine: Original | Processed | Panel
    combined = np.hstack((original_img, processed_img, panel))
    cv2.imwrite(output_path, combined)
    print(f"Portfolio Artifact saved to {output_path}")
    return output_path
