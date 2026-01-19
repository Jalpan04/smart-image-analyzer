from collections import Counter
import torch

def evaluate_detections(results):
    """
    Evaluates and prints statistics about detected objects.
    
    Args:
        results (list): List of YOLO results.
    """
    result = results[0]  # Single image inference
    
    boxes = result.boxes
    if len(boxes) == 0:
        print("\nNo objects detected.")
        return
        
    class_indices = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()
    names = result.names
    
    # 1. Total objects
    total_objects = len(boxes)
    
    # 2. Per class counts
    class_counts = Counter([names[idx] for idx in class_indices])
    
    # 3. Average confidence
    avg_confidence = confidences.mean()
    
    print("\n" + "="*30)
    print("Evaluation Results")
    print("="*30)
    print(f"Total Objects Detected: {total_objects}")
    print(f"Average Confidence:     {avg_confidence:.4f}")
    print("-" * 30)
    print("Class Counts:")
    for cls_name, count in class_counts.items():
        print(f"  - {cls_name}: {count}")
    print("="*30 + "\n")
