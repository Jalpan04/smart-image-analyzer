import os
import sys
import time
from src import utils, preprocess, detect, visualize, evaluate

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Smart Image Analyzer")
    parser.add_argument("--image", type=str, default=None, help="Path to input image or URL.")
    parser.add_argument("--input-dir", type=str, default=None, help="Path to directory of images to process.")
    parser.add_argument("--use-denoised", action="store_true", help="Use denoised image for detection instead of original.")
    parser.add_argument("--download-coco", action="store_true", help="Download and run on COCO128 dataset.")
    return parser.parse_args()

def main():
    # 0. Parse Args
    args = parse_args()

    # 1. Setup
    logger = utils.setup_logging()
    logger.info("Starting Smart Image Analyzer...")
    
    # 2. Device Config
    device = utils.get_device()
    
    # Imports here to avoid circular or early import issues if needed
    from src import dataset

    MODELS_DIR = os.path.join("smart_image_analyzer", "models")
    OUTPUT_DIR = os.path.join("smart_image_analyzer", "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load Model once
    logger.info("Loading YOLO model...")
    model = detect.load_model('yolov8n.pt') 
    detect.print_model_details(model)

    if args.download_coco:
        # Download and Run on Dataset
        logger.info("Downloading COCO128 Dataset...")
        dataset_path = dataset.download_coco128()
        images_dir = os.path.join(dataset_path, "images", "train2017")
        
        # Get list of images
        image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        logger.info(f"Found {len(image_files)} images in COCO128. Running inference on first 5 for demo...")
        
        # Run on a few
        for img_path in image_files[:5]:
            logger.info(f"Processing {img_path}...")
            image = preprocess.load_image(img_path)
            results = detect.detect_objects(image, model, device)
            annotated_image = visualize.draw_detections(image, results)
            
            # unique filename
            base_name = os.path.basename(img_path)
            visualize.save_image(annotated_image, OUTPUT_DIR, filename=f"detected_{base_name}")
            evaluate.evaluate_detections(results)
            
        logger.info(f"Check {OUTPUT_DIR} for results.")
        return

    # Single Image Mode (Default or Custom)
    if args.image:
        input_source = args.image
    elif not args.download_coco:
        # Ask user for input if no specific mode is selected
        print("\n" + "="*40)
        print("ENTER IMAGE SOURCE")
        print("Leave empty for default sample.")
        print("Or enter a path (e.g. C:/img.jpg)")
        print("Or enter a URL (e.g. https://site.com/img.jpg)")
        print("="*40)
        user_input = input(">> ").strip()
        if user_input:
            input_source = user_input
        else:
             input_source = None
    else:
        input_source = None

    if input_source:
        if input_source.startswith(('http://', 'https://')):
            logger.info(f"Detected URL: {input_source}")
            local_path = utils.download_image(input_source)
            if local_path:
                IMAGE_PATH = local_path
            else:
                logger.error("Could not download image from URL.")
                return
        else:
            IMAGE_PATH = input_source
            logger.info(f"Using user provided image: {IMAGE_PATH}")
    else:
        IMAGE_PATH = os.path.join("smart_image_analyzer", "data", "sample.jpg")
    
    # 3. Load & Preprocess
    logger.info(f"Loading image from {IMAGE_PATH}")
    try:
        image = preprocess.load_image(IMAGE_PATH)
    except FileNotFoundError as e:
        logger.error(e)
        return

    # Resize image to a standard size for consistency (Classic CV requirement)
    # YOLOv8 works well with 640x640.
    logger.info("Resizing image to 640x640 using OpenCV...")
    image = preprocess.resize_image(image, target_size=(640, 640))

    logger.info("Running preprocessing pipeline...")
    preprocessed_data = preprocess.preprocess_pipeline(image)
    visualize.save_preprocessing_stages(preprocessed_data, OUTPUT_DIR)
    logger.info("Preprocessing artifacts (Edges, Denoised, Grayscale) saved to outputs/.")

    # Decide which image to use for detection
    if args.use_denoised:
        logger.warning("Using scikit-image denoised output (grayscale) for YOLO.")
        image_to_detect = cv2.cvtColor(preprocessed_data['denoised'], cv2.COLOR_GRAY2BGR)
    else:
        image_to_detect = image

    logger.info("Running inference...")
    
    start_time = time.time()
    results = detect.detect_objects(image_to_detect, model, device)
    end_time = time.time()
    
    inference_time_ms = (end_time - start_time) * 1000
    fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
    
    logger.info(f"Inference Time: {inference_time_ms:.2f} ms")
    logger.info(f"FPS: {fps:.2f}")

    # 5. Evaluation
    evaluate.evaluate_detections(results)
    
    # 6. Visualization
    logger.info("Visualizing results...")
    annotated_image = visualize.draw_detections(image_to_detect, results)
    visualize.save_image(annotated_image, OUTPUT_DIR)
    
    # 7. Portfolio Artifact
    # Generate a nice summary text
    num_objects = len(results[0].boxes)
    stats_text = f"""
    Device: {device}
    Model: YOLOv8n (COCO)
    Objects Detected: {num_objects}
    Inference: {inference_time_ms:.1f} ms
    FPS: {fps:.1f}
    resolution: {image_to_detect.shape[1]}x{image_to_detect.shape[0]}
    """
    visualize.create_portfolio_artifact(image, annotated_image, stats_text, os.path.join(OUTPUT_DIR, "portfolio_report.jpg"))
    
    # 7. Model Export
    logger.info("Exporting model to ONNX...")
    try:
        # detect.export_to_onnx returns the path
        onnx_path = detect.export_to_onnx(model) # This usually saves as yolov8n.onnx in current dir or model dir
        
        if onnx_path and os.path.exists(onnx_path):
            target_path = os.path.join(MODELS_DIR, "export.onnx")
            os.replace(onnx_path, target_path)
            logger.info(f"Model moved to {target_path}")
    except Exception as e:
        logger.error(f"Export failed: {e}")
    except Exception as e:
        logger.error(f"Export failed: {e}")

    logger.info("Smart Image Analyzer finished successfully.")

if __name__ == "__main__":
    main()
