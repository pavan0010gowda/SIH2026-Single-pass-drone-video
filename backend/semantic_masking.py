import cv2
import os
import numpy as np
from ultralytics import YOLO

def mask_dynamic_objects(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the ultra-fast YOLOv8 segmentation model (it will auto-download)
    print("Loading YOLOv8 AI Model...")
    model = YOLO("yolov8n-seg.pt")
    
    # COCO Class IDs for dynamic objects we want to mask:
    # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck, 16: dog, 17: horse
    target_classes = [0, 1, 2, 3, 5, 7, 16, 17]

    print(f"Scanning frames in {input_dir}...")
    
    for filename in sorted(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(input_dir, filename)
            image = cv2.imread(filepath)
            
            # Run YOLO AI to find objects
            results = model.predict(image, verbose=False)
            result = results[0]
            
            # If the AI found a mask, check if it belongs to our target classes
            if result.masks is not None:
                for mask, box in zip(result.masks.data, result.boxes):
                    class_id = int(box.cls[0])
                    
                    if class_id in target_classes:
                        # Convert the AI mask into a pixel-perfect black silhouette
                        mask_np = mask.cpu().numpy()
                        mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
                        
                        # Paint the detected object pitch black (0, 0, 0)
                        image[mask_resized > 0.5] = [0, 0, 0]
            
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, image)
            print(f"Processed and masked: {filename}")

    print(f"Done! Masked frames saved to {output_dir}")

if __name__ == "__main__":
    INPUT_FRAMES = "../data/processed_frames/"
    OUTPUT_FRAMES = "../data/masked_frames/"
    
    mask_dynamic_objects(INPUT_FRAMES, OUTPUT_FRAMES)