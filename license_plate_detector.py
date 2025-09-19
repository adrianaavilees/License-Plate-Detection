import cv2
from ultralytics import YOLO
import easyocr
import os
import numpy as np
import torch
from pathlib import Path

class LicensePlateDetector:
    def __init__(self, model_path="license_plate_model.pt", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if os.path.exists(model_path):
            self.license_model = YOLO(model_path) 
            print(f"Loaded custom license plate model from {model_path}")
        else:
            self.license_model = YOLO("yolov8n.pt")
            print("Custom model not found, using base YOLOv8n model")
                
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        self.model_path = model_path
    
    def create_dataset(self, dataset_path, yaml_path="dataset.yaml"):
        """Create a file dataset.yaml for training the model"""
        
        config_content = f"""
        path: {dataset_path}  
        train: images/train  
        val: images/val
        test: images/test
        names: 
          0: license_plate"""
        
        with open(yaml_path, 'w') as file:
            file.write(config_content)
        print(f"Dataset configuration file created at {yaml_path}")
        return yaml_path
        
    def train_license_model(self, dataset_path, epochs=100, batch=16):
        """Train a custom model for license plate detection."""

        yaml_path = self.create_dataset(dataset_path)
        # Load base model
        model = YOLO("yolov8n.pt")

        print("Starting training...")
        # Train the model on the custom dataset
        results = model.train(data=yaml_path, epochs=epochs, batch=batch, project="license_plate_training", name="run1")

        # Save the best model
        best_model_path = Path("license_plate_training/run1/weights/best.pt")
        model.save(best_model_path)
        
        print("Training completed")

        return results
    
    def detect_license_plate(self, input_dir, output_dir):
        """Detect vehicles in the image using the model trained"""
        os.makedirs(output_dir, exist_ok=True)
        image_paths = input_dir  

        for img_path in image_paths:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Unable to load image at {image_path}")
                return [], None
            
            results = self.license_model(image) #! Podemos jugar con los parametros de confianza para mejorar resultados
        
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()

                        # Draw bounding box of the license plate (green)
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(image, f"Plate {i+1}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)    
        
        out_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, image)
    
# Main
dataset_path = r"\Repte Matriculas\BD_LicensePlate"

detector = LicensePlateDetector(model_path="license_plate_model.pt")

# Train the model
best_model = detector.train_license_model(dataset_path, epochs=10, batch=8)

test_images_dir = os.path.join(dataset_path, "images/test")
output_dir = r"/Repte Matricules/test_results"
detector.detect_license_plate(test_images_dir, output_dir)

