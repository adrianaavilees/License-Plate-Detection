import os
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
# from google.colab.patches import cv2_imshow

class LicensePlateDetector:
    def __init__(self, model_path="best_model.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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

    def train_model(self, dataset_path, epochs=10, batch=8, project="license_plate_training"):
        """Train a custom model for license plate detection."""

        yaml_path = self.create_dataset(dataset_path)
        
        print("Starting training...")

        # Load base model
        model = YOLO("yolov8n.pt")
        # Train the model on the custom dataset
        results = model.train(data=yaml_path, epochs=epochs, batch=batch, project=project, name="run1")

        # Route where yolo automatically saves the model
        best_model_path = Path(project) / "run1" / "weights" / "best.pt"
    
        print("Training completed")

        return str(best_model_path)

    def detect_license_plate(self, best_model_path,test_dir, output_dir, conf=0.25):
        """Detect vehicles in the image using the model trained"""
        os.makedirs(output_dir, exist_ok=True)
        best_model = YOLO(best_model_path)
        print(f"Detecting license plates with model {best_model_path}")
        
        best_model.predict(source=test_dir, save=True, project=output_dir, name="results", conf=conf)
        print("Detection completed")
    
    def show_results(self, output_dir):
      """Show the results of the detection"""
      results_dir = os.path.join(output_dir, "results")
      saved_files = [f for f in os.listdir(results_dir) if f.lower().endswith((".jpg", ".png", ".jpeg" ))]
      
      if not saved_files:
          print("No images found")
  
      for file_name in saved_files:
          img_path = os.path.join(results_dir, file_name)
          image = cv2.imread(img_path)
          if image is not None:
              cv2.imshow(f"Result - {file_name}", image)
              cv2.waitKey(0)
              cv2.destroyAllWindows()
          else:
              print(f"Could not read image: {img_path}")

# ================= MAIN =================

dataset_path = "/content/BD/BD_LicensePlate"
test_dir = "/content/BD/BD_LicensePlate/images/test"
output_dir = "/content/test_results"

detector = LicensePlateDetector()

# 1) TRAIN and save the model. Only if we don't have the model already
best_model_path = detector.train_model(dataset_path, epochs=10, batch=8)

# 2) LOAD the model trained and detect license plates
best_model_path = "/content/license_plate_training/best.pt"  # Ruta donde se ha guardado
detector.detect_license_plate(best_model_path, test_dir, output_dir, conf=0.25)

# 3) SHOW results
detector.show_results(output_dir)