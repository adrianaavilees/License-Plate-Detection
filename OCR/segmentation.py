import os
import cv2
import torch
from pathlib import Path
!pip install ultralytics
from ultralytics import YOLO
from google.colab.patches import cv2_imshow
from google.colab import drive
drive.mount('/drive')
import numpy as np

class LicensePlateDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def create_dataset(self, dataset_path, yaml_path="dataset.yaml"):
        """Create a file dataset.yaml for training the model"""
        train_path = os.path.join(dataset_path, 'images', 'train')
        val_path = os.path.join(dataset_path, 'images', 'val')
        test_path = os.path.join(dataset_path, 'images', 'test')

        config_content = f"""
path: {dataset_path}
train: {train_path}
val: {val_path}
test: {test_path}
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
        self.model = YOLO("yolov8n.pt")
        results = self.model.train(data=yaml_path, epochs=epochs, batch=batch, project=project)
        best_model_path = Path(results.save_dir) / "weights" / "best.pt"
        print("Training completed")
        print(f"Best model saved at: {str(best_model_path)}")
        return str(best_model_path)

    def detect_license_plate(self, best_model_path, test_dir, output_dir, conf=0.25):
        """Detect license plates in images using the trained model."""
        self.model = YOLO(best_model_path)
        print(f"Detecting license plates with model {best_model_path}")
        
        prediction_results = self.model.predict(source=test_dir, save=False, conf=conf)
        
        results_dir = os.path.join(output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        for i, result in enumerate(prediction_results):
            annotated_image = result.plot()
            output_path = os.path.join(results_dir, f"detected_{i}.jpg")
            cv2.imwrite(output_path, annotated_image)
        
        print("Detection completed. Results saved.")
        return prediction_results

    # =========================================================================
    # Character Segmentation Functions (modificat)
    # =========================================================================
    
    def show_preprocessed_image(self, image):
        """Displays the image after preprocessing steps."""
        cv2_imshow(image)
        
    def preprocess_plate_for_segmentation(self, plate_image):
        """
        Preprocesses a license plate image for character segmentation.
        - Transforms to grayscale and uses HSV to handle the blue section.
        - Applies Otsu's thresholding for binarization.
        """
        # Convert to HSV to better handle the blue band
        hsv = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)
        
        # Define range for blue color and create a mask
        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Convert to grayscale
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        
        # Paint the blue area white on the grayscale image
        gray[mask_blue > 0] = 255
        
        # Apply Otsu's thresholding to get a binary image
        _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert the image so characters are white on a black background
        otsu_thresh = cv2.bitwise_not(otsu_thresh)
            
        return otsu_thresh

    def segment_characters(self, plate_image):
        """
        Finds and segments individual characters from a preprocessed license plate image.
        - Uses cv.findContours to identify character shapes.
        - Filters contours based on size and aspect ratio to isolate characters.
        """
        preprocessed_plate = self.preprocess_plate_for_segmentation(plate_image)
        
        print("  > Showing preprocessed (grayscale + binarized) image.")
        self.show_preprocessed_image(preprocessed_plate)
        
        contours, _ = cv2.findContours(preprocessed_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        characters = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            area = cv2.contourArea(contour)
            
            # Heuristic filters to identify a character based on area and aspect ratio.
            # We have removed the position filter to detect the first character.
            if area > 100 and 0.2 < aspect_ratio < 1.0 and h > 10 and w > 5:
                character_roi = plate_image[y:y+h, x:x+w]
                characters.append({'image': character_roi, 'bbox': (x, y, w, h)})
        
        characters.sort(key=lambda c: c['bbox'][0])
        
        return characters

    def show_segmented_characters(self, characters, original_plate):
        """Displays the original license plate with bounding boxes around each character."""
        output_image = original_plate.copy()
        for char_info in characters:
            x, y, w, h = char_info['bbox']
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2_imshow(output_image)

    def process_and_show_results(self, prediction_results):
        """Processes each detected plate to segment and show the characters."""
        for i, result in enumerate(prediction_results):
            image = result.orig_img
            boxes = result.boxes
            
            if boxes:
                print(f"\n--- Processing Image {i+1} ---")
                
                for j, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    plate_image = image[y1:y2, x1:x2]
                    
                    if plate_image is not None and plate_image.size > 0:
                        print(f"  > Detected License Plate {j+1}: Bbox {x1, y1, x2, y2}")
                        
                        annotated_full_image = result.plot()
                        print("  > Showing full image with detected license plate.")
                        cv2_imshow(annotated_full_image)

                        characters = self.segment_characters(plate_image)
                        print(f"  > Found {len(characters)} character candidates.")
                        
                        if characters:
                            print("  > Showing cropped plate with segmented characters.")
                            self.show_segmented_characters(characters, plate_image)
                        else:
                            print("  > No characters found after segmentation.")
                    else:
                        print("  > Could not crop license plate. Skipping.")
            else:
                print(f"\n--- Processing Image {i+1} ---")
                print("  > No license plate detected in this image.")

# ================= MAIN EXECUTION =================

dataset_path = "/drive/MyDrive/BD_LicensePlate"
test_dir = "/drive/MyDrive/BD_LicensePlate/images/test"
output_dir = "/content/test_results"

detector = LicensePlateDetector()

best_model_path = detector.train_model(dataset_path, epochs=10, batch=8)

prediction_results = detector.detect_license_plate(best_model_path, test_dir, output_dir, conf=0.25)

detector.process_and_show_results(prediction_results)
