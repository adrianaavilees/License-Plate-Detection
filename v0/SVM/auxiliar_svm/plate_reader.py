import cv2
import joblib
import numpy as np
from .cropper import PlatePredictor
from .segmentator import Segmentator
import os


class PlateReader:
    def __init__(self, model_path, svm_digits_path, svm_letters_path):
        self.predictor = PlatePredictor(model_path)
        self.clf_digits = joblib.load(svm_digits_path)
        self.clf_letters = joblib.load(svm_letters_path)

    def preprocess_char(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (32, 32)).flatten() / 255.0
        return img
    
    def predict_plate(self, image_path, save_dir=None):  
        results, crop = self.predictor.predict_and_crop_image(image_path, conf=0.40, save_dir=save_dir)
        if crop is None:
            print("❌ No se pudo detectar ninguna matrícula en la imagen.")
            return None

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image_save_dir = os.path.join(save_dir, base_name)
        os.makedirs(image_save_dir, exist_ok=True)

        crop_path = os.path.join(image_save_dir, f"{base_name}_plate.jpg")
        cv2.imwrite(crop_path, crop)
        print(f"✅ Matrícula recortada guardada en {crop_path}")

        segmentator = Segmentator(crop)
        numbers, letters = segmentator.segment_characters()



        predicted_numbers = []
        for i, img in enumerate(numbers):
            processed = self.preprocess_char(img)
            pred = self.clf_digits.predict([processed])[0]
            predicted_numbers.append(str(pred))

            cv2.imwrite(os.path.join(image_save_dir, f"num_{i}_{pred}.jpg"), img)


        predicted_letters = []
        for i, img in enumerate(letters):
            processed = self.preprocess_char(img)
            pred = self.clf_letters.predict([processed])[0]
            predicted_letters.append(str(pred))

            cv2.imwrite(os.path.join(image_save_dir, f"let_{i}_{pred}.jpg"), img)

        plate_str = "".join(predicted_numbers + predicted_letters)
        print(f"✅ Matrícula predicha: {plate_str}")

        with open(os.path.join(image_save_dir, "prediction.txt"), "w") as f:
            f.write(f"Predicción: {plate_str}\n")
            f.write(f"Números: {predicted_numbers}\n")
            f.write(f"Letras: {predicted_letters}\n")

        return plate_str
