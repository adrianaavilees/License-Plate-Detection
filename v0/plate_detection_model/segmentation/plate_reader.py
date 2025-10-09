import cv2
import joblib
import numpy as np
from segmentation.cropper import PlatePredictor
from segmentation.segmentator import Segmentator


class PlateReader:
    def __init__(self, model_path, svm_digits_path, svm_letters_path):
        """
        model_path: ruta al modelo YOLO (best.pt)
        svm_digits_path: ruta al modelo SVM entrenado para dígitos
        svm_letters_path: ruta al modelo SVM entrenado para letras
        """
        self.predictor = PlatePredictor(model_path)
        self.clf_digits = joblib.load(svm_digits_path)
        self.clf_letters = joblib.load(svm_letters_path)

    def preprocess_char(self, img):
        """Redimensiona, aplana y normaliza la imagen del carácter."""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (32, 32)).flatten() / 255.0
        return img

    def predict_plate(self, image_path, save_dir=None):
        """Detecta la matrícula en una imagen y devuelve la predicción final (por ejemplo, '1234ABC')."""
        
        # --- 1. Detectar y recortar la matrícula ---
        results, crop = self.predictor.predict_and_crop_image(image_path, conf = 0.50, save_dir=save_dir)
        if crop is None:
            print("❌ No se pudo detectar ninguna matrícula en la imagen.")
            return None

        # --- 2. Segmentar caracteres ---
        segmentator = Segmentator(crop)
        numbers, letters = segmentator.segment_characters()

        # --- 3. Predecir números ---
        predicted_numbers = []
        for i, img in enumerate(numbers):
            processed = self.preprocess_char(img)
            pred = self.clf_digits.predict([processed])[0]
            predicted_numbers.append(str(pred))

        # --- 4. Predecir letras ---
        predicted_letters = []
        for i, img in enumerate(letters):
            processed = self.preprocess_char(img)
            pred = self.clf_letters.predict([processed])[0]
            predicted_letters.append(str(pred))

        # --- 5. Concatenar resultado final ---
        plate_str = "".join(predicted_numbers + predicted_letters)
        print(f"✅ Matrícula predicha: {plate_str}")

        return plate_str


"""
IMAGE_PATH = "/Users/lianbaguebatlle/Desktop/Dades/Tercer/1rsemestre/PSIV2/Repte1/Lateral/PXL_20210921_094926329.jpg"
MODEL_PATH = "/Users/lianbaguebatlle/Desktop/Dades/Tercer/1rsemestre/PSIV2/Repte1/models/best.pt"

reader = PlateReader(
    model_path=MODEL_PATH,
    svm_digits_path="svm_digits.pkl",
    svm_letters_path="svm_letters.pkl"
)


plate = reader.predict_plate(
    IMAGE_PATH,
    save_dir="/Users/lianbaguebatlle/Desktop/Dades/Tercer/1rsemestre/PSIV2/Repte1/Crops"
)

print("Resultado final:", plate)
"""