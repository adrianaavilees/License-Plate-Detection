
import cv2
import os
from ultralytics import YOLO
import shutil


class PlatePredictor:
    def __init__(self, model_path: str):
        """Carga el modelo entrenado (best.pt)"""
        self.model = YOLO(model_path)

    def predict_and_crop_folder(self, folder_path: str, conf: float = 0.25, save_dir: str = "crops"):
        """Predice todas las imágenes de una carpeta y guarda recortes"""
        results = self.model.predict(
            source=folder_path,
            save=True,
            conf=conf
        )

        # Crear carpeta de salida
        os.makedirs(save_dir, exist_ok=True)

        # Procesar resultados
        for r in results:
            image_path = r.path  # ruta de la imagen original
            img = cv2.imread(image_path)
            base_name = os.path.basename(image_path)

            for i, box in enumerate(r.boxes):
                # Coordenadas (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Recorte
                crop = img[y1:y2, x1:x2]

                # Nombre único para cada recorte
                crop_path = os.path.join(save_dir, f"crop_{i}_{base_name}")
                cv2.imwrite(crop_path, crop)

        return results
    
    def predict_and_crop_image(self, image_path: str, conf: float = 0.25, save_dir: str = "crops"):
        """Predice una sola imagen y guarda recortes"""
        crop = None

        # 🔥 Borrar carpeta si existe y recrearla limpia
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        # Hacer predicción
        results = self.model.predict(
            source=image_path,
            save=True,
            conf=conf
        )

        # Leer imagen original
        img = cv2.imread(image_path)
        base_name = os.path.basename(image_path)

        for i, box in enumerate(results[0].boxes):
            # Coordenadas del bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Recorte
            crop = img[y1:y2, x1:x2]

            # Guardar recorte
            crop_path = os.path.join(save_dir, f"crop_{i}_{base_name}")
            cv2.imwrite(crop_path, crop)

        return results, crop