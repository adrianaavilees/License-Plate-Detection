
import cv2
import os
from ultralytics import YOLO
import shutil


class PlatePredictor:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict_and_crop_folder(self, folder_path: str, conf: float = 0.25, save_dir: str = "crops"):
        results = self.model.predict(
            source=folder_path,
            save=True,
            conf=conf
        )

        os.makedirs(save_dir, exist_ok=True)

        for r in results:
            image_path = r.path  
            img = cv2.imread(image_path)
            base_name = os.path.basename(image_path)

            for i, box in enumerate(r.boxes):
                # Coordenadas (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = img[y1:y2, x1:x2]

                crop_path = os.path.join(save_dir, f"crop_{i}_{base_name}")
                cv2.imwrite(crop_path, crop)

                print(f"✅ Recorte guardado en {crop_path}")

        return results

    def predict_and_crop_image(self, image_path: str, conf: float = 0.25, save_dir: str = "crops"):
        crop = None

        os.makedirs(save_dir, exist_ok=True)

        results = self.model.predict(
            source=image_path,
            save=True,
            conf=conf
        )

        img = cv2.imread(image_path)
        base_name = os.path.basename(image_path)

        for i, box in enumerate(results[0].boxes):
            # Coordenadas del bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]

            # Guardar crop
            crop_path = os.path.join(save_dir, f"crop_{i}_{base_name}")
            cv2.imwrite(crop_path, crop)

            print(f"✅ Recorte guardado en {crop_path}")

        return results, crop
  


