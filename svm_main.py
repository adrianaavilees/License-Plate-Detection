from plate_reader import PlateReader
import os

FOLDER_PATH = "/Users/lianbaguebatlle/Desktop/Dades/Tercer/1rsemestre/PSIV2/Projecte1/License-Plate-Detection/converted_jpg"
MODEL_PATH = "/Users/lianbaguebatlle/Desktop/Dades/Tercer/1rsemestre/PSIV2/Projecte1/License-Plate-Detection/models/best_license_plate.pt"
SVM_DIGITS = "svm_digits.pkl"
SVM_LETTERS = "svm_letters.pkl"

SAVE_CROPPED_LICENSES="/Users/lianbaguebatlle/Desktop/Dades/Tercer/1rsemestre/PSIV2/Projecte1/License-Plate-Detection/Crops"

# --- Inicializar lector de matrículas ---
reader = PlateReader(
    model_path=MODEL_PATH,
    svm_digits_path=SVM_DIGITS,
    svm_letters_path=SVM_LETTERS
)

# --- Iterar sobre todas las imágenes de la carpeta ---
for filename in sorted(os.listdir(FOLDER_PATH)):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(FOLDER_PATH, filename)
        print(f"\n📸 Procesando: {filename}")
        
        plate_pred = reader.predict_plate(image_path, save_dir=SAVE_CROPPED_LICENSES)
        
        if plate_pred is not None:
            print(f"✅ Matrícula predicha: {plate_pred}")
        else:
            print("⚠️ No se pudo detectar ninguna matrícula en esta imagen.")