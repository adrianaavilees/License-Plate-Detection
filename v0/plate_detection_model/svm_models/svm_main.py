import os
import matplotlib.pyplot as plt
from auxiliar_svm.plate_reader import PlateReader  

# Configuración


IMAGE_DIR = "/Users/lianbaguebatlle/Desktop/Dades/Tercer/1rsemestre/PSIV2/Projecte1/License-Plate-Detection/final_test_dataset_60"
SAVE_DIR = "/Users/lianbaguebatlle/Desktop/Dades/Tercer/1rsemestre/PSIV2/Projecte1/License-Plate-Detection/Crops"
MODEL_PATH = "/Users/lianbaguebatlle/Desktop/Dades/Tercer/1rsemestre/PSIV2/Projecte1/License-Plate-Detection/models/best_license_plate.pt"


SVM_DIGITS = "/Users/lianbaguebatlle/Desktop/Dades/Tercer/1rsemestre/PSIV2/Projecte1/License-Plate-Detection/v0/plate_detection_model/svm_models/svm_digits.pkl"
SVM_LETTERS = "/Users/lianbaguebatlle/Desktop/Dades/Tercer/1rsemestre/PSIV2/Projecte1/License-Plate-Detection/v0/plate_detection_model/svm_models/svm_letters.pkl"

plate_reader = PlateReader(MODEL_PATH, SVM_DIGITS, SVM_LETTERS)

# Iterar sobre todas las imágenes
for img_file in os.listdir(IMAGE_DIR):
    if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_file)
    print(f"\n--- Procesando {img_file} ---")

    # Llamar a PlateReader para obtener predicción
    plate_str = plate_reader.predict_plate(img_path, save_dir=SAVE_DIR)

    if plate_str is None:
        continue  # saltar si no detectó matrícula

    # --- Mostrar resultados en matplotlib ---
    base_name = os.path.splitext(img_file)[0]
    image_save_dir = os.path.join(SAVE_DIR, base_name)

    # Leer crop completo
    crop_path = os.path.join(image_save_dir, f"{base_name}_plate.jpg")
    crop_img = plt.imread(crop_path)

    # Leer caracteres segmentados en orden correcto: números primero, luego letras
    char_imgs = []
    char_titles = []

    num_files = sorted([f for f in os.listdir(image_save_dir) if f.startswith("num_")])
    let_files = sorted([f for f in os.listdir(image_save_dir) if f.startswith("let_")])
    ordered_files = num_files + let_files

    for f in ordered_files:
        char_imgs.append(plt.imread(os.path.join(image_save_dir, f)))
        char_titles.append(f.split("_")[2].split(".")[0])  # predicción del carácter

    # Si no hay caracteres, solo mostrar crop
    total_chars = len(char_imgs)
    fig, axes = plt.subplots(1, total_chars + 1, figsize=(2*(total_chars+1), 3))

    # Asegurarse de que axes sea siempre iterable
    if total_chars + 1 == 1:
        axes = [axes]

    # Mostrar crop completo
    axes[0].imshow(crop_img, cmap='gray')
    axes[0].set_title("Crop")
    axes[0].axis("off")

    # Mostrar cada carácter segmentado
    for i, ax in enumerate(axes[1:]):
        ax.imshow(char_imgs[i], cmap='gray')
        ax.set_title(char_titles[i])
        ax.axis("off")

    # Título general con predicción
    plt.suptitle(f"Predicción final: {plate_str}", fontsize=16)
    plt.show()