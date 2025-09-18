import cv2
from ultralytics import YOLO
import easyocr
import os
import torch
import numpy as np

# Load the general YOLOv8 model for vehicle detection
print("Carregant model YOLOv8 per a la detecció de vehicles...")
model = YOLO("yolov8n.pt")

# Initialize EasyOCR reader
print("Inicialitzant EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)

def find_license_plate_by_color(vehicle_region):
    """
    Detects the blue stripe of the license plate to locate the plate.
    
    Args:
        vehicle_region (np.array): Cropped image of the vehicle.
        
    Returns:
        tuple: Bounding box of the license plate, or None if not found.
    """
    if vehicle_region is None or vehicle_region.size == 0:
        return None

    height, width = vehicle_region.shape[:2]
    
    hsv = cv2.cvtColor(vehicle_region, cv2.COLOR_BGR2HSV)
    
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([130, 255, 255])

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    cleaned_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_rect)

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_candidate = None
    max_area = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        if area > 100 and w < h and h > 15:
            if area > max_area:
                max_area = area
                plate_width = int(h * 4.5)
                px1 = x
                py1 = y
                px2 = x + plate_width
                py2 = y + h
                
                if px2 < width and py2 < height:
                    best_candidate = (px1, py1, px2, py2)
    
    return best_candidate

def detect_license_plate(image_path, output_path="detected_image_final.jpg"):
    """
    Detects vehicles and then finds license plates within the detected regions.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No s'ha pogut carregar la imatge a {image_path}")
        return [], None
    
    results = model(image)
    license_plates = []

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0].cpu().numpy())
                if class_id not in [2, 3, 5, 7]:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                if conf < 0.5:
                    continue

                padding = 20
                x1_pad = max(0, int(x1) - padding)
                y1_pad = max(0, int(y1) - padding)
                x2_pad = min(image.shape[1], int(x2) + padding)
                y2_pad = min(image.shape[0], int(y2) + padding)
                
                vehicle_region = image[y1_pad:y2_pad, x1_pad:x2_pad]
                
                plate_bbox = find_license_plate_by_color(vehicle_region)
                
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                
                if plate_bbox:
                    px1, py1, px2, py2 = plate_bbox
                    abs_px1 = x1_pad + px1
                    abs_py1 = y1_pad + py1
                    abs_py2 = y1_pad + py2
                    abs_px2 = x1_pad + px2
                    
                    cv2.rectangle(image, (abs_px1, abs_py1), (abs_px2, abs_py2), (0, 255, 0), 2)
                    cv2.putText(image, "Matricula Trobada", 
                               (abs_px1, abs_py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    license_plates.append({
                        'bbox': (abs_px1, abs_py1, abs_px2, abs_py2),
                        'confidence': 1.0
                    })

    cv2.imwrite(output_path, image)
    return license_plates, image

def preprocess_license_plate(image, bbox):
    """Enhanced preprocessing for better OCR results"""
    x1, y1, x2, y2 = bbox
    license_plate_img = image[y1:y2, x1:x2]
    
    if license_plate_img.size == 0:
        return None

    height, width = license_plate_img.shape[:2]
    if height == 0 or width == 0:
        return None
        
    scale = 60 / height
    new_width = int(width * scale)
    license_plate_img = cv2.resize(license_plate_img, (new_width, 60))
    
    gray = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    return thresh

def validate_spanish_license_plate(text):
    """Validate and correct Spanish license plate format"""
    cleaned = ''.join(c for c in text.upper() if c.isalnum())
    
    if len(cleaned) == 7:
        digits_part = cleaned[:4]
        letters_part = cleaned[4:]
        
        if digits_part.isdigit() and letters_part.isalpha():
            return digits_part + letters_part
    
    return ""

def extract_text_from_plate(processed_image):
    """Extract text from preprocessed license plate image"""
    if processed_image is None:
        return ""
    
    try:
        ocr_result = reader.readtext(processed_image, detail=0, paragraph=False, width_ths=0.8,
                                    allowlist='0123456789BCDFGHJKLMNPRSTVWXYZ')
        
        full_text = ''.join(ocr_result)
        validated_text = validate_spanish_license_plate(full_text)
        
        return validated_text
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

def process_images_in_folder(folder_path):
    """
    Processes all images in a given folder to detect and recognize license plates.
    
    Args:
        folder_path (str): The path to the folder containing the images.
    """
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_extensions):
            image_path = os.path.join(folder_path, filename)
            output_path = os.path.join(folder_path, "processed_" + filename)
            
            print(f"\n---- Processant imatge: {filename} ----")
            detected_plates, processed_image = detect_license_plate(image_path, output_path)
            
            if processed_image is not None and detected_plates:
                print(f"S'han trobat {len(detected_plates)} matrícules candidates:")
                for i, plate in enumerate(detected_plates):
                    print(f"\nMatrícula {i+1}:")
                    print(f"  Confiança de detecció: {plate['confidence']:.2f}")
                    
                    processed_plate_img = preprocess_license_plate(processed_image, plate['bbox'])
                    
                    if processed_plate_img is not None:
                        text = extract_text_from_plate(processed_plate_img)
                        print(f"  Text detectat: '{text}'")
                    else:
                        print("  Error en processar la imatge de la matrícula")
            else:
                print("No s'ha trobat cap matrícula en aquesta imatge.")

# Entry point of the script
if __name__ == "__main__":
    folder_to_process = r"C:\Users\PC\OneDrive\Escriptori\UNI\4t\MA PSIV\License Plate Detection\License-Plate-Detection\Lateral"
    
    if os.path.isdir(folder_to_process):
        process_images_in_folder(folder_to_process)
    else:
        print(f"Error: La carpeta '{folder_to_process}' no existeix.")