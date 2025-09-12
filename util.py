# YOLOV8 model trained for detecting

import cv2
from ultralytics import YOLO
import easyocr

# Load the YOLOv8 model
model = YOLO("yolov8n.pt") #! Esta detectando coches, NO matriculas. Detecta 4 coches en la imagen de prueba

def detect_license_plate(image_path):
    """Detect license plates in an image using YOLOv8"""
    # Read the image
    image = cv2.imread(image_path)

    # Perform object detection
    results = model(image)

    license_plates = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Obtain bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                    
                # Search possible areas for license plates
                # Assuming license plates are rectangular
                license_plates.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(conf)
                })
        
    return license_plates, image

def preprocess_license_plate(image, bbox):
    """Preprocess the license plate image for better OCR results"""
    x1, y1, x2, y2 = bbox
    # Extract the license plate region
    license_plate_img = image[y1:y2, x1:x2]

    # Resize the image to a standard size
    #?plate_image = cv2.resize(plate_image, (200, 100)) --> esto es mas simple, puede distorsionar la imagen

    height, width = license_plate_img.shape[:2]
    if height < 50:
        scale = 50 / height
        new_width = int(width * scale)
        license_plate_img = cv2.resize(license_plate_img, (new_width, 50))
    
    gray = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2GRAY)
    
    #Apply filters to remove noise and improve contrast
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    #? Apply morphological operations to enhance the characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return processed

def extract_text_from_plate(processed_image):
    if processed_image is None:
        return ""
    
    ocr_result = reader.readtext(processed_image, detail=0)
    
    if not ocr_result:
        return ""   
    
    # Join the recognized text parts
    full_text = ''.join(ocr_result)
    cleaned_text = ''.join(c for c in full_text if c.isalnum())

    # Validate the format of the license plate (spanish: 4 digits and 3 letters)
    #! esto es muy restrictivo de momento que no detecta bien
    # if len(cleaned_text) == 7 and cleaned_text[:4].isdigit() and cleaned_text[4:].isalpha():
    #     return cleaned_text
    # else:
    #     return ""
    return cleaned_text
    
# Example usage
license, image = detect_license_plate(r"C:\Users\adria\OneDrive - UAB\4 ENGINY\Processament Imatge i Video\Repte Matriculas\BD_Matriculas\PXL_20210921_094926329.jpg")

reader = easyocr.Reader(['en'], gpu=False)  # Initialize EasyOCR reader
for plate in license:
    bbox = plate['bbox']
    processed_image = preprocess_license_plate(image, bbox)
    text = extract_text_from_plate(processed_image)
    print(f"Detected License Plate: {text} ")
    # Draw bounding box and text on the image
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)