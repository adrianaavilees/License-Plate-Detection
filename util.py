# YOLOV8 model trained for detecting

import cv2
from ultralytics import YOLO
import easyocr
import os
import numpy as np
import torch

# Load the YOLOv8 model
model = YOLO("yolov8n.pt") #! Esta detectando coches, NO matriculas. Detecta 4 coches en la imagen de prueba

"https://www.kaggle.com/code/ztrollk/license-plate-detection-with-yolov8"
"https://docs.ultralytics.com/es/datasets/detect/coco/#what-is-the-coco-dataset-and-why-is-it-important-for-computer-vision"
"https://www.kaggle.com/code/yyazidd/yolov8-license-plate-detection"

def create_output_folder():
    output_folder = "detected_regions"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder

def find_license_plates_in_vehicle(vehicle_region):
    """Detect license plates within the vechile region using contour detection"""
    if vehicle_region.size == 0:
        return []
    
    gray = cv2.cvtColor(vehicle_region, cv2.COLOR_BGR2GRAY)

    # # Detect white rectangular regions that could be license plates
    # _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY) # try 180, 255

    # # Apply some morphological operations to clean up the mask
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # white_regions = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(filtered, 30, 200)

    # Combine white regions and edges
    # combined = cv2.bitwise_and(edges, white_regions)
    
    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    possible_plates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Obtain bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)

        # Verify ratio and area tipically of license plates
        if h > 0:  # Prevent division by zero
            aspect_ratio = w / h
            # Typical spanish license plate aspect ratio 
            if 2 < aspect_ratio < 6: 
                #Filter relative area too small or too big respect the car
                width, height = vehicle_region.shape[1], vehicle_region.shape[0]
                relative_area = area / (width * height)
                #if relative_area > 0.010 and relative_area < 0.35: 
                possible_plates.append({
                        'bbox': (x, y, x + w, y + h),
                        'area': area,
                        'aspect_ratio': aspect_ratio
                    })
    possible_plates.sort(key=lambda x: x['area'], reverse=True)

    return possible_plates[:3] #? Ponemos un maximo por cada foto de possibles matriculas detectadas?

def detect_license_plate(image_path, output_path="detected_image14.jpg"):
    """Detect license plates in an image using YOLOv8"""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return [], None

    # Perform object detection
    results = model(image)

    license_plates = []
    vehicle_count = 0

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0].cpu().numpy())
                # Check if the detected object is a vechicle
                if class_id not in [2,2,5,7]:  # car, motorcycle, bus, truck 
                    continue
                
                # Obtain bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()

                # Extract vehicle region
                vehicle_region = image[int(y1):int(y2), int(x1):int(x2)]
                
                # Find license plates within the vehicle region
                plate_candidates = find_license_plates_in_vehicle(vehicle_region)       
    
                # Draw bounding box of the car (blue)
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(image, f"Conf: {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2) 
                
                #Draw bounding boxes of the license plates (green)
                for i, plate in enumerate(plate_candidates):
                    px1, py1, px2, py2 = plate['bbox']
                    # Adjust coordinates relative to the original image
                    abs_px1, abs_py1 = int(x1) + px1, int(y1) + py1
                    abs_px2, abs_py2 = int(x1) + px2, int(y1) + py2
                    
                    cv2.rectangle(image, (abs_px1, abs_py1), (abs_px2, abs_py2), (0, 255, 0), 2)
                    cv2.putText(image, f"Plate {i+1}", (abs_px1, abs_py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1) 
                    
                    license_plates.append({
                        'vehicle_id': vehicle_count,
                        'vehicle_bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'bbox': (abs_px1, abs_py1, abs_px2, abs_py2),
                        'confidence': conf
                    })
                    
                vehicle_count += 1

    cv2.imwrite(output_path, image) 
    print(f"Processed image saved to {output_path}")

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
    #TODO: diccionario de correciones, por ejemplo, si detecta 0 en lugar de O, o 1 en lugar de I, etc.
    #TODO: diccionario sin las letras que no se usan como Q, Ñ, vocales, etc.
    if processed_image is None:
        return ""
    
    ocr_result = reader.readtext(processed_image, detail=0)
    
    if not ocr_result:
        return ""   
    
    # Join the recognized text parts
    full_text = ''.join(ocr_result)
    cleaned_text = ''.join(c for c in full_text if c.isalnum())

    # Validate the format of the license plate (spanish: 4 digits and 3 letters)
    # if len(cleaned_text) == 7 and cleaned_text[:4].isdigit() and cleaned_text[4:].isalpha():
    #     return cleaned_text
    # else:
    #     return ""
    return cleaned_text
    
# Example usage
license, image = detect_license_plate(r"C:\Users\adria\OneDrive - UAB\4 ENGINY\Processament Imatge i Video\Repte Matriculas\BD_Matriculas\PXL_20210921_095129495.jpg")
#license, image = detect_license_plate(r"C:\Users\adria\OneDrive - UAB\4 ENGINY\Processament Imatge i Video\Repte Matriculas\BD_Matriculas\WhatsApp Image 2025-09-12 at 14.34.01 (1).jpeg")
#license, image = detect_license_plate(r"C:\Users\adria\OneDrive - UAB\4 ENGINY\Processament Imatge i Video\Repte Matriculas\BD_Matriculas\PXL_20210921_094938026.jpg")

reader = easyocr.Reader(['en'], gpu=False)  # Initialize EasyOCR reader
for plate in license:
    bbox = plate['bbox']
    processed_image = preprocess_license_plate(image, bbox)
    text = extract_text_from_plate(processed_image)
    print(f"Detected License Plate: {text} ")