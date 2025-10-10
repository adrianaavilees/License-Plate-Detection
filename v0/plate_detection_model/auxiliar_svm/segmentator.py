
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os


class Segmentator:
    def __init__(self, img):
        self.img = img

    def preprocess_plate_for_segmentation(self):
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        _, _, image_v = cv2.split(hsv)
        image_v = cv2.GaussianBlur(image_v, (5, 5), 0)
        _, binary = cv2.threshold(image_v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def segment_characters(self):
        preprocessed_plate = self.preprocess_plate_for_segmentation()
        plate_height, plate_width = preprocessed_plate.shape
        EXPANSION_FACTOR = 0.04 
        eu_zone_width = int(plate_width * 0.05)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            preprocessed_plate, connectivity=8
        )
        
        characters = []
        for i in range(1, num_labels):
            x_orig = stats[i, cv2.CC_STAT_LEFT]
            y_orig = stats[i, cv2.CC_STAT_TOP]
            w_orig = stats[i, cv2.CC_STAT_WIDTH]
            h_orig = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            aspect_ratio = w_orig / h_orig if h_orig > 0 else 0
            
            min_char_height = plate_height * 0.25
            max_char_height = plate_height * 0.95
            min_char_width = plate_width * 0.02
            min_area = (min_char_height * min_char_width) * 0.3 
            
            valid_height = min_char_height <= h_orig <= max_char_height
            valid_width = w_orig >= min_char_width
            valid_area = area > min_area
            valid_aspect = 0.05 < aspect_ratio < 2.0
            
            in_eu_zone = x_orig < eu_zone_width
            touches_border = (x_orig == 0 or y_orig == 0 or 
                              (x_orig + w_orig) >= plate_width or (y_orig + h_orig) >= plate_height)

            if (valid_height and valid_width and valid_area and 
                valid_aspect and not touches_border and not in_eu_zone):
                
                expansion_amount = int(h_orig * EXPANSION_FACTOR)
                x_start = max(0, x_orig - expansion_amount)
                y_start = max(0, y_orig - expansion_amount)
                x_end = min(plate_width, x_orig + w_orig + expansion_amount)
                y_end = min(plate_height, y_orig + h_orig + expansion_amount)
                
                character_roi = preprocessed_plate[y_start:y_end, x_start:x_end]
                
                characters.append({
                    'image': character_roi, 
                    'bbox': (x_start, y_start, x_end - x_start, y_end - y_start), 
                    'area': area, 
                    'center_x': x_start + (x_end - x_start)/2
                })
        
        # Ordenar de izquierda a derecha
        characters.sort(key=lambda c: c['bbox'][0])
        
        # Si hay más de 7, quedarse con los de mayor área
        if len(characters) > 7:
            characters = sorted(characters, key=lambda c: c['area'], reverse=True)[:7]
            characters.sort(key=lambda c: c['bbox'][0])

        char_images = [c['image'] for c in characters]

        numbers = char_images[:4]
        letters = char_images[4:]

        return numbers, letters
