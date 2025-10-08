
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os


class Segmentator:
    def __init__(self, img):
        self.img = img
    
    def convert_to_HSV(self):
        image_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        image_h, image_s, image_v = cv2.split(image_hsv)
        return image_v
    
    def gaussian_blur(self, gray: np.ndarray, ksize: int = 5) -> np.ndarray:
        return cv2.GaussianBlur(gray, (ksize, ksize), 0)
    
    def binarize(self, image: np.ndarray) -> np.ndarray:
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary_image

    def closing(self, binary_image: np.ndarray, ksize: int = 3) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        return closed_image
    
    def opening(self, binary_image: np.ndarray, ksize: int = 3) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        return opened_image
    
    
    def find_contours(self, binary_image: np.ndarray):
        binary_inverted = 255 - binary_image
        contours, hierarchy = cv2.findContours(binary_inverted, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)          
        return contours
    
    def filter_license_plate_contours(self, contours, binary: np.ndarray):
        img_h, img_w = binary.shape
        filtered = []
        area_image = img_h * img_w

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            area_ratio = area / area_image
            aspect_ratio = h / w if w != 0 else 0

            # Filters: not touching borders, reasonable area, tall characters
            if (x > 0 and (x+w) < img_w and
                y > 0 and (y+h) < img_h and
                0.002 < area_ratio < 0.1 and
                1.2 < aspect_ratio < 7.0):
                filtered.append((x, y, w, h))

        # Sort left-to-right
        filtered = sorted(filtered, key=lambda b: b[0])

        # Keep exactly 7 characters if more are detected
        if len(filtered) > 7:
            filtered = sorted(filtered, key=lambda b: b[2]*b[3], reverse=True)[:7]
            filtered = sorted(filtered, key=lambda b: b[0])

        return filtered
    
    def segment_characters(self):
        image_v = self.convert_to_HSV()
        image_v_blurred = self.gaussian_blur(image_v, ksize=5)
        binary_image = self.binarize(image_v_blurred)
        contours = self.find_contours(binary_image)
        boxes = self.filter_license_plate_contours(contours,binary_image)

        characters = []
        for idx, (x, y, w, h) in enumerate(boxes):
            char_crop = binary_image[y:y+h, x:x+w]
            characters.append(char_crop)

        numbers = characters[:4]   # primeros 4
        letters = characters[4:]   # últimos 3


        return numbers, letters