
import cv2
import numpy as np

try:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    denoised = cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)
    print("Success")
except Exception as e:
    print(f"Error: {e}")
