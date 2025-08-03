import cv2 
import numpy as np

KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

def erosion (img): return cv2.erode(img,KERNEL,iteration=1)
def dilation (img): return cv2.dilate(img,KERNEL,iteration=1)
def opening (img): return cv2.morphologyEx(img,cv2.MORPH_OPEN,KERNEL)
def closing (img): return cv2.morphologyEx(img,cv2.MORPH_CLOSE,KERNEL)
def gradient (img): return cv2.morphologyEx(img,cv2.MORPH_GRADIENT,KERNEL)
def hittmass (img): return cv2.morphologyEx(img,cv2.MORPH_HITMISS,KERNEL)
