import cv2
import numpy as np
from utils.morpho_ops import erosion,dilation,opening,closing,gradient,hittmass
from utils.viz import show_images

img = cv2.imread ("images", cv2.IMREAD_GRAYSCALE)
-,bin_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV + cv2.thresh_otsu)
ops = [
    (erosion(bin_img), "Erosion"),
    (dilation(bin_img), "Dilation"),
    (opening(bin_img), "Opening"),
    (closing(bin_img), "Closing"),
    (gradient(bin_img), "Gradient"),
    (hittmass(bin_img), "Gradient")
]
show_images([bin_img] + [o for o,_ in ops], ["Original"] + [t for _,t in ops])
