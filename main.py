import cv2
import numpy as np
import sys
sys.path.insert(1, './lib/')
from random_window import Generate_Window
from comp_sal import Gen_ISM
from fusion import Image_Fusion

img = cv2.imread("images/rose.jpeg")
img = cv2.GaussianBlur(img, (7,7), 0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

h,w = img.shape[:2]
print (h,w)
win_list = Generate_Window(10,h,w)
print (win_list)

print (img[:,:,0].shape)
ISM_L = Gen_ISM(img[:, :, 0], 10, win_list)
ISM_A = Gen_ISM(img[:, :, 1], 10, win_list)
ISM_B = Gen_ISM(img[:, :, 2], 10, win_list)

map = Image_Fusion(ISM_L, ISM_A, ISM_B)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.imshow("map", map)
cv2.waitKey(0)
