import cv2
import numpy as np
import sys
sys.path.insert(1, './lib/')
from random_window import Generate_Window
from comp_sal import Gen_ISM
from fusion import Image_Fusion

im = cv2.imread("images/rose.jpeg")
img = cv2.GaussianBlur(im, (7,7), 0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

h,w = img.shape[:2]
print (h,w)
win_list = Generate_Window(10,h,w)
print (win_list)

# print (img[:,:,0])
ISM_L = Gen_ISM(img[:, :, 0], 10, win_list)
# print (ISM_L)
ISM_A = Gen_ISM(img[:, :, 1], 10, win_list)
ISM_B = Gen_ISM(img[:, :, 2], 10, win_list)

map = Image_Fusion(ISM_L, ISM_A, ISM_B)
median = cv2.medianBlur(map, 5)

norm_img = cv2.normalize(median, None, 0, 255, cv2.NORM_MINMAX)
norm_img = norm_img.astype(np.uint8)
norm_img = cv2.equalizeHist(norm_img)
# median = cv2.medianBlur(norm_img, 5)
# cv.imshow('Normalized Image', final_img)

cv2.imshow("img", im)
cv2.waitKey(0)
cv2.imshow("map", norm_img)
cv2.waitKey(0)
