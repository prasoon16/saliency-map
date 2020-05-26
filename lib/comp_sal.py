import numpy as np
import cv2

def Gen_ISM(img, n, win_list):
    h,w = img.shape[:2]
    ISM = np.zeros((h,w), dtype=np.float64)
    for i in range(n):
        # sum = np.float64(0.0)
        x1,y1,x2,y2 = win_list[i][0],win_list[i][1],win_list[i][2],win_list[i][3]
        # print ("x1 = {} y1 = {} x2 = {} y2 = {}".format(x1,y1,x2,y2))
        # area = (x2 - x1 +1)*(y2 - y1 + 1)
        # for j in range(x1,x2+1):
        #     for k in range(y1,y2+1):
        #         sum = sum + ISM[j][k]
        # mean = np.mean(ISM[x1:x2+1,y1:y2+1])
        # mean = np.float64(sum/area)
        mean = cv2.mean(img[x1:x2+1,y1:y2+1])[0]
        # mean_cv = cv2.mean(ISM[x1:x2+1,y1:y2+1])
        # print ("sum = {} area = {} mean = {} mean_cv = {}".format(sum, area, mean, mean_cv))
        ISM[x1:x2+1,y1:y2+1] = ISM[x1:x2+1,y1:y2+1] + abs(img[x1:x2+1,y1:y2+1] - mean)
        # for j in range(x1,x2+1):
        #     for k in range(y1,y2+1):
        #         ISM[j][k] =  ISM[j][k]+ abs(img[j][k] - mean)
    return ISM
