import numpy as np

def Gen_ISM(img, n, win_list):
    h,w = img.shape[:2]
    ISM = np.zeros((h,w))
    for i in range(n):
        sum = 0
        x1,y1,x2,y2 = win_list[i][0],win_list[i][1],win_list[i][2],win_list[i][3]
        area = (x2 - x1 +1)*(y2 - y1 + 1)
        for j in range(x1,x2):
            for k in range(y1,y2):
                sum = sum + ISM[j][k]
        mean = sum/area
        for j in range(x1,x2):
            for k in range(y1,y2):
                ISM[j][k] = ISM[j][k] + abs(img[j][k] - mean)
    return ISM
