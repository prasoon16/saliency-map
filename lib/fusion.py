import numpy as np

def Image_Fusion(L, A, B):
    h,w = L.shape[:2]
    img = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            img[i][j] = np.sqrt((L[i][j]*L[i][j]) + (A[i][j]*A[i][j]) + (B[i][j]*B[i][j]))
    return img
