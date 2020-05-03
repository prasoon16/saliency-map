import numpy as np

def Image_Fusion(L, A, B):
    h,w = L.shape[:2]
    img = np.empty((h,w))
    for i in range(h):
        for j in range(w):
            img[i][j] = np.sqrt((L[i][j]**2) + (A[i][j]**2) + (B[i][j]**2))
    return img.astype(np.uint8)
