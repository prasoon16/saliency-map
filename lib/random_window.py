import random
import time
import numpy as np

def Generate_Window(n, r, c):
    rand_list = np.empty((n,4), dtype=np.int64)
    random.seed(time.time())
    for i in range(n):
        rand_list[i][0] = random.randint(0, r-1)
        rand_list[i][1] = random.randint(0, c-1)
        rand_list[i][2] = random.randint(rand_list[i][0] + 1, r-1)
        rand_list[i][3] = random.randint(rand_list[i][1] + 1, c-1)
    return rand_list
