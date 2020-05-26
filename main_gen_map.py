import cv2
import numpy as np
import sys
sys.path.insert(1, './lib/')
from random_window import Generate_Window
from comp_sal import Gen_ISM
from fusion import Image_Fusion
from color_conversion import bgr2lab
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from skimage.filters import threshold_otsu
from skimage.filters import threshold_minimum
import skimage.filters
import os

MSRA_PATH = "MSRA_DATA/"
RES_SAL_MAP_PATH = "MSRA_DATA_SAL_MAP/"
RES_COMP_PATH = "MSRA_DATA_COMP/"
def get_image_list(ext):
    cwd = os.getcwd()
    msra_data = os.path.join(cwd, MSRA_PATH)
    img_list = []
    for file in os.listdir(msra_data):
        extension = os.path.splitext(file)[1]
        if (extension == ext):
            # print (os.path.join(msra_data,file))
            img_list.append(file)
    print (len(img_list))
    return img_list

def get_img_sal_map(img_name):
    cwd = os.getcwd()
    msra_data = os.path.join(cwd, MSRA_PATH)
    img_path = os.path.join(msra_data,img_name)
    file_name = os.path.splitext(img_name)[0]
    img_ground_path = os.path.join(msra_data,file_name+".png")
    print (img_ground_path)
    im = cv2.imread(img_path)
    img_ground = cv2.imread(img_ground_path)
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    img_gauss = cv2.GaussianBlur(im, (3,3), sigmaX=0.5,sigmaY=0.5)
    img = cv2.cvtColor(img_gauss, cv2.COLOR_BGR2Lab)
    img = img.astype(np.float64)
    h,w = img.shape[:2]
    L,A,B = cv2.split(img)
    NUM = int(0.02*h*w)
    # NUM = 500
    win_list = Generate_Window(NUM,h,w)
    L_ISM = Gen_ISM(L, NUM, win_list)
    A_ISM = Gen_ISM(A, NUM, win_list)
    B_ISM = Gen_ISM(B, NUM, win_list)
    IMG = Image_Fusion(L_ISM, A_ISM, B_ISM)
    normalizedImg = cv2.normalize(IMG,  None, 0, 255, cv2.NORM_MINMAX)
    normalizedImg = normalizedImg.astype(np.uint8)
    normalizedImg = cv2.medianBlur(normalizedImg, 11)
    ret,th_img = cv2.threshold(normalizedImg, 100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cwd = os.getcwd()
    res_data = os.path.join(cwd, RES_SAL_MAP_PATH)
    if not os.path.exists(res_data):
        os.mkdir(res_data)
    name = os.path.splitext(img_name)[0]
    map_name = name + "_map.png"
    mask_name = name + "_mask.png"
    print (os.path.join(res_data,map_name))
    cv2.imwrite(os.path.join(res_data,map_name), normalizedImg)
    cv2.imwrite(os.path.join(res_data,mask_name), th_img)

    cwd = os.getcwd()
    res_comp_data = os.path.join(cwd, RES_COMP_PATH)
    if not os.path.exists(res_comp_data):
        os.mkdir(res_comp_data)
    comp_name = name + "_comp.png"
    plt.subplot(311),plt.imshow(im_rgb),plt.title("original"),plt.axis("off")
    plt.subplot(312),plt.imshow(normalizedImg, cmap="gray"),plt.title("Saliency Map"),plt.axis("off")
    # plt.subplot(224),plt.imshow(th_img, cmap="gray"),plt.title("Saliency Map Otsu"),plt.axis("off")
    plt.subplot(313),plt.imshow(img_ground, cmap="gray"),plt.title("Ground Truth"),plt.axis("off")
    plt.savefig(os.path.join(res_comp_data, comp_name))

input_images = get_image_list(".jpg")
mask_images = get_image_list(".png")
input_images = sorted(input_images, reverse=True)
inp_images = input_images
print (inp_images)
for img_name in inp_images:
    get_img_sal_map(img_name)
