import cv2
import numpy as np
import torch
from PIL import Image
 #获取文件夹内的图片
import os
def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.bmp')]

def get_picture():
    feature = np.zeros((1,1,640,640),dtype=np.uint8)
    for index in range(1):
         img_path =  get_imlist("../images/")[index]
         image = Image.open(img_path)
         image_image=image.convert('L')
         # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
         image_image=image_image.resize((640,640))
         feature[index,:,:,:] = image_image
    return  feature
