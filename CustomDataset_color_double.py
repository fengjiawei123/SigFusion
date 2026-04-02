import os
import re
import numpy as np
from skimage.io import imread
import torch
from torch.utils.data import Dataset
import skimage
import cv2

def rgb2yuv(rgb):
    rgb = rgb.astype(np.float32)

    m = [[0.299, 0.587, 0.114], [-0.147, -0.289, 0.436], [0.615, -0.515, -0.100]]
    shape1 = rgb.shape
    yuv = np.empty(shape1, dtype=np.float32)
    for i in range(3):
        yuv[:, :, i] = rgb[:, :, 0]*m[i][0] + rgb[:, :, 1]*m[i][1] + rgb[:, :, 2]*m[i][2]
    return yuv

def yuv2rgb(yuv):

    mtxYUVtoRGB = np.array([[1.0000, -0.0000, 1.1398],
                            [1.0000, -0.3946, -0.5805],
                            [1.0000,  2.0320, -0.0005]])
    rgb = np.zeros(yuv.shape)
    for i in range(3):
        rgb[:, :, i] = yuv[:, :, 0] * mtxYUVtoRGB[i, 0] + yuv[:, :, 1] * mtxYUVtoRGB[i, 1] + yuv[:, :, 2] * mtxYUVtoRGB[i, 2]
    return rgb
class MyDataset(Dataset):
    def __init__(self, root_dir_mri, root_dir_ct):
        self.root_dir_mri = root_dir_mri
        self.root_dir_ct = root_dir_ct
        # self.mri_files = sorted(self.get_img_files(root_dir_mri))
        # self.ct_files = sorted(self.get_img_files(root_dir_ct))
        self.mri_files = sorted(self.get_img_files(root_dir_mri),key=self.extract_number)
        self.ct_files = sorted(self.get_img_files(root_dir_ct),key=self.extract_number)

        
        assert len(self.mri_files) == len(self.ct_files)
        
    def get_img_files(self, root_dir):
        imagelist = []
        for parent, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')):
                    imagelist.append(os.path.join(parent, filename))
        return imagelist
    def extract_number(self, filename):
        # 使用正则表达式提取文件名中的数字部分
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else float('inf')

    def __len__(self):
        return len(self.mri_files)
    
    def rgb2y(self, img):
        y = img[0:1, :, :] * 0.299000 + img[0:1, :, :] * 0.587000 + img[2:3, :, :] * 0.114000
        return y
    
    def __getitem__(self, idx):
        mri_file = self.mri_files[idx]
        ct_file = self.ct_files[idx]

        mri_img = imread(mri_file)
        if mri_img.shape[-1] != 3:
            mri_img = skimage.color.gray2rgb(mri_img)
        mri_img = mri_img.astype(np.float32)
         
        yuv_img_mri = cv2.cvtColor(mri_img, cv2.COLOR_RGB2YUV)

        Test_IR = yuv_img_mri[:,:,0]/255.
 
        uv_Test_IR = yuv_img_mri[:, :,1:3]
        y_Test_IR = torch.Tensor(Test_IR).unsqueeze(0)
        

        ct_img = imread(ct_file)
        if ct_img.shape[-1] != 3:
            ct_img = skimage.color.gray2rgb(ct_img)     
        ct_img = ct_img.astype(np.float32)

        yuv_img_ct = cv2.cvtColor(ct_img, cv2.COLOR_RGB2YUV)

        Test_Vis = yuv_img_ct[:,:,0]/255.
 
        uv_Test_vis = yuv_img_ct[:, :,1:3]
        y_Test_vis = torch.Tensor(Test_Vis).unsqueeze(0)

        return y_Test_IR,uv_Test_IR, y_Test_vis, uv_Test_vis
    

