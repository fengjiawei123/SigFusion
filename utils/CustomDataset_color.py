import os
import re
import numpy as np
from skimage.io import imread
import torch
from torch.utils.data import Dataset
import skimage
from natsort import natsorted

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
        self.mri_files = natsorted(self.get_img_files(root_dir_mri))
        self.ct_files = natsorted(self.get_img_files(root_dir_ct))
        # self.mri_files = sorted(self.get_img_files(root_dir_mri),key=self.extract_number)
        # self.ct_files = sorted(self.get_img_files(root_dir_ct),key=self.extract_number)

        
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

    # def extract_number(filename):
    #     # 定义可能的正则表达式
    #     regex_patterns = [
    #         r'image_(\d+)',  # 匹配 "image_数字"
    #         r'(\d+)'         # 匹配任何数字
    #     ]
        
    #     for pattern in regex_patterns:
    #         match = re.search(pattern, filename)
    #         if match:
    #             return int(match.group(1))
        
    #     # 如果没有匹配，返回一个非常大的数字
    #     return float('inf')
    def __len__(self):
        return len(self.mri_files)
    
    def rgb2y(self, img):
        y = img[0:1, :, :] * 0.299000 + img[0:1, :, :] * 0.587000 + img[2:3, :, :] * 0.114000
        return y
    
    def __getitem__(self, idx):
        mri_file = self.mri_files[idx]
        ct_file = self.ct_files[idx]
        # print(mri_file)
        # print(ct_file)
        # mri_img = imread(mri_file)
        # if mri_img.shape[-1] != 3:
        #   mri_img = skimage.color.gray2rgb(mri_img)
              
        # mri_img = mri_img.astype(np.float32).transpose(2,0,1)/255.
        # mri_img = self.rgb2y(mri_img)
        
        # ct_img = imread(ct_file)
        # if ct_img.shape[-1] != 3:
        #     ct_img = skimage.color.gray2rgb(ct_img)     
        # ct_img = ct_img.astype(np.float32).transpose(2,0,1)/255.
        # ct_img = self.rgb2y(ct_img)
        mri_img = imread(mri_file)
        if mri_img.shape[-1] != 3:
            mri_img = skimage.color.gray2rgb(mri_img)
                 
        mri_img = mri_img.astype(np.float32).transpose(2,0,1)/255.
        
        mri_img = self.rgb2y(mri_img)
        mri_img_tensor = torch.Tensor(mri_img)
        name = os.path.basename(ct_file)
        ct_img = imread(ct_file)
        if ct_img.shape[-1] != 3:
            ct_img = skimage.color.gray2rgb(ct_img)     
        ct_img = ct_img.astype(np.float32)
        # print(ct_img.shape)
        # print(ct_img.max(),ct_img.min())
        # ct_img = self.rgb2y(ct_img)
        # yuv_Test_IR = rgb2yuv(ct_img)
        # yuv_img = cv2.cvtColor(ct_img, cv2.COLOR_RGB2YUV)
        yuv_img = rgb2yuv(ct_img)
        Test_IR = yuv_img[:,:,0]/255.#.transpose(2,0,1)
 
        uv_Test_IR = yuv_img[:, :,1:3]#.transpose(2,0,1)
        spect_img_tensor = torch.Tensor(Test_IR).unsqueeze(0)

        return mri_img_tensor, spect_img_tensor, uv_Test_IR, name
    
# import os
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt

# def test_dataset():
#     # 设置数据集路径（请根据实际情况修改）
#     root_dir_mri = 'train_img/pets/mri_150'
#     root_dir_ct = 'train_img/pets/pet_150'
    
#     # 创建数据集实例
#     dataset = MyDataset(root_dir_mri, root_dir_ct)
    
#     # 打印数据集长度
#     print(f"Dataset length: {len(dataset)}")
    
#     # 创建数据加载器
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
#     # 遍历数据加载器
#     for i, (mri_img, uv_Test_IR, mri_img_tensor, spect_img_tensor) in enumerate(dataloader):
#         print(f"Batch {i+1}:")
        
#         # 打印图像的形状
#         print(f"  MRI Image Shape: {mri_img.shape}")
#         print(f"  UV Test IR Shape: {uv_Test_IR.shape}")
#         print(f"  MRI Image Tensor Shape: {mri_img_tensor.shape}")
#         print(f"  Spect Image Tensor Shape: {spect_img_tensor.shape}")
        
#         # 可视化图像（只展示前几个图像）
#         if i == 0:  # 只展示第一个批次的图像
#             plt.figure(figsize=(12, 6))
            
#             # 显示 MRI 图像
#             plt.subplot(1, 2, 1)
#             plt.title("MRI Image")
#             plt.imshow(mri_img[0].numpy().transpose(1, 2, 0), cmap='gray')
#             plt.axis('off')
            
#             # 显示 Spectral Image
#             plt.subplot(1, 2, 2)
#             plt.title("Spectral Image")
#             plt.imshow(spect_img_tensor[0].numpy().squeeze(), cmap='gray')
#             plt.axis('off')
            
#             plt.show()
        
#         # 停止测试
#         if i >= 5:  # 只测试前6个批次
#             break

# # 运行测试
# test_dataset()
