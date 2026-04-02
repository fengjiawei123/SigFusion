# import os
# import numpy as np
# from skimage.io import imread
# import torch
# from torch.utils.data import Dataset

# class MyDataset(Dataset):
#     def __init__(self, root_dir_mri, root_dir_ct, transform=None):
#         self.root_dir_mri = root_dir_mri
#         self.root_dir_ct = root_dir_ct
#         self.transform = transform
        
#         self.mri_files = sorted(self.get_img_files(root_dir_mri))
#         self.ct_files = sorted(self.get_img_files(root_dir_ct))
        
#         assert len(self.mri_files) == len(self.ct_files)
        
#     def get_img_files(self, root_dir):
#         imagelist = []
#         for parent, _, filenames in os.walk(root_dir):
#             for filename in filenames:
#                 if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')):
#                     imagelist.append(os.path.join(parent, filename))
#         return imagelist
    
#     def __len__(self):
#         return len(self.mri_files)
    
#     def rgb2y(self, img):
#         y = img[0:1, :, :] * 0.299000 + img[0:1, :, :] * 0.587000 + img[2:3, :, :] * 0.114000
#         return y
    
#     def __getitem__(self, idx):
#         mri_file = self.mri_files[idx]
#         ct_file = self.ct_files[idx]
        
#         mri_img = imread(mri_file).astype(np.float32).transpose(2,0,1)/255.

#         mri_img = self.rgb2y(mri_img)

#         ct_img = imread(ct_file).astype(np.float32).transpose(2,0,1)/255.
#         ct_img = self.rgb2y(ct_img)

#         if self.transform:
#             mri_img = self.transform(mri_img)
#             ct_img = self.transform(ct_img)
        
#         return torch.Tensor(mri_img), torch.Tensor(ct_img)
import os
import re
import numpy as np
from skimage.io import imread
import torch
from torch.utils.data import Dataset
import skimage
from torchvision import transforms
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
        # 提取文件名中的数字部分进行排序
        # match = re.search(r'image_(\d+)', filename)
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
        
        ct_img = imread(ct_file)
        if ct_img.shape[-1] != 3:
            ct_img = skimage.color.gray2rgb(ct_img)     
        ct_img = ct_img.astype(np.float32).transpose(2,0,1)/255.
        ct_img = self.rgb2y(ct_img)
        ct_img_tensor = torch.Tensor(ct_img)
        

        # return torch.Tensor(mri_img), torch.Tensor(ct_img)
        return mri_img_tensor, ct_img_tensor
    
# class MyDataset_CT(Dataset):
#     def __init__(self, root_dir_ct, transform_ct=None, mode='none'):
   
#         self.root_dir_ct = root_dir_ct
#         self.transform_ct = transform_ct
#         self.mode = mode
#         self.ct_files = sorted(self.get_img_files(root_dir_ct))
        
#     def get_img_files(self, root_dir):
#         imagelist = []
#         for parent, _, filenames in os.walk(root_dir):
#             for filename in filenames:
#                 if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')):
#                     imagelist.append(os.path.join(parent, filename))
#         return imagelist

#     def __len__(self):
#         return len(self.ct_files)
    
#     def rgb2y(self, img):
#         y = img[0:1, :, :] * 0.299000 + img[0:1, :, :] * 0.587000 + img[2:3, :, :] * 0.114000
#         return y
    
#     def __getitem__(self, idx):
#         ct_file = self.ct_files[idx]

#         # mri_img = imread(mri_file)
#         # if mri_img.shape[-1] != 3:
#         #   mri_img = skimage.color.gray2rgb(mri_img)
              
#         # mri_img = mri_img.astype(np.float32).transpose(2,0,1)/255.
#         # mri_img = self.rgb2y(mri_img)
        
#         # ct_img = imread(ct_file)
#         # if ct_img.shape[-1] != 3:
#         #     ct_img = skimage.color.gray2rgb(ct_img)     
#         # ct_img = ct_img.astype(np.float32).transpose(2,0,1)/255.
#         # ct_img = self.rgb2y(ct_img)


#         ct_img = imread(ct_file)
#         if ct_img.shape[-1] != 3:
#             ct_img = skimage.color.gray2rgb(ct_img)     
#         ct_img = ct_img.astype(np.float32).transpose(2,0,1)/255.
#         ct_img = self.rgb2y(ct_img)
#         ct_img_tensor = torch.Tensor(ct_img)





#         # return torch.Tensor(mri_img), torch.Tensor(ct_img)
#         return  ct_img,  ct_img_tensor
    
    
# class MyDataset_MRI(Dataset):
#     def __init__(self, root_dir_mri , transform_mri=None, mode='none'):
#         self.root_dir_mri = root_dir_mri

#         self.transform_mri = transform_mri
    
#         self.mode = mode
#         self.mri_files = sorted(self.get_img_files(root_dir_mri))
      
      
        
#     def get_img_files(self, root_dir):
#         imagelist = []
#         for parent, _, filenames in os.walk(root_dir):
#             for filename in filenames:
#                 if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')):
#                     imagelist.append(os.path.join(parent, filename))
#         return imagelist

#     def __len__(self):
#         return len(self.mri_files)
    
#     def rgb2y(self, img):
#         y = img[0:1, :, :] * 0.299000 + img[0:1, :, :] * 0.587000 + img[2:3, :, :] * 0.114000
#         return y
    
#     def __getitem__(self, idx):
#         mri_file = self.mri_files[idx]


#         # mri_img = imread(mri_file)
#         # if mri_img.shape[-1] != 3:
#         #   mri_img = skimage.color.gray2rgb(mri_img)
              
#         # mri_img = mri_img.astype(np.float32).transpose(2,0,1)/255.
#         # mri_img = self.rgb2y(mri_img)
        
#         # ct_img = imread(ct_file)
#         # if ct_img.shape[-1] != 3:
#         #     ct_img = skimage.color.gray2rgb(ct_img)     
#         # ct_img = ct_img.astype(np.float32).transpose(2,0,1)/255.
#         # ct_img = self.rgb2y(ct_img)
#         mri_img = imread(mri_file)
#         if mri_img.shape[-1] != 3:
#             mri_img = skimage.color.gray2rgb(mri_img)
                 
#         mri_img = mri_img.astype(np.float32).transpose(2,0,1)/255.
#         mri_img = self.rgb2y(mri_img)
#         mri_img_tensor = torch.Tensor(mri_img)
        

        


#         # return torch.Tensor(mri_img), torch.Tensor(ct_img)
#         return mri_img, mri_img_tensor