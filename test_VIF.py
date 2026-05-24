# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

from net import Signal_Encoder, Signal_Fusion_Decoder
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from utils.CustomDataset_color import MyDataset, yuv2rgb
import numpy as np
import cv2
'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']


batch_size = 1                                                                                                                                                                                                                               
dataset = 'TNO/'   # "RoadScene" "TNO"
# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Encoder = nn.DataParallel(Signal_Encoder()).to(device)
Fusion_Decoder = nn.DataParallel(Signal_Fusion_Decoder()).to(device)

train_ir_dir = "Test/TNO/source_1"
train_vi_dir = "Test/TNO/source_2"

train_dataset = MyDataset(root_dir_mri=train_ir_dir, root_dir_ct=train_vi_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
# for k in range(1,101):
ckpt_path = 'checkpoints/VIF.pth'
Encoder.load_state_dict(torch.load(ckpt_path)['Encoder'])
Fusion_Decoder.load_state_dict(torch.load(ckpt_path)['Fusion_Decoder'])

'''
------------------------------------------------------------------------------
Test
------------------------------------------------------------------------------
'''

torch.backends.cudnn.benchmark = True

for i, (data_IR,data_VIS_y, data_VIS_uv,name) in enumerate(train_loader):
    data_VIS, data_IR = data_VIS_y.cuda(), data_IR.cuda()
    Encoder.eval()
    Fusion_Decoder.eval()
    # print(i)
    with torch.no_grad():
        begin = time.time()
        signal_feature_IR, signal_ewt_IR, signal_mfb_IR = Encoder(data_IR)
        signal_feature_VIS, signal_ewt_VIS, signal_mfb_VIS = Encoder(data_VIS)
        data_Fuse = Fusion_Decoder(signal_feature_IR, signal_ewt_IR, signal_mfb_IR, signal_feature_VIS, signal_ewt_VIS, signal_mfb_VIS)
        end = time.time()
        # print("--------------time:"+str(end-begin))
        
        path = "test_result/"+dataset
        if not os.path.exists(path):
            os.makedirs(path)
        # torchvision.utils.save_image(data_Fuse, path + f'/{i + 1}.bmp')
        # print("save:"+ dataset +f'_fusion{i+1}.bmp')

        for b in range(batch_size):
            data_Fuse_numpy = data_Fuse[b].cpu().detach().numpy().transpose(1, 2, 0) * 255
            tensor2 = data_VIS_uv[b].squeeze(0).cpu().detach().numpy()
            YUV_image = np.concatenate((data_Fuse_numpy, tensor2), axis=2)

            # rgb_image_again = cv2.cvtColor(YUV_image, cv2.COLOR_YUV2RGB)
            rgb_image_again = yuv2rgb(YUV_image)
            # data_Fuse_color = cv2.cvtColor(rgb_image_again, cv2.COLOR_RGB2BGR)
            data_Fuse_color = torch.Tensor(rgb_image_again).permute(2, 0, 1)/255.
            # cv2.imwrite(os.path.join(path,f'/fusion{b}.png'), data_Fuse_color)
            torchvision.utils.save_image(data_Fuse_color,path+name[0])
            print("save:"+ dataset + name[0])




