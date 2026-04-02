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
from CustomDataset import MyDataset

'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

batch_size = 4
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']


Modality = "MIF/"
dataset = '/'
# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Encoder = nn.DataParallel(Signal_Encoder()).to(device)
Fusion_Decoder = nn.DataParallel(Signal_Fusion_Decoder()).to(device)

train_mri_dir = "Dataset/data_AAL_1/t1ce"
train_ct_dir = "Dataset/data_AAL_1/flair"

train_dataset = MyDataset(root_dir_mri=train_mri_dir, root_dir_ct=train_ct_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

ckpt_path = 'models/Phase Ⅴ/VIF/signal_level_PhaseⅤ.pth'
Encoder.load_state_dict(torch.load(ckpt_path)['Encoder'])
Fusion_Decoder.load_state_dict(torch.load(ckpt_path)['Fusion_Decoder'])

'''
------------------------------------------------------------------------------
Test
------------------------------------------------------------------------------
'''

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

for i, (data_VIS, data_IR) in enumerate(train_loader):
    data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
    Encoder.eval()
    Fusion_Decoder.eval()
    with torch.no_grad():
        begin = time.time()
        signal_feature_IR, signal_ewt_IR, signal_mfb_IR = Encoder(data_IR)
        signal_feature_VIS, signal_ewt_VIS, signal_mfb_VIS = Encoder(data_VIS)
        data_Fuse = Fusion_Decoder(signal_feature_IR, signal_ewt_IR, signal_mfb_IR, signal_feature_VIS, signal_ewt_VIS, signal_mfb_VIS)
        end = time.time()
        print("--------------time:"+str(end-begin))

        # if i %100 ==0:
        #     path = "test_result/PhaseⅤ/"+Modality+dataset
        #     if not os.path.exists(path):
        #         os.makedirs(path)
        #     torchvision.utils.save_image(data_Fuse, path+f'/{i+1}.bmp')
        #     print("save:"+f'/fusion{i+1}.bmp')






