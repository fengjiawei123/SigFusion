import os
import h5py
import numpy as np
from tqdm import tqdm
from skimage.io import imread


def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tif', '.npy')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist
    
def rgb2y(img):
    y = img[0:1, :, :] * 0.299000 + img[1:2, :, :] * 0.587000 + img[2:3, :, :] * 0.114000
    return y

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def is_low_contrast(image, fraction_threshold=0.1, lower_percentile=10,
                    upper_percentile=90):
    """Determine if an image is low contrast."""
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / limits[1]
    return ratio < fraction_threshold

data_name="MSRS_train"
img_size=128   #patch size
stride=200     #patch stride
h5f = h5py.File(os.path.join('data',
                                 data_name+'_imgsize_'+str(img_size)+"_stride_"+str(stride)+'.h5'),
                    'w')
h5_ir = h5f.create_group('ir_patchs')
h5_vis = h5f.create_group('vis_patchs')
# h5_fusion = h5f.create_group('fusion_patchs')
train_num = 0
# for fusion_name in ['MSRS_MetaFusion', 'MSRS_DIDFuse', 'MSRS_CDDFuse', 'MSRS_fusionGAN', 'MSRS_GANMcC', 'MSRS_LRRNet','MSRS_EMMA' , 'MSRS_MUFusion', 'MSRS_SDNet', 'MSRS_U2fusion']:

IR_files = sorted(get_img_file(r"Datasets/MSRS_train/ir"))
VIS_files = sorted(get_img_file(r"Datasets/MSRS_train/vi"))
# fusion_files = sorted(get_img_file(r"data/MSRS_10/" + fusion_name))

assert len(IR_files) == len(VIS_files)
if 'ir_patchs' not in h5f:
    h5_ir = h5f.create_group('ir_patchs')
else:
    h5_ir = h5f['ir_patchs']

if 'vis_patchs' not in h5f:
    h5_vis = h5f.create_group('vis_patchs')
else:
    h5_vis = h5f['vis_patchs']

# if 'fusion_patchs' not in h5f:
#     h5_fusion = h5f.create_group('fusion_patchs')
# else:
#     h5_fusion = h5f['fusion_patchs']

for i in tqdm(range(len(IR_files))):
        I_VIS = imread(VIS_files[i]).astype(np.float32).transpose(2,0,1)/255. # [3, H, W] Uint8->float32
        I_VIS = rgb2y(I_VIS) # [1, H, W] Float32
        I_IR = imread(IR_files[i]).astype(np.float32)[None, :, :]/255.  # [1, H, W] Float32
        # I_fusion = imread(fusion_files[i]).astype(np.float32).transpose(2, 0, 1) / 255.  # [3, H, W] Uint8->float32
        # I_fusion = rgb2y(I_fusion)  # [1, H, W] Float32
        # 处理fusion图像
        # I_fusion = imread(fusion_files[i]).astype(np.float32) / 255.  # 先读取并转换为浮点型
        # if I_fusion.ndim == 3:  # 如果是RGB图像
        #     I_fusion = I_fusion.transpose(2, 0, 1)  # [3, H, W] Uint8->float32
        # elif I_fusion.ndim == 2:  # 如果是灰度图像
        #     I_fusion = I_fusion[None, :, :]  # 添加一个维度，使其成为 [1, H, W]

        # I_fusion = rgb2y(I_fusion)  # [1, H, W] Float32
        # crop
        I_IR_Patch_Group = Im2Patch(I_IR,img_size,stride)
        I_VIS_Patch_Group = Im2Patch(I_VIS, img_size, stride)  # (3, 256, 256, 12)
        # I_fusion_Patch_Group = Im2Patch(I_fusion, img_size, stride)  # (3, 256, 256, 12)
        for ii in range(I_IR_Patch_Group.shape[-1]):
            bad_IR = is_low_contrast(I_IR_Patch_Group[0,:,:,ii])
            bad_VIS = is_low_contrast(I_VIS_Patch_Group[0,:,:,ii])
            # bad_fusion = is_low_contrast(I_fusion_Patch_Group[0,:,:,ii])
            # Determine if the contrast is low
            if not (bad_IR or bad_VIS):
                avl_IR= I_IR_Patch_Group[0,:,:,ii]  #  available IR
                avl_VIS= I_VIS_Patch_Group[0,:,:,ii]
                # avl_fusion = I_fusion_Patch_Group[0,:,:,ii]
                avl_IR=avl_IR[None,...]
                avl_VIS=avl_VIS[None,...]
                # avl_fusion=avl_fusion[None,...]

                h5_ir.create_dataset(str(train_num),     data=avl_IR,
                                dtype=avl_IR.dtype,   shape=avl_IR.shape)
                h5_vis.create_dataset(str(train_num),    data=avl_VIS,
                                dtype=avl_VIS.dtype,  shape=avl_VIS.shape)
                # h5_fusion.create_dataset(str(train_num), data=avl_fusion,
                #                       dtype=avl_fusion.dtype, shape=avl_fusion.shape)
                train_num += 1

h5f.close()

with h5py.File(os.path.join('data',
                                 data_name+'_imgsize_'+str(img_size)+"_stride_"+str(stride)+'.h5'),"r") as f:
    for key in f.keys():
        print(f[key], key, f[key].name)
    
# data_name = "MSRS_train_fusion"
# h5f = h5py.File(os.path.join('.\\data', data_name + '_fullsize.h5'), 'w')
#
# # 创建数据集组
# h5_ir = h5f.create_group('ir_images')
# h5_vis = h5f.create_group('vis_images')
# h5_fusion = h5f.create_group('fusion_images')
#
# train_num = 0
#
# for fusion_name in ['MSRS_MetaFusion']:#, 'MSRS_DIDFuse', 'MSRS_CDDFuse', 'MSRS_fusionGAN', 'MSRS_GANMcC', 'MSRS_LRRNet','MSRS_EMMA' , 'MSRS_MUFusion', 'MSRS_SDNet', 'MSRS_U2fusion']:
#
#     IR_files = sorted(get_img_file(r"data/MSRS_10/MSRS_train/ir"))
#     VIS_files = sorted(get_img_file(r"data/MSRS_10/MSRS_train/vi"))
#     fusion_files = sorted(get_img_file(r"data/MSRS_10/" + fusion_name))
#
#     assert len(IR_files) == len(VIS_files)
#
#     for i in tqdm(range(len(IR_files))):
#         # 读取和预处理可见光和红外图像
#         I_VIS = imread(VIS_files[i]).astype(np.float32).transpose(2, 0, 1) / 255.  # [3, H, W] Uint8 -> Float32
#         I_VIS = rgb2y(I_VIS)  # [1, H, W] 转换为灰度
#
#         I_IR = imread(IR_files[i]).astype(np.float32)[None, :, :] / 255.  # [1, H, W] 转换为Float32
#
#         # 读取并处理融合图像
#         I_fusion = imread(fusion_files[i]).astype(np.float32) / 255.  # 将融合图像转换为Float32
#         if I_fusion.ndim == 3:  # 如果是RGB图像
#             I_fusion = I_fusion.transpose(2, 0, 1)  # [3, H, W]
#         elif I_fusion.ndim == 2:  # 如果是灰度图像
#             I_fusion = I_fusion[None, :, :]  # [1, H, W]
#         I_fusion = rgb2y(I_fusion)  # 转换为灰度图像 [1, H, W]
#
#         # 将整张图像存储到HDF5中
#         h5_ir.create_dataset(str(train_num), data=I_IR, dtype=I_IR.dtype, shape=I_IR.shape)
#         h5_vis.create_dataset(str(train_num), data=I_VIS, dtype=I_VIS.dtype, shape=I_VIS.shape)
#         h5_fusion.create_dataset(str(train_num), data=I_fusion, dtype=I_fusion.dtype, shape=I_fusion.shape)
#
#         train_num += 1
#
# h5f.close()
#
# # 检查生成的 HDF5 文件
# with h5py.File(os.path.join('data', data_name + '_fullsize.h5'), "r") as f:
#     for key in f.keys():
#         print(f[key], key, f[key].name)
    



    
