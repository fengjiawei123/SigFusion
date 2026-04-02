# import numpy as np
# import torch

# # fft
# def fft(img):
#     # img = img.npu()  # 将张量移回 npU
#     # img = img.detach()  # 将张量从计算图中分离出来
#     # img = img.numpy()  # 转换为 numpy 数组
#     # f_transform = np.fft.fft2(img)
#     # f_shift = np.fft.fftshift(f_transform)
#     # # 计算频谱
#     # fre_m = np.log(np.abs(f_shift) + 1e-8)
#     # # 计算相位角
#     # fre_p = np.angle(f_shift)
#     fre = torch.fft.fft2(img)
#     fre_m = torch.abs(fre)   #幅度谱，求模得到；
#     fre_p = torch.angle(fre)
#     return fre_m, fre_p
#     # return torch.tensor(fre_m), torch.tensor(fre_p)
import torch
import numpy as np

class FFTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img):
        # 将张量从计算图中分离并转换为 numpy 数组
        img_np = img.detach().npu().numpy()

        # 检查图像的形状，并在必要时进行调整
        if img_np.ndim == 4:  # (batch_size, channels, height, width)
            img_np = img_np[0, 0, :, :]
        elif img_np.ndim == 3:  # (channels, height, width)
            img_np = img_np[0, :, :]

        # 计算 FFT
        fre = np.fft.fft2(img_np, axes=(0, 1))
        fre_m = np.abs(fre)
        fre_p = np.angle(fre)

        # 保存需要在反向传播中使用的变量
        ctx.save_for_backward(torch.tensor(fre, dtype=torch.complex64))

        # 转换为 PyTorch 张量
        fre_m = torch.tensor(fre_m, dtype=torch.float32).to(img.device)
        fre_p = torch.tensor(fre_p, dtype=torch.float32).to(img.device)

        return fre_m, fre_p

    @staticmethod
    def backward(ctx, grad_fre_m, grad_fre_p):
        fre, = ctx.saved_tensors

        # 将梯度从 GPU 移动到 npU 并转换为 numpy 数组
        grad_fre_m_np = grad_fre_m.npu().numpy()
        grad_fre_p_np = grad_fre_p.npu().numpy()

        # 计算复数梯度
        fre_grad = grad_fre_m_np + 1j * grad_fre_p_np

        # 调试信息：打印输入形状
        # print(f"fre_grad shape: {fre_grad.shape}")

        # 检查 fre_grad 形状是否正确
        if len(fre_grad.shape) != 2:
            raise ValueError(f"Expected fre_grad to be 2D, but got shape {fre_grad.shape}")

        # 计算逆 FFT
        img_grad = np.fft.ifft2(fre_grad, axes=(0, 1)).real

        # 调整 img_grad 的形状，使其与原始输入张量形状匹配
        img_grad = img_grad[np.newaxis, np.newaxis, :, :]  # 将 (height, width) 转换为 (1, 1, height, width)

        # 转换为 PyTorch 张量并移动回原始设备
        img_grad = torch.tensor(img_grad, dtype=torch.float32).to(grad_fre_m.device)

        return img_grad


# 使用自定义的 FFTFunction
def fft(img):
    return FFTFunction.apply(img)

    
    
