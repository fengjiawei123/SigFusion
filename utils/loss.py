import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.filters import threshold_otsu
from torch.autograd import Variable
from math import exp
# import kornia
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 /
                         float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).double().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window

class FFTLoss(nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()

    def forward(self, input_magnitude, input_phase, target_magnitude, target_phase):
        # 计算幅度上的损失
        magnitude_loss = torch.mean((input_magnitude - target_magnitude) ** 2)
        # 计算相位上的损失
        phase_loss = torch.mean((input_phase - target_phase) ** 2)
        # 总损失为幅度损失和相位损失之和
        total_loss = magnitude_loss + phase_loss
        return total_loss

class L_Intensity_Max_RGB(nn.Module):
    def __init__(self):
        super(L_Intensity_Max_RGB, self).__init__()

    def forward(self, image_visible, image_infrared, image_fused, max_mode="l1"):
        gray_visible = torch.mean(image_visible, dim=1, keepdim=True)
        gray_infrared = torch.mean(image_infrared, dim=1, keepdim=True)

        mask = (gray_infrared > gray_visible).float()

        fused_image = mask * image_infrared + (1 - mask) * image_visible
        if max_mode == "l1":
            Loss_intensity = F.l1_loss(fused_image, image_fused)
        else:
            Loss_intensity = F.mse_loss(fused_image, image_fused)
        return Loss_intensity
class Fusionloss(nn.Module):
    def __init__(self,coeff_int=1,coeff_grad=10,in_max=True, device='cuda'):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy(device=device)
        self.coeff_int=coeff_int
        self.coeff_grad=coeff_grad
        self.in_max=in_max
    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        if self.in_max:
            x_in_max=torch.max(image_y,image_ir)
        else:
            x_in_max=(image_y+image_ir)/2.0
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=self.coeff_int*loss_in+self.coeff_grad*loss_grad
        return loss_total,loss_in,loss_grad    
# class Fusionloss(nn.Module):
#     def __init__(self):
#         super(Fusionloss, self).__init__()
#         self.sobelconv=Sobelxy()

#     def forward(self,image_vis,image_ir,generate_img):
#         image_y=image_vis[:,:1,:,:]
#         x_in_max=torch.max(image_y,image_ir)
#         loss_in=F.l1_loss(x_in_max,generate_img)
#         y_grad=self.sobelconv(image_y)
#         ir_grad=self.sobelconv(image_ir)
#         generate_img_grad=self.sobelconv(generate_img)
#         x_grad_joint=torch.max(y_grad,ir_grad)
#         loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
#         loss_total=loss_in+10*loss_grad
#         return loss_total,loss_in,loss_grad
def listTotensor(signal_ewt, fusion_ewt):
    signal_high_all = []
    fusion_ewt_all = []
    for j in range(len(signal_ewt[0])):
        for i in range(5):
            signal_high = torch.tensor(signal_ewt[i][j][1:2])
            signal_high_all.append(signal_high)
            fusion_ewt_high = torch.tensor(fusion_ewt[i][j][1:2])
            fusion_ewt_all.append(fusion_ewt_high)
    signal_high_all = torch.stack(signal_high_all)  # 变成 (5 * len(signal_ewt[0]), 1024)
    signal_high_tensor = signal_high_all.view(1,-1)
    fusion_ewt_high_all = torch.stack(fusion_ewt_all)  # 变成 (5 * len(signal_ewt[0]), 1024)
    fusion_ewt_high_tensor = fusion_ewt_high_all.view(1,-1)
    return signal_high_tensor, fusion_ewt_high_tensor
class LSDloss(nn.Module):
    def __init__(self, nfft=512):
        super(LSDloss, self).__init__()
        self.nfft = nfft

    def forward(self, y_true_s, y_est_s):
        # 计算 STFT (Short-Time Fourier Transform)
        yt_spec = torch.stft(y_true_s, n_fft=self.nfft, return_complex=True)
        ye_spec = torch.stft(y_est_s, n_fft=self.nfft, return_complex=True)

        # 计算功率谱密度
        yt_spec = torch.log10(yt_spec.abs() ** 2 + 1e-10)  # 加 1e-10 以避免 log(0)
        ye_spec = torch.log10(ye_spec.abs() ** 2 + 1e-10)

        # 计算 LSD 损失
        lsd_loss = torch.mean(torch.sqrt(torch.mean((yt_spec - ye_spec) ** 2, dim=-1)))

        return lsd_loss
def get_threshold(feature_map):
    # 计算特征图的均值和标准差
    mean_val = feature_map.mean().item()
    std_val = feature_map.std().item()
    
    # 设定一个基于均值和标准差的初步阈值
    threshold = mean_val + std_val
    
    # 确保阈值在0.5到0.7之间
    if threshold < 0.5:
        threshold = 0.5
    elif threshold > 0.7:
        threshold = 0.7
    
    return threshold



class Seg_Fusionloss(nn.Module):
    def __init__(self):
        super(Seg_Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        level_mri = get_threshold(image_ir)
        Seg_ir_image = torch.where(image_ir > level_mri, torch.tensor(1.0), torch.tensor(0.0))
        gt_ir_image = torch.mul(image_ir,Seg_ir_image)
        gt_vis_image = torch.mul(image_vis,1-Seg_ir_image)
        gt_image = gt_ir_image + gt_vis_image
        loss=F.l1_loss(gt_image,generate_img)
        # image_y=image_vis[:,:1,:,:]
        # x_in_max=torch.max(image_y,image_ir)
        # loss_in=F.l1_loss(x_in_max,generate_img)
        # y_grad=self.sobelconv(image_y)
        # ir_grad=self.sobelconv(image_ir)
        # generate_img_grad=self.sobelconv(generate_img)
        # x_grad_joint=torch.max(y_grad,ir_grad)
        # loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        # loss_total=loss_in+10*loss_grad
        return loss
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()
def Get_threshold(feature_map):
    # 计算特征图的均值和标准差
    mean_val = feature_map.mean().item()
    std_val = feature_map.std().item()
    
    # 设定一个基于均值和标准差的初步阈值
    threshold = mean_val + std_val
    
    # 确保阈值在0.5到0.7之间
    if threshold < 0.4:
        threshold = 0.4
    elif threshold > 0.5:
        threshold = 0.5
    return threshold
MSELoss = nn.MSELoss()
# Loss_ssim = kornia.losses.SSIM(11, reduction='mean')
class  color_Fusionloss(nn.Module):
    def __init__(self):
        super(color_Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()
    def forward(self,image_vis,image_ir,generate_img):
        level_mri = Get_threshold(image_ir)
        Seg_ir_image = torch.where(image_ir > level_mri, torch.tensor(1.0), torch.tensor(0.0))
        gt_ir_image = torch.mul(image_ir,Seg_ir_image)
        gt_VIS_image = torch.mul(image_vis,Seg_ir_image)
        gt_vis_image = torch.mul(image_vis,1-Seg_ir_image)
        gt_image = gt_ir_image*0.5 + gt_vis_image + gt_VIS_image*0.5
        loss_ssim=F.l1_loss(gt_image,generate_img)
        # loss_mse = MSELoss(gt_image,generate_img)
        return loss_ssim #+ loss_mse
# class Fusionloss(nn.Module):
#     def __init__(self,coeff_int=1,coeff_grad=10,in_max=True, device='cuda'):
#         super(Fusionloss, self).__init__()
#         self.sobelconv=Sobelxy(device=device)
#         self.coeff_int=coeff_int
#         self.coeff_grad=coeff_grad
#         self.in_max=in_max
#     def forward(self,image_vis,image_ir,generate_img):
#         image_y=image_vis[:,:1,:,:]
#         if self.in_max:
#             x_in_max=torch.max(image_y,image_ir)
#         else:
#             x_in_max=(image_y+image_ir)/2.0
#         loss_in=F.l1_loss(x_in_max,generate_img)
#         y_grad=self.sobelconv(image_y)
#         ir_grad=self.sobelconv(image_ir)
#         generate_img_grad=self.sobelconv(generate_img)
#         x_grad_joint=torch.max(y_grad,ir_grad)
#         loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
#         loss_total=self.coeff_int*loss_in+self.coeff_grad*loss_grad
#         return loss_total,loss_in,loss_grad

class Sobelxy(nn.Module):
    def __init__(self,device='cuda'):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


def _mef_ssim(X, Ys, window, ws, denom_g, denom_l, C1, C2, is_lum=False, full=False):
    K, C, H, W = list(Ys.size())

    # compute statistics of the reference latent image Y
    muY_seq = F.conv2d(Ys, window, padding=ws // 2, groups=C).view(K, C, H, W)
    muY_sq_seq = muY_seq * muY_seq
    sigmaY_sq_seq = F.conv2d(Ys * Ys, window, padding=ws // 2, groups=C).view(K, C, H, W) \
        - muY_sq_seq
    sigmaY_sq, patch_index = torch.max(sigmaY_sq_seq, dim=0)

    # compute statistics of the test image X
    muX = F.conv2d(X, window, padding=ws // 2, groups=C).view(C, H, W)
    muX_sq = muX * muX
    sigmaX_sq = F.conv2d(X * X, window, padding=ws // 2,
                         groups=C).view(C, H, W) - muX_sq

    # compute correlation term
    sigmaXY = F.conv2d(X.expand_as(Ys) * Ys, window, padding=ws // 2, groups=C).view(K, C, H, W) \
        - muX.expand_as(muY_seq) * muY_seq

    # compute quality map
    cs_seq = (2 * sigmaXY + C2) / (sigmaX_sq + sigmaY_sq_seq + C2)
    cs_map = torch.gather(cs_seq.view(K, -1), 0,
                          patch_index.view(1, -1)).view(C, H, W)
    if is_lum:
        lY = torch.mean(muY_seq.view(K, -1), dim=1)
        lL = torch.exp(-((muY_seq - 0.5) ** 2) / denom_l)
        lG = torch.exp(- ((lY - 0.5) ** 2) /
                       denom_g)[:, None, None].expand_as(lL)
        LY = lG * lL
        muY = torch.sum((LY * muY_seq), dim=0) / torch.sum(LY, dim=0)
        muY_sq = muY * muY
        l_map = (2 * muX * muY + C1) / (muX_sq + muY_sq + C1)
    else:
        l_map = torch.Tensor([1.0])
        if Ys.is_cuda:
            l_map = l_map.cuda(Ys.get_device())

    if full:
        l = torch.mean(l_map)
        cs = torch.mean(cs_map)
        return l, cs

    qmap = l_map * cs_map
    q = qmap.mean()

    return q



class MEFSSIM(torch.nn.Module):
    def __init__(self, window_size=11, channel=3, sigma_g=0.2, sigma_l=0.2, c1=0.01, c2=0.03, is_lum=False):
        super(MEFSSIM, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = create_window(window_size, self.channel)
        self.denom_g = 2 * sigma_g**2
        self.denom_l = 2 * sigma_l**2
        self.C1 = c1**2
        self.C2 = c2**2
        self.is_lum = is_lum

    def forward(self, X, Ys):
        (_, channel, _, _) = Ys.size()

        if channel == self.channel and self.window.data.type() == Ys.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if Ys.is_cuda:
                window = window.cuda(Ys.get_device())
            window = window.type_as(Ys)

            self.window = window
            self.channel = channel

        return _mef_ssim(X, Ys, window, self.window_size,
                         self.denom_g, self.denom_l, self.C1, self.C2, self.is_lum)


class LpLssimLossweight(nn.Module):
    def __init__(self, window_size=5, size_average=True):
        """
            Constructor
        """
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        """
            Get the gaussian kernel which will be used in SSIM computation
        """
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        """
            Create the gaussian window
        """
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)   # [window_size, 1]
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0) # [1,1,window_size, window_size]
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        """
            Compute the SSIM for the given two image
            The original source is here: https://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow
        """
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, image_in, image_out, weight):

        # Check if need to create the gaussian window
        (_, channel, _, _) = image_in.size()
        if channel == self.channel and self.window.data.type() == image_in.data.type():
            pass
        else:
            window = self.create_window(self.window_size, channel)
            window = window.to(image_out.get_device())
            window = window.type_as(image_in)
            self.window = window
            self.channel = channel

        # Lp
        Lp = torch.sqrt(torch.sum(torch.pow((image_in - image_out), 2)))  # 二范数
        # Lp = torch.sum(torch.abs(image_in - image_out))  # 一范数
        # Lssim
        Lssim = 1 - self._ssim(image_in, image_out, self.window, self.window_size, self.channel, self.size_average)
        return Lp + Lssim * weight, Lp, Lssim * weight    