import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import numpy as np
import math
from einops import rearrange
from EWT.ewt.ewt1d import ewt1d, iewt1d
from EWT.ewt.utilities import ewt_params
from utils.MLP import *
# from self_attention_cv import TransformerEncoder
# from ewtpy import EWT1D, IEWT1D
# 定义自定义的 EWT 前向和反向传播类
class EWTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, params):
        f_np = f.cpu().numpy()
        ewt, mfb, bounds = ewt1d(f_np, params)
        ewt_input = ewt
        mfb_input = mfb
        if len(ewt) !=3:
            ewt_input = np.concatenate([ewt, np.copy(ewt[-1:])], axis=0)
            mfb_input = np.concatenate([mfb, np.copy(mfb[-1:])], axis=0)
        ewt = ewt_input
        mfb = mfb_input
        ctx.save_for_backward(f.clone())
        ctx.mfb = mfb
        ctx.bounds = bounds
        ewt_np = np.array(ewt)
        ewt_torch = torch.tensor(ewt_np).cuda()
        return ewt_torch, mfb

    @staticmethod
    def backward(ctx, grad_output, grad_mfb):
        f = ctx.saved_tensors
        mfb = ctx.mfb
        
        grad_output_np = grad_output.cpu().numpy()
        
        grad_input_np = iewt1d(grad_output_np, mfb)
        
        grad_input = torch.tensor(grad_input_np).cuda()
        # print(grad_input.shape)
        
        # 返回 forward 中所有输入的梯度
        return grad_input, None




# 定义自定义的 IEWT 前向和反向传播类
class IEWTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ewt, mfb, params):
        # 将 EWT 系数从 torch.Tensor 转换为 numpy 数组
        ewt_np = ewt.detach().cpu().numpy()
        # print(ewt.shape)
        # 调用已有的 iewt1d 函数
        rec = iewt1d(ewt_np, mfb.detach().cpu().numpy())
        
        # 使用 clone() 来避免就地修改
        ctx.save_for_backward(ewt.clone(), mfb.clone())
        ctx.params = params 
        # print(ctx.shape)
        # 返回重建后的信号
        rec_np = np.array(rec)
        rec_torch = torch.tensor(rec_np).cuda()
        return rec_torch

    @staticmethod
    def backward(ctx, grad_output):
        # 获取前向传播中保存的 EWT 系数和滤波器
        ewt, mfb = ctx.saved_tensors
        # print("grad_output:",grad_output.shape)
        # 获取保存的 params
        params = ctx.params
        
        # 将 grad_output 转换为 numpy 格式
        grad_output_np = grad_output.cpu().detach().numpy()
        # print("grad_output_np:",grad_output_np.shape)
        # 使用 ewt1d 来计算逆向传播的梯度
        grad_input_np, _, _ = ewt1d(grad_output_np, params)
        if isinstance(grad_input_np, list):
            grad_input_np = np.array(grad_input_np)

        # 将结果转换回 torch.Tensor 格式并返回
        grad_input = torch.tensor(grad_input_np).cuda()
        # print("grad_input:",grad_input.shape)
        # 返回 3 个梯度：ewt, mfb, 和 params 的梯度
        if len(grad_input) !=3:
            grad_input = torch.cat([grad_input, grad_input[-1:].clone()], dim=0)
        grad_ewt = grad_input #if grad_input.shape == ewt.shape else grad_input.view(ewt.shape)
        grad_mfb = None
        grad_params = None
        
        return grad_ewt, grad_mfb, grad_params

class EWT(nn.Module):
    def __init__(
            self,
            inp_channels=1,
            dim=32,
    ):
        super().__init__()
        self.channel = inp_channels
        self.dim = dim
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=inp_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=inp_channels, out_channels=dim, kernel_size=3, padding=1)
        self.len_to_32 = lenth_transfer1(inp_channels=inp_channels)
        self.len_to_hw = lenth_transfer2(inp_channels=inp_channels)
        # MLP 用于处理 EWT 变换后的特征

    def forward(self, image):
        b,c,h,w = image.size()
        # 通过卷积层缩小通道
        feature = self.conv1(image)

        # 将特征从二维转换为一维
        feature = self.len_to_32(feature)
        feature = feature.view(b,-1)
        # 调用 EWT，自定义的 EWT 前向和反向传播函数
        ewt_batch, mfb_batch = [], []
        params = ewt_params()
        params.log = 1
        params.N = 2
        params.detect = 'locmaxminf'
        params.typeDetect = 'otsu'
        for b_idx in range(b):
            feature_w = feature[b_idx, :]
            # params.removeTrends = 'opening' 
            # 使用自定义 EWT 前向传播操作，确保可以反向传播
            ewt_output, mfb_output = EWTFunction.apply(feature_w, params)
            ewt_batch.append(ewt_output)
            mfb_batch.append(mfb_output)
        # 将 EWT 的第一个系数作为特征输入到 MLP 中进行处理
        feature_l = [sublist[0] for sublist in ewt_batch]
        feature_l = torch.stack(feature_l)

        # 将一维特征输入到 MLP 进行进一步处理
        feature_l = feature_l.view(b, -1, 32, 32)  # 调整为适合卷积的形状

        # 通过卷积将特征转换回图像
        feature_l = self.len_to_hw(feature_l.float(),h,w)
        feature_l = self.conv2(feature_l)

        return feature_l, ewt_batch, mfb_batch
class IEWT(nn.Module):
    def __init__(
            self,
            dim=32,
            signal_batch=32,
            inp_channels=1,
            in_channels=1024,
            out_channels=1024,
    ):
        super().__init__()
        self.signal_batch = signal_batch
        self.len_to_32 = lenth_transfer1(inp_channels=inp_channels)
        self.len_to_hw = lenth_transfer2(inp_channels=inp_channels)
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=inp_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=inp_channels, out_channels=dim, kernel_size=3, padding=1)
        # 定义 MLP 结构用于处理 EWT 特征
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels)
        )

    def forward(self, feature_low, signal_ewt, signal_mfb):
        feature_low = self.conv1(feature_low)
        # 将特征转换为 1D
        _,_,h,w = feature_low.size()
        feature_low = self.len_to_32(feature_low)
        feature_low = feature_low.view(len(signal_ewt),-1)
        # **去掉 detach()，以确保梯度传播** 
        feature_low = feature_low.float()  # 确保特征是浮点类型

        features = []
        signal_ewt = torch.stack(signal_ewt)
        params = ewt_params()
        params.N = 2
        params.log = 1
        params.detect = 'locmaxminf'
        params.typeDetect = 'otsu'
        # params.removeTrends = 'opening'
        for b_idx in range(len(signal_ewt)):
            # 将 feature_low 替换为 `signal_ewt` 的第一个分量
            signal_ewt[b_idx][0] = feature_low[b_idx]
            # 调用自定义 IEWTFunction 进行反向传播
            rec_feature = IEWTFunction.apply(signal_ewt[b_idx], signal_mfb[b_idx], params)
            features.append(rec_feature)

        # 将列表转换为张量并放置在 CUDA 上
        feature = torch.stack(features).float()

        # 调整特征形状为适合卷积的形式
        feature = feature.view(len(signal_ewt), 1, self.signal_batch, self.signal_batch)

        # 将特征转换回原始的图像形状
        feature = self.len_to_hw(feature,h,w)
        feature = self.conv2(feature)
        return feature

class TransformerEWTBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerEWTBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.ewt = EWT()
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        feature_low, ewt_output, mfb_output = self.ewt(x)
        x = x + feature_low
        x = x + self.ffn(self.norm2(x))

        return x, ewt_output, mfb_output


class TransformerIEWTBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerIEWTBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.iewt = IEWT()
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, ewt_output, mfb_output):
        mfb_output = np.array(mfb_output)
        rec_feature = self.iewt(x, ewt_output, torch.tensor(mfb_output))
        x = x + rec_feature
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x




class Signal_Encoder(nn.Module):
    def __init__(
            self,
            width=256,
            high=256,
            dim=32,
            inp_channels=1,
            out_channels=1,
            ffn_expansion_factor=2.66,
            bias=False,
            heads=8,
            LayerNorm_type='WithBias'
    ):
        super(Signal_Encoder, self).__init__()
        self.dim = dim
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.transformer_ewts = nn.ModuleList([TransformerEWTBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(5)])
        # self.conv = nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1)
        self.width = width
        self.high = high
    def forward(self, image):
        dim = self.dim
        b, c, h, w = image.size()
        # 检查通道是否符合期望，如果不符合则使用 patch embedding
        signal_ewt, signal_mfb = [], []
        # feature = F.interpolate(image, [self.width, self.high], mode='nearest')
        # if c != dim:
        feature = self.patch_embed(image)
        for transformer_ewt in self.transformer_ewts:
            feature_l, ewt, mfb = transformer_ewt(feature)
            signal_ewt.append(ewt)
            signal_mfb.append(mfb)
            feature = feature_l
        # feature = self.conv(feature)
        return feature, signal_ewt, signal_mfb

class Signal_Decoder(nn.Module):
    def __init__(
            self,
            dim=32,
            channel=1,
            inp_channels=1,
            ffn_expansion_factor=2.66,
            bias=False,
            heads=8,
            LayerNorm_type='WithBias'
    ):
        super(Signal_Decoder, self).__init__()

        self.transformer_iewts = nn.ModuleList([TransformerIEWTBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(5)])
        # self.iewts = nn.ModuleList([IEWT() for _ in range(5)])
        self.transformer = TransformerBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type)
        self.transformer1 = TransformerBlock(dim=dim*2, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type)
        self.conv2 = nn.Conv2d(dim, dim*2, kernel_size=1)
        self.conv3 = nn.Conv2d(dim*2, channel, kernel_size=1)
        self.conv4 = nn.Conv2d(channel, channel, kernel_size=1)
        self.softmax = nn.Sigmoid()

    def forward(self, feature, signal_ewt, signal_mfb):
        for i in range(5):
            feature = self.transformer_iewts[i](feature, signal_ewt[i], signal_mfb[i])
        # h, w = image_size[0], image_size[1]
        feature = self.transformer(feature)
        feature = self.conv2(feature)
        feature = self.transformer1(feature)
        feature = self.conv3(feature)
        # feature = F.interpolate(feature, [h, w], mode='nearest')
        # feature = self.conv4(feature)
        feature = self.softmax(feature)
        return feature

class Signal_Fusion_Decoder(nn.Module):
    def __init__(
            self,
            dim=32,
            channel=1,
            inp_channels=1,
            ffn_expansion_factor=2.66,
            bias=False,
            heads=8,
            LayerNorm_type='WithBias'
    ):
        super(Signal_Fusion_Decoder, self).__init__()

        self.transformer_iewts = nn.ModuleList([TransformerIEWTBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(5)])
        self.channel_reduce = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.conv2_1 = nn.Conv1d(2, 1, kernel_size=1)
        self.transformer = TransformerBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type)
        self.transformer1 = TransformerBlock(dim=dim*2, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type)
        self.conv2 = nn.Conv2d(dim, dim*2, kernel_size=1)
        self.conv3 = nn.Conv2d(dim*2, channel, kernel_size=1)
        self.conv4 = nn.Conv2d(channel, channel, kernel_size=1)
        self.softmax = nn.Sigmoid()

    def forward(self, feature_a, signal_ewt_a, signal_mfb_a, feature_b, signal_ewt_b, signal_mfb_b):
        # 将 signal_low_a 和 signal_low_b 在通道维度拼接
        # signal_low_a_b = [torch.cat((a, b), dim=1) for a, b in zip(feature_a, feature_b)]
        # signal_high_a_b = [torch.cat((a, b), dim=1) for a, b in zip(signal_high_a, signal_high_b)]
        # # 通过通道融合层
        # signal_low_a_b = [self.channel_fusion(x) for x in signal_low_a_b]
        # signal_high_a_b = [self.channel_fusion(x) for x in signal_high_a_b]
        feature_fusion = torch.cat((feature_a,feature_b),dim=1)
        feature_fusion = self.channel_reduce(feature_fusion)
        signal_mfb_a = np.array(signal_mfb_a)
        signal_mfb_b = np.array(signal_mfb_b)
        singal_mfb_fusion = (signal_mfb_a+signal_mfb_b)/2
        signal_ewt_fusion = []

# 遍历两个 5*2 的列表，逐行处理
        for i in range(5):
            row = []
            for j in range(len(signal_ewt_a[0])):
                # 初始化一个存储卷积处理后 tensor 的列表
                processed_tensor = torch.empty(3, 1024)

                # 对每个 3*1024 tensor 的每一行进行处理
                for k in range(3):
                    # 取出 signal_ewt_a 和 signal_ewt_b 中的第 k 行 (1*1024)
                    row_a = signal_ewt_a[i][j][k, :].unsqueeze(0)  # 1*1024
                    row_b = signal_ewt_b[i][j][k, :].unsqueeze(0)  # 1*1024

                    # 在第 0 维拼接，得到 2*1024
                    concatenated_tensor = torch.cat((row_a, row_b), dim=0)  # 2*1024
                    concatenated_tensor = concatenated_tensor.float()
                    # 将拼接后的 tensor 输入卷积层 self.conv2_1，得到 1*1024
                    conv_result = self.conv2_1(concatenated_tensor.unsqueeze(0)).squeeze(0)  # 1*1024

                    # 将卷积结果放入 processed_tensor 的第 k 行
                    processed_tensor[k, :] = conv_result

                # 将处理后的 3*1024 tensor 添加到当前 row
                row.append(processed_tensor.cuda())

            # 将 row 添加到 result_list
            signal_ewt_fusion.append(row)
        # signal_ewt_fusion = signal_ewt_fusion.cuda()

        for i in range(5):
            feature = self.transformer_iewts[i](feature_fusion, signal_ewt_fusion[i], singal_mfb_fusion[i])
        # h, w = image_size[0], image_size[1]
        feature = self.transformer(feature)
        feature = self.conv2(feature)
        feature = self.transformer1(feature)
        feature = self.conv3(feature)
        # feature = F.interpolate(feature, [h, w], mode='nearest')
        # feature = self.conv4(feature)
        feature = self.softmax(feature)
        return feature



class Signal_Transformer(nn.Module):
    def __init__(
            self, 
            input_dim=1024,
            output_dim=1024,
            dim=32,
            channel=1,
            num_transformer=5,
            ffn_expansion_factor=2.66,
            bias=False,
            heads=8,
            LayerNorm_type='WithBias'
        ):
        super(Signal_Transformer, self).__init__()
        # 使用循环来创建多个 gMLP 实例
        # self.gmlps = nn.ModuleList([gMLP(input_dim, output_dim) for _ in range(20)])
        # self.conv3 = ConvToFlat()
        # self.conv4 = FlatToConv()
        self.MLP = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.GELU(),
            nn.Linear(input_dim*2, input_dim*4),
            nn.GELU(),
            nn.Linear(input_dim*4, output_dim)
        )
        self.transformer = MultiLayerTransformer(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                               num_transformer=num_transformer,bias=bias, LayerNorm_type=LayerNorm_type)
        self.mlps = nn.ModuleList([self.MLP for _ in range(10)])

    def forward(self, fusion_ewt, signal_ewt):
        fusion_ewt = self.transformer(fusion_ewt)
        # fusion_ewt = fusion_ewt.cpu().detach().numpy()
        # signal_ewt[0] = fusion_ewt.squeeze()
        new_signal_ewt = []
        for i in range(5):
            batch_signal = []
            for j in range(len(signal_ewt[0])):
                # Clone and process tensors to avoid inplace operations
                tensor1 = signal_ewt[i][j][1].clone().float().unsqueeze(0)
                tensor2 = signal_ewt[i][j][2].clone().float().unsqueeze(0)
                tensor1 = self.mlps[i](tensor1)
                tensor2 = self.mlps[i+5](tensor2)
                batch_signal.append(torch.cat([signal_ewt[i][j][0].unsqueeze(0), tensor1, tensor2], dim=0))
            new_signal_ewt.append(batch_signal)

        return fusion_ewt, new_signal_ewt


class lenth_transfer1(nn.Module):
    def __init__(self, inp_channels):
        super(lenth_transfer1, self).__init__()
        # 创建多个 TransformerBlock 实例
        self.Conv1 = nn.Conv2d(in_channels=inp_channels, out_channels=inp_channels, kernel_size=1, padding=0)
        self.Conv2 = nn.Conv2d(in_channels=inp_channels, out_channels=inp_channels, kernel_size=1, padding=0)
        self.Conv3 = nn.Conv2d(in_channels=inp_channels, out_channels=inp_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = F.interpolate(x, [128, 128], mode='nearest')
        x = self.Conv1(x)
        x = F.interpolate(x, [64, 64], mode='nearest')
        x = self.Conv2(x)
        x = F.interpolate(x, [32, 32], mode='nearest')
        x = self.Conv3(x)
        return x

class lenth_transfer2(nn.Module):
    def __init__(self, inp_channels):
        super(lenth_transfer2, self).__init__()
        # 创建多个 TransformerBlock 实例
        self.Conv1 = nn.Conv2d(in_channels=inp_channels, out_channels=inp_channels, kernel_size=1, padding=0)
        self.Conv2 = nn.Conv2d(in_channels=inp_channels, out_channels=inp_channels, kernel_size=1, padding=0)
        self.Conv3 = nn.Conv2d(in_channels=inp_channels, out_channels=inp_channels, kernel_size=1, padding=0)

    def forward(self, x,h,w):
        x = F.interpolate(x, [64, 64], mode='nearest')
        x = self.Conv1(x)
        x = F.interpolate(x, [128, 128], mode='nearest')
        x = self.Conv2(x)
        x = F.interpolate(x, [h,w], mode='nearest')
        x = self.Conv3(x)
        return x
class MultiLayerTransformer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, num_transformer, bias=True, LayerNorm_type='layernorm'):
        super(MultiLayerTransformer, self).__init__()
        # 创建多个 TransformerBlock 实例
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_transformer)
        ])

    def forward(self, x):
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        return x
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x

class ConvToFlat(nn.Module):
    def __init__(self):
        super(ConvToFlat, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=16)  # 调整 kernel_size 为 13
        self.flatten = nn.Flatten()  # 将 (b, 1024, 1, 1) 展平为 (b, 1024)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.conv4(x)
        x = nn.ReLU()(x)
        x = self.conv5(x)  # 输出应该是 (b, 1024, 1, 1)
        x = nn.ReLU()(x)
        x = self.flatten(x)
        return x
class FlatToConv(nn.Module):
    def __init__(self):
        super(FlatToConv, self).__init__()
        # 转置卷积层定义
        self.deconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)    # 输出: (b, 512, 2, 2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)    # 输出: (b, 256, 4, 4)
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)    # 输出: (b, 128, 8, 8)
        self.deconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)     # 输出: (b, 64, 16, 16)
        self.deconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)      # 输出: (b, 32, 32, 32)
        self.deconv6 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=8, stride=8, padding=0)      # 输出: (b, 1, 256, 256)

    def forward(self, x):
        x = self.deconv1(x)
        x = nn.ReLU()(x)
        x = self.deconv2(x)
        x = nn.ReLU()(x)
        x = self.deconv3(x)
        x = nn.ReLU()(x)
        x = self.deconv4(x)
        x = nn.ReLU()(x)
        x = self.deconv5(x)
        x = nn.ReLU()(x)
        x = self.deconv6(x)
        return x
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x
# class Signal_Fusion_Dncoder(nn.Module):
#     def __init__(self):
#         super(Signal_Fusion_Dncoder, self).__init__()
#
#         # 延迟初始化，稍后根据输入特征维度动态初始化
#         self.iewts = None
#         self.channel_fusion = None
#
#     def initialize_layers(self, input_dim, output_dim):
#         # 根据动态计算的输入维度，初始化 IEWT 和通道融合层
#         self.iewts = nn.ModuleList([IEWT() for _ in range(5)])
#         self.channel_fusion = nn.Conv1d(input_dim * 2, output_dim, kernel_size=1)
#         # self.channel_fusion = nn.Linear(input_dim * 2, output_dim)
#     def forward(self, signal_low_a, signal_high_a, signal_low_b, signal_high_b):
#         # 在第一次前向传播时确定特征维度
#         if self.iewts is None or self.channel_fusion is None:
#             input_dim = signal_low_a[0].shape[1]  # 假设输入为 (batch_size, channels, length)
#             output_dim = signal_high_a[0].shape[1]
#             self.initialize_layers(input_dim, output_dim)
#
#         # 将 signal_low_a 和 signal_low_b 在通道维度拼接
#         signal_low_a_b = [torch.cat((a, b), dim=1) for a, b in zip(signal_low_a, signal_low_b)]
#         signal_high_a_b = [torch.cat((a, b), dim=1) for a, b in zip(signal_high_a, signal_high_b)]
#
#         # 通过通道融合层
#         signal_low_a_b = [self.channel_fusion(x) for x in signal_low_a_b]
#         signal_high_a_b = [self.channel_fusion(x) for x in signal_high_a_b]
#
#         # IEWT 处理过程
#         feature = signal_low_a_b[4]
#         for i in range(5):
#             feature = self.iewts[i](feature, signal_high_a_b[4 - i])
#
#         return feature
# class Signal_Fusion_Dncoder(nn.Module):
#     def __init__(self, input_dim=1024, output_dim=1024):
#         super(Signal_Fusion_Dncoder, self).__init__()
#         # 使用循环创建多个 IEWT 实例
#         self.iewts = nn.ModuleList([IEWT() for _ in range(5)])

#         # 定义通道融合层，可以是卷积层或全连接层
#         self.channel_fusion = nn.Conv1d(input_dim * 2, output_dim, kernel_size=1)
#         # 如果需要使用全连接层：
#         # self.channel_fusion = nn.Linear(input_dim * 2, output_dim)

#     def forward(self, signal_low_a, signal_high_a, signal_low_b, signal_high_b):
#         # 将 signal_low_a 和 signal_low_b 在通道维度拼接
#         signal_low_a_b = [torch.cat((a, b), dim=1) for a, b in zip(signal_low_a, signal_low_b)]
#         signal_high_a_b = [torch.cat((a, b), dim=1) for a, b in zip(signal_high_a, signal_high_b)]
#         # 通过通道融合层
#         signal_low_a_b = [self.channel_fusion(x) for x in signal_low_a_b]
#         signal_high_a_b = [self.channel_fusion(x) for x in signal_high_a_b]

#         feature = signal_low_a_b[4]
#         for i in range(5):
#             feature = self.iewts[i](feature, signal_high_a_b[4 - i])

#         return feature

