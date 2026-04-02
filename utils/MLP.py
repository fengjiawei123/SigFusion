from collections import OrderedDict
import torch.nn as nn
import torch


## gMLP
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x):
        return self.fn(x) + x

class SpatialGatingUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.proj = nn.Conv1d(dim, dim, 1)

        nn.init.zeros_(self.proj.weight)
        nn.init.ones_(self.proj.bias)
    
    def forward(self, x):
        res, gate = torch.chunk(x, 2, dim=-1)  # bs, d_ff
        gate = self.ln(gate)                   # Normalize
        gate = self.proj(gate.unsqueeze(-1)).squeeze(-1)  # Spatial Projection

        return res * gate

class gMLP(nn.Module):
    def __init__(self, input_dim=1024, output_dim=256, dim=512, d_ff=1024, num_layers=6):
        super().__init__()
        self.num_layers = num_layers

        self.fc_in = nn.Linear(input_dim, dim)

        self.gmlp = nn.ModuleList([Residual(nn.Sequential(OrderedDict([
            ('ln1_%d' % i, nn.LayerNorm(dim)),
            ('fc1_%d' % i, nn.Linear(dim, d_ff * 2)),
            ('gelu_%d' % i, nn.GELU()),
            ('sgu_%d' % i, SpatialGatingUnit(d_ff)),
            ('fc2_%d' % i, nn.Linear(d_ff, dim)),
        ]))) for i in range(num_layers)])

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, output_dim)
        )

    def forward(self, x):
        x = self.fc_in(x)

        # gMLP
        y = nn.Sequential(*self.gmlp)(x)

        # To logits
        logits = self.to_logits(y)

        return logits

### Mixermlp

class MlpBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, input_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

class MixerBlock(nn.Module):
    def __init__(self, channels_mlp_dim=1024, tokens_hidden_dim=32, channels_hidden_dim=512):
        super().__init__()
        self.ln1 = nn.LayerNorm(channels_mlp_dim)
        self.tokens_mlp_block = MlpBlock(channels_mlp_dim, mlp_dim=tokens_hidden_dim)
        self.ln2 = nn.LayerNorm(channels_mlp_dim)
        self.channels_mlp_block = MlpBlock(channels_mlp_dim, mlp_dim=channels_hidden_dim)

    def forward(self, x):
        y = self.ln1(x)
        y = self.tokens_mlp_block(y)
        out = x + y
        y = self.ln2(out)
        y = out + self.channels_mlp_block(y)
        return y

class MlpMixer(nn.Module):
    def __init__(self, input_dim=1024, output_dim=256, num_blocks=6, tokens_hidden_dim=32, channels_hidden_dim=512):
        super().__init__()
        self.num_blocks = num_blocks
        self.channels_mlp_dim = input_dim
        
        self.mlp_blocks = nn.ModuleList([
            MixerBlock(self.channels_mlp_dim, tokens_hidden_dim, channels_hidden_dim)
            for _ in range(num_blocks)
        ])
        
        self.ln = nn.LayerNorm(self.channels_mlp_dim)
        self.fc = nn.Linear(self.channels_mlp_dim, output_dim)

    def forward(self, x):
        x = x.view(1, -1)  # Ensure the input is treated as a single batch
        for i in range(self.num_blocks):
            x = self.mlp_blocks[i](x)
        x = self.ln(x)
        x = torch.mean(x, dim=0, keepdim=True)  # Reducing dimension to 1xN before final FC
        out = self.fc(x)
        return out
    
    
##resMLP

class Affine(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, 1, channel))
        self.b = nn.Parameter(torch.zeros(1, 1, channel))

    def forward(self, x):
        # Apply an affine transformation element-wise
        return x * self.g + self.b

class PreAffinePostLayerScale(nn.Module):
    def __init__(self, dim, depth, fn):
        super().__init__()
        init_eps = 0.1 if depth <= 18 else 1e-5 if depth <= 24 else 1e-6
        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.affine = Affine(dim)
        self.fn = fn

    def forward(self, x):
        # Apply affine transformation, then the function, and scale the result
        return self.fn(self.affine(x)) * self.scale + x

class ResMLP(nn.Module):
    def __init__(self, input_dim=1024, output_dim=256, expansion_factor=4, depth=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # A simple MLP model with several layers
        self.layers = nn.Sequential()
        current_dim = input_dim
        for i in range(depth):
            next_dim = current_dim * expansion_factor
            self.layers.add_module(
                f"block_{i}",
                PreAffinePostLayerScale(current_dim, i + 1, nn.Sequential(
                    nn.Linear(current_dim, next_dim),
                    nn.GELU(),
                    nn.Linear(next_dim, current_dim)
                ))
            )

        # Affine layer before the final classification layer
        self.final_affine = Affine(current_dim)
        self.classifier = nn.Linear(current_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Forward pass through the layers
        x = self.layers(x)
        x = self.final_affine(x)
        x = torch.mean(x, dim=1)  # Reduce mean across patches if it were higher-dimensional
        out = self.softmax(self.classifier(x))
        return out
####Vipmlp

class vip_MLP(nn.Module):
    """一个简单的多层感知机模型，用于处理一维数据"""
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act2 = nn.GELU()

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return x

class Visionpermutator(nn.Module):
    """ 适用于一维数据的 Vision Permutator """
    def __init__(self, input_dim=1024, output_dim=256, hidden_dim=512, depth=4):
        super().__init__()
        self.layers = nn.Sequential()
        current_dim = input_dim
        
        # 构建网络层
        for i in range(depth - 1):
            self.layers.add_module(f'mlp_{i}', vip_MLP(current_dim, hidden_dim, hidden_dim))
            current_dim = hidden_dim
        
        # 最后一层调整到输出尺寸
        self.layers.add_module('final_layer', vip_MLP(current_dim, hidden_dim, output_dim))

    def forward(self, x):
        return self.layers(x)