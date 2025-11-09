import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ManualLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        """
        手动实现LayerNorm
        
        Args:
            normalized_shape: 要归一化的维度大小
            eps: 数值稳定性参数
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # 可学习的参数：gamma（缩放）和 beta（偏移）
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x):
        """
        LayerNorm公式: 
        y = gamma * (x - mean) / sqrt(var + eps) + beta
        """
        # 计算均值和方差（在最后一个维度上）
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        
        # 归一化
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # 缩放和偏移
        return self.gamma * x_normalized + self.beta

    def __repr__(self):
        return f"ManualLayerNorm(normalized_shape={self.normalized_shape}, eps={self.eps})"
class ManualResidualLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        """
        手动实现残差连接 + LayerNorm
        
        Args:
            dim: 特征维度
            eps: LayerNorm的数值稳定性参数
        """
        super().__init__()
        self.dim = dim
        self.layer_norm = ManualLayerNorm(dim, eps=eps)
    
    def forward(self, x, sublayer_output):
        """
        残差连接 + LayerNorm公式:
        output = LayerNorm(x + sublayer_output)
        
        Args:
            x: 原始输入
            sublayer_output: 子层（如FFN或Self-Attention）的输出
        """
        # 残差连接
        residual_output = x + sublayer_output
        
        # LayerNorm
        output = self.layer_norm(residual_output)
        
        return output