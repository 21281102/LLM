import math
import torch
import torch.nn as nn
class PositionwiseFFN(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else dim * 4
        
        # 对应公式: W₂ φ(W₁ x + b₁) + b₂
        self.w1 = nn.Linear(dim, self.hidden_dim)  # W₁
        self.w2 = nn.Linear(self.hidden_dim, dim)  # W₂
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 对应: φ(W₁ x + b₁)
        hidden = self.activation(self.w1(x))
        hidden = self.dropout(hidden)
        
        # 对应: W₂ φ(W₁ x + b₁) + b₂
        output = self.w2(hidden)
        output = self.dropout(output)
        
        return output
    

# 测试代码
def test_positionwise_ffn():
    # 创建测试数据：batch_size=2, seq_len=4, dim=8
    x = torch.randn(2, 4, 8)
    print(f"输入形状: {x.shape}")
    
    # 测试第二个版本（简洁版）
    ffn2 = PositionwiseFFN(dim=8, hidden_dim=32)
    output2 = ffn2(x)
    print(f"FFN 输出形状: {output2.shape}")
    
    # 验证输出范围

    print(f"FFN 输出范围: [{output2.min():.4f}, {output2.max():.4f}]")

# 运行测试
test_positionwise_ffn()

