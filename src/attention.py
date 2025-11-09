# 导入相关需要的包
import math
import torch
import torch.nn as nn

import warnings

warnings.filterwarnings(action="ignore")

class SelfAttV4(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

        self.att_drop = nn.Dropout(0.1)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, X, attention_mask=None):

        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.value_proj(X)

        att_weight = Q @ K.transpose(-1, -2) / math.sqrt(self.dim)
        if attention_mask is not None:
            # 给 weight 填充一个极小的值
            att_weight = att_weight.masked_fill(attention_mask == 0, float("-1e20"))

        att_weight = torch.softmax(att_weight, dim=-1)
        print(att_weight)

        att_weight = self.att_drop(att_weight)

        output = att_weight @ V
        ret = self.output_proj(output)
        return ret


X = torch.rand(3, 4, 2)
b = torch.tensor(
    [
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
    ]
)
print(b.shape)

mask = b.unsqueeze(dim=1).repeat(1, 4, 1)

net = SelfAttV4(2)
net(X, mask).shape

