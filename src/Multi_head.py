import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim必须能被num_heads整除"
        
        # 多头投影层
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        
        self.att_drop = nn.Dropout(0.1)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, query, key=None, value=None, attention_mask=None):
        # 如果只传入一个参数，则是自注意力
        if key is None and value is None:
            key = value = query
        
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_k, _ = key.shape
        
        # 计算Q, K, V
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)
        
        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力权重
        att_weight = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # 应用注意力掩码
        if attention_mask is not None:
            # 扩展掩码以匹配多头形状
            if len(attention_mask.shape) == 2:  # [seq_len, seq_len]
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            elif len(attention_mask.shape) == 3:  # [batch_size, seq_len, seq_len]
                attention_mask = attention_mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            
            attention_mask = attention_mask.repeat(1, self.num_heads, 1, 1)  # [batch_size, num_heads, seq_len, seq_len]
            att_weight = att_weight.masked_fill(attention_mask == 0, float("-1e20"))
        
        # Softmax和Dropout
        att_weight = torch.softmax(att_weight, dim=-1)
        att_weight = self.att_drop(att_weight)
        
        # 应用注意力权重到V
        output = torch.matmul(att_weight, V)
        
        # 转置并重塑回原始形状
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.dim)
        
        # 输出投影
        ret = self.output_proj(output)
        return ret