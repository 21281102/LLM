# Transformer 消融实验项目

本项目实现了完整的 Transformer 模型，并在 IWSLT2017 英语-德语翻译数据集上进行消融实验，分析各个组件对模型性能的影响。

## 📋 项目概述

本项目从零开始实现了 Transformer 架构的核心组件，包括：
- 多头注意力机制 (Multi-Head Attention)
- 位置编码 (Positional Encoding)
- 位置前馈网络 (Position-wise FFN)
- 层归一化 (Layer Normalization)
- 残差连接 (Residual Connections)

通过消融实验，系统性地研究了各个组件对模型性能的贡献，包括：
- **完整 Transformer** (基准模型)
- **无位置编码** (移除位置编码)
- **无残差连接** (移除残差连接)
- **无编码器** (仅使用解码器)

## 🗂️ 项目结构

```
.
├── IWSLT2017/                    # 数据集目录
│   ├── train.tags.en-de.en      # 训练集英语文件
│   ├── train.tags.en-de.de      # 训练集德语文件
│   ├── IWSLT17.TED.dev2010.en-de.en.xml  # 验证集英语文件
│   └── IWSLT17.TED.dev2010.en-de.de.xml  # 验证集德语文件
├── src/                          # 源代码目录
│   ├── main.py                  # 主程序入口
│   ├── attention.py             # 注意力机制实现（示例代码）
│   ├── Multi_head.py            # 多头注意力机制
│   ├── Positional_encoding.py   # 位置编码实现
│   ├── Position_wise_FFN.py     # 位置前馈网络
│   └── LayerNorm.py             # 层归一化实现
├── scripts/                      # 脚本目录
│   └── run.sh                   # 批量运行脚本
├── results/                      # 实验结果目录
│   ├── ablation_results.json    # 消融实验结果（JSON格式）
│   ├── ablation_comparison.png  # 消融实验对比图
│   ├── ablation_impact.png      # 组件影响程度图
│   ├── training_curves_*.png    # 各实验训练曲线
│   └── transformer_ablation_*.pth  # 训练好的模型权重
└── README.md                     # 项目说明文档
```

## 🚀 快速开始

### 环境要求

- Python 3.10.19
- Torch 2.4.1
- NumPy
- Matplotlib
- tqdm

### 安装依赖

```bash
pip install torch torchvision numpy matplotlib tqdm
```

### 数据准备

确保 IWSLT2017 数据集已放置在 `IWSLT2017/` 目录下。数据集应包含：
- 训练集：`train.tags.en-de.en` 和 `train.tags.en-de.de`
- 验证集：`IWSLT17.TED.dev2010.en-de.en.xml` 和 `IWSLT17.TED.dev2010.en-de.de.xml`

### 运行实验

#### 运行单个消融实验

```bash
# 训练完整Transformer（基准模型）
python src/main.py --ablation none --epochs 5 --seed 42

# 训练无位置编码模型
python src/main.py --ablation posenc --epochs 5 --seed 42

# 训练无残差连接模型
python src/main.py --ablation residual --epochs 5 --seed 42

# 训练无编码器模型
python src/main.py --ablation encoder --epochs 5 --seed 42
```

#### 批量运行所有实验

```bash
# 运行所有消融实验
python src/main.py --ablation all --epochs 5 --seed 42

# 或使用shell脚本（Linux/Mac）
bash scripts/run.sh
```

## 📊 实验配置

### 模型参数

- **模型维度 (d_model)**: 64
- **注意力头数 (num_heads)**: 2
- **前馈网络维度 (d_ff)**: 128
- **编码器/解码器层数 (num_layers)**: 1
- **最大序列长度**: 64
- **Dropout率**: 0.1
- **批次大小**: 8
- **学习率**: 1e-4
- **优化器**: AdamW
- **梯度裁剪**: 0.5

### 消融实验设置

| 消融类型 | 描述 | 移除的组件 |
|---------|------|-----------|
| `none` | 完整Transformer（基准） | 无 |
| `posenc` | 无位置编码 | 位置编码 |
| `residual` | 无残差连接 | 所有残差连接 |
| `encoder` | 无编码器 | 编码器层 |

## 📈 实验结果

实验结果显示在 `results/` 目录中：

### 性能对比

根据实验结果（`results/ablation_results.json`），各模型的最终验证集困惑度：

- **完整Transformer (none)**: 5.30
- **无位置编码 (posenc)**: 6.48
- **无残差连接 (residual)**: 6.92
- **无编码器 (encoder)**: 5.05

### 关键发现

1. **残差连接最重要**：移除残差连接导致性能下降最显著（约30%），说明残差连接对模型训练至关重要。
2. **位置编码的重要性**：移除位置编码导致性能下降约22%，说明位置信息对序列建模很重要。
3. **编码器的作用**：在本次实验中，无编码器模型反而表现略好，这可能与数据集规模和模型容量有关。

### 可视化结果

项目自动生成以下可视化图表：

- `ablation_comparison.png`: 所有模型的损失和困惑度对比
- `ablation_impact.png`: 各组件对性能的影响程度
- `training_curves_*.png`: 各个实验的详细训练曲线

## 🔧 核心组件实现

### 1. 多头注意力机制 (`Multi_head.py`)

实现了标准的多头自注意力和交叉注意力机制，支持：
- 可配置的注意力头数
- 注意力掩码支持
- Dropout正则化

### 2. 位置编码 (`Positional_encoding.py`)

实现了 Transformer 原始论文中的正弦位置编码：
- 使用 sin/cos 函数生成位置编码
- 支持可配置的最大序列长度
- 通过加法与词嵌入融合

### 3. 位置前馈网络 (`Position_wise_FFN.py`)

实现了两层全连接前馈网络：
- ReLU 激活函数
- 可配置的隐藏层维度（默认4倍模型维度）
- Dropout 正则化

### 4. 层归一化 (`LayerNorm.py`)

手动实现了 LayerNorm：
- 可学习的缩放（gamma）和偏移（beta）参数
- 在最后一个维度进行归一化
- 支持残差连接的 LayerNorm

## 🎯 使用方法

### 命令行参数

```bash
python src/main.py [OPTIONS]

选项:
  --ablation {none,posenc,residual,encoder,all}
                        选择消融类型或'all'运行所有实验
  --epochs EPOCHS       训练周期数（默认: 5）
  --seed SEED           随机种子（默认: 42）
```

### 代码示例

```python
from src.main import TransformerSeq2Seq, IWSLTDataset
from torch.utils.data import DataLoader

# 创建数据集
dataset = IWSLTDataset('IWSLT2017/train.tags.en-de.en', 
                       'IWSLT2017/train.tags.en-de.de')

# 创建模型（完整Transformer）
model = TransformerSeq2Seq(
    src_vocab_size=dataset.vocab_size,
    tgt_vocab_size=dataset.vocab_size,
    d_model=64,
    num_heads=2,
    d_ff=128,
    num_layers=1,
    ablation=None  # 完整模型
)

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 训练模型...
```

## 📝 实验结果分析

实验结果保存在 `results/ablation_results.json` 中，包含：
- 训练损失和验证损失
- 训练困惑度和验证困惑度
- 每个epoch的详细指标

可以使用以下代码加载和分析结果：

```python
import json

with open('results/ablation_results.json', 'r') as f:
    results = json.load(f)

# 分析各个实验的性能
for ablation_type, metrics in results.items():
    print(f"{ablation_type}: 最终验证困惑度 = {metrics['val_ppls'][-1]:.2f}")
```

## 🔍 技术细节

### 数据处理

- **字符级编码**：使用字符级词汇表，适合多语言场景
- **特殊标记**：`<pad>`, `<bos>`, `<eos>`, `<unk>`
- **序列长度**：固定为64，超出部分截断，不足部分填充

### 模型架构

- **编码器-解码器结构**：标准的Transformer架构
- **注意力机制**：缩放点积注意力
- **正则化**：Dropout和梯度裁剪
- **优化**：AdamW优化器，学习率1e-4

### 训练策略

- **Teacher Forcing**：训练时使用目标序列作为解码器输入
- **梯度裁剪**：防止梯度爆炸
- **随机种子**：确保实验可重现

## 📚 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原始论文
- [IWSLT 2017](https://wit3.fbk.eu/) - 数据集官网
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) - Transformer实现教程

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目仅供学习和研究使用。

## 👥 作者

本项目是大模型课程作业的一部分。

## 🙏 致谢

感谢 IWSLT 2017 数据集提供者和 PyTorch 社区的支持。

