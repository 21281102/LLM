#!/bin/bash

# Transformer消融实验运行脚本
# 硬件要求: GPU (至少8GB显存) 或 CPU (需要更多时间)

set -e  # 出错时退出

echo "开始Transformer消融研究实验..."

# 设置随机种子以保证可重现性
SEED=42

# 创建结果目录
mkdir -p results

# 安装依赖
echo "安装依赖..."
pip install -r requirements.txt

# 运行所有消融实验
echo "运行消融实验 (随机种子: $SEED)..."

# 1. 完整Transformer (基准)
echo "=== 训练完整Transformer模型 ==="
python src/main.py --ablation none --epochs 5 --seed $SEED

# 2. 无位置编码
echo "=== 训练无位置编码模型 ==="
python src/main.py --ablation posenc --epochs 5 --seed $SEED

# 3. 无残差连接
echo "=== 训练无残差连接模型 ==="
python src/main.py --ablation residual --epochs 5 --seed $SEED

# 4. 无编码器
echo "=== 训练无编码器模型 ==="
python src/main.py --ablation encoder --epochs 5 --seed $SEED

# 5. 运行所有实验 (批量运行)
echo "=== 批量运行所有消融实验 ==="
python src/main.py --ablation all --epochs 5 --seed $SEED

echo "所有实验完成！结果保存在 results/ 目录"