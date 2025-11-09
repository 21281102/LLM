import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math, random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse

from Multi_head import MultiHeadAttention
from Position_wise_FFN import PositionwiseFFN
from LayerNorm import ManualResidualLayerNorm, ManualLayerNorm
from Positional_encoding import OriginalTransformerPositionalEncoding

# 添加随机种子设置函数
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 确保结果目录存在
os.makedirs('results', exist_ok=True)

# ===================== 数据集定义 =====================
class IWSLTDataset(Dataset):
    def __init__(self, src_path, tgt_path, block_size=64):
        with open(src_path, 'r', encoding='utf-8') as f:
            src_lines = [line.strip() for line in f.readlines() if line.strip()]
        with open(tgt_path, 'r', encoding='utf-8') as f:
            tgt_lines = [line.strip() for line in f.readlines() if line.strip()]

        assert len(src_lines) == len(tgt_lines), "源语言和目标语言文件行数不匹配"
        self.block_size = block_size
        self.data = list(zip(src_lines, tgt_lines))
        
        # 改为字符级处理，与第一段代码一致
        all_text = "".join(src_lines + tgt_lines)
        chars = sorted(set(all_text))
        self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        for i, char in enumerate(chars):
            self.vocab[char] = i + 4  # 从4开始，避开特殊标记
        
        self.vocab_size = len(self.vocab)
        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

    def encode(self, text):
        # 字符级编码
        ids = [self.vocab.get(char, self.unk_idx) for char in text]
        ids = [self.bos_idx] + ids + [self.eos_idx]
        
        # 填充或截断
        if len(ids) < self.block_size:
            ids += [self.pad_idx] * (self.block_size - len(ids))
        else:
            ids = ids[:self.block_size]
        return torch.tensor(ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return self.encode(src), self.encode(tgt)

# ===================== Transformer 模块 =====================
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, ablation=None):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.norm1 = ManualLayerNorm(d_model)
        self.norm2 = ManualLayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ablation = ablation

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, x, x, mask)
        if self.ablation == "residual":
            x = self.norm1(self.dropout(attn_out))
        else:
            x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        if self.ablation == "residual":
            x = self.norm2(self.dropout(ffn_out))
        else:
            x = self.norm2(x + self.dropout(ffn_out))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, ablation=None):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.norm1 = ManualLayerNorm(d_model)
        self.norm2 = ManualLayerNorm(d_model)
        self.norm3 = ManualLayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ablation = ablation

    def forward(self, x, enc_out=None, src_mask=None, tgt_mask=None):
        attn_out = self.self_attn(x, x, x, tgt_mask)
        if self.ablation == "residual":
            x = self.norm1(self.dropout(attn_out))
        else:
            x = self.norm1(x + self.dropout(attn_out))

        if enc_out is not None:
            cross_out = self.cross_attn(x, enc_out, enc_out, src_mask)
            if self.ablation == "residual":
                x = self.norm2(self.dropout(cross_out))
            else:
                x = self.norm2(x + self.dropout(cross_out))

        ffn_out = self.ffn(x)
        if self.ablation == "residual":
            x = self.norm3(self.dropout(ffn_out))
        else:
            x = self.norm3(x + self.dropout(ffn_out))
        return x


class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=64, num_heads=2, d_ff=128,
                 num_layers=1, max_seq_length=64, dropout=0.1, ablation=None):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.ablation = ablation
        
        # 统一位置编码接口，添加max_seq_length参数
        if ablation == "posenc":
            self.pos_encoding = nn.Identity()
        else:
            self.pos_encoding = OriginalTransformerPositionalEncoding(d_model, max_seq_length)  # 添加max_seq_length

        if ablation == "encoder":
            self.encoder = None
        else:
            self.encoder = nn.ModuleList([
                TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, ablation=ablation)
                for _ in range(num_layers)
            ])

        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, ablation=ablation)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        # 添加缩放因子，与第一段代码统一
        src = self.src_embed(src) * math.sqrt(self.src_embed.embedding_dim)
        tgt = self.tgt_embed(tgt) * math.sqrt(self.tgt_embed.embedding_dim)
        
        src = self.pos_encoding(src)
        tgt = self.pos_encoding(tgt)

        if self.encoder is not None:
            for layer in self.encoder:
                src = layer(src)

        enc_out = src if self.encoder is not None else None

        for layer in self.decoder:
            tgt = layer(tgt, enc_out)
        return self.fc_out(tgt)

# ===================== 改进的绘图函数 =====================
def plot_ablation_comparison(ablation_results):
    """绘制消融实验对比图"""
    plt.figure(figsize=(15, 5))
    
    # 损失对比
    plt.subplot(1, 3, 1)
    for name, results in ablation_results.items():
        plt.plot(results['val_losses'], label=name, marker='o')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 困惑度对比
    plt.subplot(1, 3, 2)
    for name, results in ablation_results.items():
        plt.plot(results['val_ppls'], label=name, marker='o')
    plt.title('Validation Perplexity Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.grid(True)
    
    # 最终性能柱状图
    plt.subplot(1, 3, 3)
    final_ppls = [results['val_ppls'][-1] for results in ablation_results.values()]
    names = list(ablation_results.keys())
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    bars = plt.bar(names, final_ppls, color=colors[:len(names)])
    plt.title('Final Performance Comparison')
    plt.ylabel('Final Perplexity')
    plt.xticks(rotation=45)
    
    # 在柱子上添加数值
    for bar, ppl in zip(bars, final_ppls):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{ppl:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/ablation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_ablation_impact(ablation_results):
    """显示各组件对性能的影响程度"""
    if 'none' not in ablation_results:
        print("⚠️ 没有基准模型(none)，无法计算性能影响")
        return
        
    baseline_ppl = ablation_results['none']['val_ppls'][-1]
    
    impacts = {}
    for name, results in ablation_results.items():
        if name != 'none':
            final_ppl = results['val_ppls'][-1]
            degradation = ((final_ppl - baseline_ppl) / baseline_ppl) * 100
            impacts[name] = degradation
    
    if not impacts:
        return
        
    # 绘制影响程度图
    plt.figure(figsize=(10, 6))
    names = list(impacts.keys())
    degradations = list(impacts.values())
    
    colors = ['red' if x > 0 else 'green' for x in degradations]
    bars = plt.bar(names, degradations, color=colors, alpha=0.7)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Performance Impact of Ablated Components')
    plt.ylabel('Performance Degradation (%)')
    plt.xlabel('Ablated Components')
    
    # 添加数值标签
    for bar, deg in zip(bars, degradations):
        plt.text(bar.get_x() + bar.get_width()/2, deg + (1 if deg > 0 else -3), 
                f'{deg:+.1f}%', ha='center', va='bottom' if deg > 0 else 'top',
                fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/ablation_impact.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_ablation_report(ablation_results):
    """生成详细的消融实验报告"""
    print("=" * 60)
    print("            TRANSFORMER 消融实验分析报告")
    print("=" * 60)
    
    if 'none' not in ablation_results:
        print("⚠️ 没有基准模型(none)，无法生成完整报告")
        return
        
    baseline = ablation_results['none']
    baseline_final_ppl = baseline['val_ppls'][-1]
    
    print(f"\n📊 基准模型 (完整Transformer) 最终性能:")
    print(f"  - 验证集困惑度: {baseline_final_ppl:.2f}")
    print(f"  - 最终损失: {baseline['val_losses'][-1]:.4f}")
    
    print("\n🔬 各消融设置性能对比:")
    print("-" * 60)
    print(f"{'消融类型':<12} {'最终困惑度':<12} {'性能下降%':<12} {'收敛速度':<10}")
    print("-" * 60)
    
    impacts = {}
    for name, results in ablation_results.items():
        if name == 'none':
            continue
            
        final_ppl = results['val_ppls'][-1]
        degradation = ((final_ppl - baseline_final_ppl) / baseline_final_ppl) * 100
        impacts[name] = degradation
        
        # 简单评估收敛速度（最后3个epoch的平均改进）
        if len(results['val_ppls']) >= 3:
            last_3_improvement = np.mean(np.diff(results['val_ppls'][-3:]))
            convergence = "慢" if last_3_improvement > -0.1 else "快"
        else:
            convergence = "未知"
        
        print(f"{name:<12} {final_ppl:<12.2f} {degradation:<12.1f} {convergence:<10}")
    
    if impacts:
        print("\n💡 关键发现:")
        worst_component = max(impacts, key=impacts.get)
        best_component = min(impacts, key=impacts.get)
        print(f"1. 最重要的组件: {worst_component} (影响: {impacts[worst_component]:+.1f}%)")
        print(f"2. 对性能影响最小的组件: {best_component} (影响: {impacts[best_component]:+.1f}%)")
    
    print("\n" + "=" * 60)

def plot_individual_training_curves(ablation_results, ablation_type):
    """绘制单个消融实验的训练曲线"""
    if ablation_type not in ablation_results:
        return
        
    results = ablation_results[ablation_type]
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['train_losses'], label="Train Loss", marker='o')
    plt.plot(results['val_losses'], label="Val Loss", marker='o')
    plt.legend()
    plt.title(f"{ablation_type} - Loss Curve")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(results['train_ppls'], label="Train PPL", marker='o')
    plt.plot(results['val_ppls'], label="Val PPL", marker='o')
    plt.legend()
    plt.title(f"{ablation_type} - Perplexity Curve")
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"results/training_curves_{ablation_type}.png", dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，避免显示

# ===================== 训练函数 =====================
def train_model(ablation_type, train_loader, val_loader, vocab_size, device, epochs=5):
    """训练单个模型并返回结果"""
    print(f"\n🎯 开始训练消融实验: {ablation_type}")
    
    model = TransformerSeq2Seq(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        ablation=ablation_type
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_losses, val_losses, train_ppls, val_ppls = [], [], [], []
    
    for epoch in range(1, epochs + 1):
        print(f"\n===== Epoch {epoch} =====")
        model.train()
        total_loss = 0
        
        for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch}"):
            src, tgt = src.to(device), tgt.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            logits = model(src, tgt_in)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt_out.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        ppl = math.exp(avg_loss)
        train_losses.append(avg_loss)
        train_ppls.append(ppl)
        print(f"Train Loss: {avg_loss:.4f}, PPL: {ppl:.2f}")
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
                logits = model(src, tgt_in)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt_out.reshape(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_ppl = math.exp(avg_val_loss)
        val_losses.append(avg_val_loss)
        val_ppls.append(val_ppl)
        print(f"Val Loss: {avg_val_loss:.4f}, PPL: {val_ppl:.2f}")
    
    # 保存模型
    torch.save(model.state_dict(), f"results/transformer_ablation_{ablation_type}.pth")
    print(f"✅ 模型已保存：results/transformer_ablation_{ablation_type}.pth")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_ppls': train_ppls,
        'val_ppls': val_ppls
    }

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer消融实验')
    parser.add_argument("--ablation", type=str, default="all",
                        choices=["none", "posenc", "residual", "encoder", "all"],
                        help="选择消融类型或'all'运行所有")
    parser.add_argument("--epochs", type=int, default=5, help="训练周期数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    print(f"🔧 设置随机种子: {args.seed}")
    
    # 确定要运行的消融实验
    if args.ablation == "all":
        ablation_types = ["none", "posenc", "residual", "encoder"]
    else:
        ablation_types = [args.ablation]
    
    print(f"🔬 运行的消融实验: {ablation_types}")
    print(f"⏰ 每个实验训练周期: {args.epochs}")
    
    # 数据准备
    data_dir = r'C:\D\大模型\IWSLT2017'
    block_size = 64
    
    train_de, train_en, val_de, val_en = None, None, None, None
    for f in os.listdir(data_dir):
        if 'train' in f and f.endswith('.de') and '.xml' not in f:
            train_de = os.path.join(data_dir, f)
        elif 'train' in f and f.endswith('.en') and '.xml' not in f:
            train_en = os.path.join(data_dir, f)
        elif 'dev2010' in f and f.endswith('.de') and '.xml' not in f:
            val_de = os.path.join(data_dir, f)
        elif 'dev2010' in f and f.endswith('.en') and '.xml' not in f:
            val_en = os.path.join(data_dir, f)
    
    if not train_de or not train_en:
        raise FileNotFoundError("找不到 IWSLT2017 训练集文件")
    
    # 创建完整训练数据集
    full_train_dataset = IWSLTDataset(train_de, train_en, block_size)
    vocab_size = full_train_dataset.vocab_size
    
    if val_de and val_en:
        val_dataset = IWSLTDataset(val_de, val_en, block_size)
        train_dataset = full_train_dataset
    else:
        train_size = int(0.9 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset, [train_size, val_size]
        )
        print("从训练集划分验证集")
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 运行所有消融实验
    ablation_results = {}
    
    for ablation_type in ablation_types:
        results = train_model(ablation_type, train_loader, val_loader, vocab_size, device, args.epochs)
        ablation_results[ablation_type] = results
        
        # 为每个实验绘制单独的曲线
        plot_individual_training_curves(ablation_results, ablation_type)
    
    # 如果有多个实验，绘制对比图
    if len(ablation_results) > 1:
        print("\n📈 生成消融实验对比图...")
        plot_ablation_comparison(ablation_results)
        plot_ablation_impact(ablation_results)
        generate_ablation_report(ablation_results)
        
        # 保存结果以便后续分析
        with open('results/ablation_results.json', 'w') as f:
            # 转换为可JSON序列化的格式
            serializable_results = {}
            for k, v in ablation_results.items():
                serializable_results[k] = {k2: [float(x) for x in v2] for k2, v2 in v.items()}
            json.dump(serializable_results, f, indent=2)
        print("✅ 结果已保存到: results/ablation_results.json")
    
    print("\n🎉 所有实验完成！")