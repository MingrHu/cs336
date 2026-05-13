import argparse
import torch
import numpy as np
import wandb
from cs336_basics import transformer,bpe,tokenizer
from utils import output_dir

# 参考Task的说明 要求如下
# 可配置的模型 / 优化器超参数（命令行参数）
# 用 np.memmap 高效加载大型数据集（不一次性全读进内存）
# 支持将检查点保存到指定路径
# 定期记录训练 / 验证指标（控制台或 WandB）
def parse_args():
    parser = argparse.ArgumentParser(description="Start to train a MR-Transformer-LM")
    # 一些超参数的输入
    # 模型参数
    parser.add_argument("--d_model", type = int, default=64)
    parser.add_argument("--num_heads", type = int, default=2)
    parser.add_argument("--num_layers", type = int, default=2)
    parser.add_argument("--d_ff", type = int, default=128)
    parser.add_argument("--context_length", type = int, default = 16)
    parser.add_argument("--vocab_size", type = int, default = 10000)
    parser.add_argument("--rope_theta",type = float,default = 10000.0)

    # 优化器参数
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=500)

    # 数据与设备
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    # WandB
    parser.add_argument("--wandb_project", type=str, default="cs336-training")
    return parser.parse_args()