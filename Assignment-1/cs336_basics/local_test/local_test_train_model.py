import argparse
import torch
import numpy as np
import wandb
import os
import torch.nn as nn
from cs336_basics import transformer,bpe,tokenizer,helper
from utils import output_dir,current_dir

# 参考Task的说明 要求如下
# 可配置的模型 / 优化器超参数（命令行参数）
# 用 np.memmap 高效加载大型数据集（不一次性全读进内存）
# 支持将检查点保存到指定路径
# 定期记录训练 / 验证指标（控制台或 WandB）
def parse_args():
    parser = argparse.ArgumentParser(description = "Start to train a MR-Transformer-LM")

    # MR-Transformer-LM模型参数
    parser.add_argument("--d_model", type = int, default = 512)
    parser.add_argument("--num_heads", type = int, default = 16)
    parser.add_argument("--num_layers", type = int, default = 4)
    parser.add_argument("--d_ff", type = int, default = 1344)
    parser.add_argument("--context_length", type = int, default = 256)
    parser.add_argument("--vocab_size", type = int, default = 10000)
    parser.add_argument("--rope_theta",type = float,default = 10000.0)

    # AdamW优化器参数 先不对超参数进行分组
    parser.add_argument("--lr", type = float, default = 3e-4)
    parser.add_argument("--beta1", type = float, default = 0.9)
    parser.add_argument("--beta2", type = float, default = 0.999)
    parser.add_argument("--weight_decay", type = float, default = 0.01)
    parser.add_argument("--eps", type = float, default = 1e-8)

    # 训练时超参数
    parser.add_argument("--batch_size", type = int, default = 256)
    parser.add_argument("--num_steps", type = int, default = 5000)

    # 学习率/梯度相关调度参数
    parser.add_argument("--max_grad_norm", type = float, default = 1.0)
    parser.add_argument("--max_lr", type = float, default = 3e-4)
    parser.add_argument("--min_lr", type = float, default = 3e-5)
    parser.add_argument("--warmup_steps", type = int, default = 2000)
    parser.add_argument("--max_steps", type = int, default = 100000)

    # 日志/检查点/评估参数
    parser.add_argument("--log_interval", type = int, default = 10)
    parser.add_argument("--save_interval", type = int, default = 500)
    parser.add_argument("--eval_interval", type = int, default = 500)

    # 其他参数
    parser.add_argument("--train_data_path", type = str, default = f"{output_dir}/MR_train.npy")
    parser.add_argument("--val_data_path", type = str, default = f"{output_dir}/MR_val.npy")
    parser.add_argument("--device", type = str, default = "cpu")
    parser.add_argument("--checkpoint_path", type = str, default = f"{output_dir}/checkpoint.pt")

    # W&B
    parser.add_argument("--wandb_name", type = str, default = "second_test_cpu")
    parser.add_argument("--wandb_project", type = str, default="cs336-training")
    return parser.parse_args()



def load_memmap_dataset(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r")

def init_model_and_optimizer(args: argparse.Namespace) -> tuple[nn.Module, torch.optim.Optimizer]:
    # 初始化模型
    model = transformer.MR_transformer_lm(
        vocab_size = args.vocab_size,
        context_length = args.context_length,
        d_model = args.d_model,
        num_layers = args.num_layers,
        num_heads = args.num_heads,
        d_ff = args.d_ff,
        rope_theta = args.rope_theta,
        device = args.device,
    )
    model = model.to(args.device)
    
    # 初始化优化器
    optimizer = helper.MR_adamw_opt(
        model.parameters(),
        lr = args.lr,
        betas = (args.beta1,args.beta2),
        weight_decay = args.weight_decay,
        eps = args.eps,
    )
    return model, optimizer


def main():
    args = parse_args()
    
    # 初始化 WandB
    wandb.init(project = args.wandb_project,name = args.wandb_name, config = vars(args))

    # 加载数据集
    train_data = load_memmap_dataset(args.train_data_path)
    val_data = load_memmap_dataset(args.val_data_path)
    
    # 初始化模型和优化器
    model, optimizer = init_model_and_optimizer(args)
    
    # 加载检查点
    start_step = 0
    if os.path.exists(args.checkpoint_path):
        start_step = helper.load_checkpoint(args.checkpoint_path, model, optimizer) + 1
    
    # 训练循环
    model.train()
    for step in range(start_step, args.num_steps):
        # 训练 step
        x, y = helper.data_loading(train_data, args.batch_size, args.context_length, args.device)
        # 清空梯度
        optimizer.zero_grad()
        logits = model(x)
        loss = helper.cross_entropy_loss(
            logits.reshape(-1, args.vocab_size),
            y.reshape(-1))
        loss.backward()
        optimizer.step()
        print(f"Step {step}, Train Loss: {loss.item():.4f}")
        
        # 记录训练 loss
        if step % args.log_interval == 0:
            wandb.log({"train_loss": loss.item()}, step = step)
        
        # 验证
        if step % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = helper.data_loading(val_data, args.batch_size, args.context_length, args.device)
                logits_val = model(x_val)
                val_loss = helper.cross_entropy_loss(
                    logits_val.reshape(-1, args.vocab_size), 
                    y_val.reshape(-1))
            
            print(f"Step {step}, Val Loss: {val_loss.item():.4f}")
            wandb.log({"val_loss": val_loss.item()}, step = step)
            model.train()
        if step % args.save_interval == 0:
            # 保存检查点
            helper.save_checkpoint(model, optimizer, step, args.checkpoint_path)
    
    wandb.finish()

def data_generator():
    input_train_path = f"{current_dir}/data/TinyStoriesV2-GPT4-train.txt"
    input_val_path = f"{current_dir}/data/TinyStoriesV2-GPT4-valid.txt"
    vocab,merges = bpe.MR_BPE.deserialize(f"{current_dir}/output/tiny_stories_vocab.json", 
                               f"{current_dir}/output/tiny_stories_merges.json")
    tk = tokenizer.MR_Tokenizer(vocab,merges,["<|endoftext|>"])
    
    train_id_list = tk._multiple_encode(input_train_path)
    val_id_list = tk._multiple_encode(input_val_path)

    train_ids_np = np.array(train_id_list, dtype = np.int64)
    val_ids_np = np.array(val_id_list, dtype = np.int64)

    train_save_path = f"{output_dir}/MR_train.npy"
    val_save_path = f"{output_dir}/MR_val.npy"
    np.save(train_save_path, train_ids_np)
    np.save(val_save_path, val_ids_np)

if __name__ == "__main__":
    # # 生成tokenIds
    # tokenizer = tokenizer.MR_tokenizer()
    # data_generator()
    main()