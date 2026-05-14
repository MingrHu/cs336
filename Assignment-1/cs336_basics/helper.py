import torch,math,os,typing
import numpy as np
import torch.nn as nn 
from collections.abc import Callable, Iterable
from typing import Optional

# @Author: MingrHu
# @Date: 2026-05-06
# @Description: 构建softmax层
# @Param x: 输入张量
# @Param dim: 指定维度
# @Return: softmax张量
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_val = torch.max(x,dim = dim,keepdim = True).values
    return torch.exp(x - max_val) / torch.exp(x - max_val).sum(dim = dim,keepdim = True)


# @Author: MingrHu
# @Date: 2026-05-06
# @Description: 构建缩放点积注意力层
# @Param q: 查询张量
# @Param k: 键张量
# @Param v: 值张量
# @Param mask: 掩码张量
# @Return: 注意力张量
def scaled_dot_product_attention(q: torch.Tensor,k: torch.Tensor,v: torch.Tensor,mask: torch.Tensor | None = None):
    # (batch_size, ..., seq_len, d_q)
    # (batch_size, ..., seq_len, d_k)
    # (batch_size, ..., seq_len, d_v)
    # (batch_size, ..., seq_len, seq_len)
    qk_dot = q @ k.transpose(-2,-1)
    d_k = k.shape[-1] 
    score = qk_dot / torch.sqrt(torch.tensor(d_k))
    # 自动广播 不要用inf 用-1e9
    if mask is not None:
        score = score + ((~mask) * -1e9)
    attention = softmax(score,dim = -1)
    return attention @ v


# @Author: MingrHu
# @Date: 2026-05-06
# @Description: 构建交叉熵损失层
# @Param x: 输入张量
# @Param target: 目标位置张量
# @Return: 交叉熵损失
def cross_entropy_loss(x: torch.Tensor,target: torch.Tensor):
    # (b × s,vocab_size) (b)
    # 分子直接简化
    tar_x = x[torch.arange(len(target)),target] 

    # 计算对数和 分母
    max_val = torch.max(x,dim = -1,keepdim = True).values
    exp_x = torch.exp(x - max_val)
    exp_sum_x = exp_x.sum(dim = -1,keepdim = True)
    return (torch.log(exp_sum_x) - tar_x + max_val).mean()


# @Author: MingrHu
# @Date: 2026-05-14
# @Description: 构建AdamW优化器
# @Param params: 参数列表
# @Param lr: 学习率
# @Param betas: Adam优化器参数
# @Param eps: Adam优化器参数
# @Param weight_decay: 权重衰减
# @Return: AdamW优化器
class MR_adamw_opt(torch.optim.Optimizer):
    # 示例 
    # opt = torch.optim.SGD([
    #     {"params": model.backbone.parameters(), "lr": 0.001},
    #     {"params": model.head.parameters(), "lr": 0.01}
    # ])
    def __init__(self,params,lr = 1e-3, betas = (0.9, 0.5), eps = 1e-8, weight_decay = 0.0):
        defaults = {
            "lr":lr,
            "betas":betas,
            "eps":eps,
            "weight_decay":weight_decay
        }
        super().__init__(params,defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]: # type: ignore
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = betas
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                # 获取当前步
                st = state.get("st", 0) + 1
                m = state.get("m", torch.zeros_like(param.grad))
                v = state.get("v", torch.zeros_like(param.grad))
                grad = param.grad.data

                m1 = beta1 * m + (1 - beta1) * grad
                v1 = beta2 * v + (1 - beta2) * grad ** 2
                lr_t = lr *(math.sqrt(1.0 - beta2 ** st) / (1.0 - beta1 ** st))
                param.data -= lr_t * (m1 / (torch.sqrt(v1) + eps))
                param.data *= (1 - weight_decay * lr)
                
                # 存储state
                state["m"] = m1
                state["v"] = v1
                state["st"] = st

        return loss


# @Author: MingrHu
# @Date: 2026-05-14
# @Description: 构建学习率调度器
# @Param t: 当前步
# @Param amax: 最大学习率
# @Param amin: 最小学习率
# @Param Tw: 学习率衰减步长
# @Param Tc: 学习率衰减步长
# @Return: 学习率
def learning_rate_schedule(t:int,amax:float,amin:float,Tw:int,Tc:int):
    if t < Tw:
        return t / Tw * amax
    elif Tw <= t and t <= Tc:
        return amin + (1 + math.cos((t - Tw) / (Tc - Tw) * math.pi)) * (amax - amin) / 2
    elif t > Tc:
        return amin
    else:
        raise ValueError("Invalid params!")


# @Author: MingrHu
# @Date: 2026-05-14
# @Description: 构建梯度裁剪层
# @Param params: 参数列表
# @Param max_l2_norm: 最大梯度范数
# @Param eps: 小常数
# @Return: None
def gradient_clipping(params: Iterable[torch.nn.Parameter], max_l2_norm: float,eps = 1e-6):
    l2:float = 0
    g2:float = 0
    for it in params:
        if it.grad != None:
            g2 += (it.grad ** 2).sum().item() 
    l2 = math.sqrt(g2)
    for it in params:
        if it.grad != None and l2 > max_l2_norm:
            it.grad *= max_l2_norm / (eps + l2)

# @Author: MingrHu
# @Date: 2026-05-14
# @Description: 构建数据加载器
# @Param dataset: 数据集 # 输入的数据集是token ids
# @Param batch_size: 批量大小
# @Param context_length: 上下文长度
# @Param device: CPU/GPU
# @Return: 数据加载器
def data_loading(dataset:np.ndarray, batch_size: int, context_length: int, device: str)->tuple[torch.Tensor, torch.Tensor]:
    X = torch.zeros(
        (batch_size,context_length),
        dtype = torch.long
    )
    Y = torch.zeros(
        (batch_size,context_length),
        dtype = torch.long
    )

    # np是[ )
    smp_idx = np.random.randint(0,len(dataset) - context_length,batch_size)
    for i,idx in enumerate(smp_idx):
        chunk = dataset[idx:idx + context_length + 1] 
        x = chunk[:-1]
        y = chunk[1:]
        X[i] = torch.from_numpy(x)
        Y[i] = torch.from_numpy(y)
    X = X.to(device)
    Y = Y.to(device)
    return (X,Y)


# @Author: MingrHu
# @Date: 2026-05-14
# @Description: 保存模型检查点
# @Param model: 模型
# @Param optimizer: 优化器
# @Param iteration: 当前迭代次数
# @Param out: 输出路径
# @Return: None
def save_checkpoint(model:nn.Module, optimizer:torch.optim.Optimizer, iteration:int, 
                    out:str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint,out)
  

# @Author: MingrHu
# @Date: 2026-05-14
# @Description: 加载模型检查点
# @Param src: 输入路径
# @Param model: 模型
# @Param optimizer: 优化器
# @Return: 当前迭代次数
def load_checkpoint(src:str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
                    model:nn.Module, optimizer:torch.optim.Optimizer)->int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
    