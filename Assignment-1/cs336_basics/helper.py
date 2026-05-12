import torch,math
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


# (batch_size, ..., seq_len, d_q)
# (batch_size, ..., seq_len, d_k)
# (batch_size, ..., seq_len, d_v)
# (batch_size, ..., seq_len, seq_len)
def scaled_dot_product_attention(q: torch.Tensor,k: torch.Tensor,v: torch.Tensor,mask: torch.Tensor | None = None):
    qk_dot = q @ k.transpose(-2,-1)
    d_k = k.shape[-1] 
    score = qk_dot / torch.sqrt(torch.tensor(d_k))
    # 自动广播 不要用inf 用-1e9
    if mask is not None:
        score = score + ((~mask) * -1e9)
    attention = softmax(score,dim = -1)
    return attention @ v


# (b × s,vocab_size) (b)
def cross_entropy_loss(x: torch.Tensor,target: torch.Tensor):
    # 分子直接简化
    tar_x = x[torch.arange(len(target)),target] 

    # 计算对数和 分母
    max_val = torch.max(x,dim = -1,keepdim = True).values
    exp_x = torch.exp(x - max_val)
    exp_sum_x = exp_x.sum(dim = -1,keepdim = True)
    return (torch.log(exp_sum_x) - tar_x + max_val).mean()


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
