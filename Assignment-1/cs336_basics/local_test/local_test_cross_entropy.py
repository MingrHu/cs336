import torch,time
from cs336_basics import helper


def incorrect_loss_cal(x: torch.Tensor,target: torch.Tensor):
   # (b × s,vocab_size) (b)
    # 分子直接简化
    tar_x = x[torch.arange(len(target),device = target.device),target] 

    # 计算对数和 分母
    max_val = torch.max(x,dim = -1,keepdim = True).values
    exp_x = torch.exp(x - max_val)
    exp_sum_x = exp_x.sum(dim = -1,keepdim = True)
    return (torch.log(exp_sum_x) - tar_x + max_val).mean()


def test_cross_entropy_loss():
    batch_size = 256
    context_length = 256
    vocab_size = 10000

    logits = torch.randn((batch_size,context_length,vocab_size),dtype=torch.float32)
    targets = torch.randint(low=0,high=vocab_size,size=(batch_size, context_length),dtype=torch.long)

    s1 = time.time()
    loss = helper.cross_entropy_loss(logits.reshape(-1, vocab_size),targets.reshape(-1))
    e1 = time.time()
    elapsed1 = e1 - s1
    print(f"正确维度计算损失耗时：{elapsed1:.4f} 秒")
    print(f"正确维度计算损失值:{loss}")

    s2 = time.time()
    loss = incorrect_loss_cal(logits.reshape(-1, vocab_size),targets.reshape(-1))
    e2 = time.time()
    elapsed2 = e2 - s2
    print(f"错误维度计算损失耗时：{elapsed2:.4f} 秒")
    print(f"错误维度计算损失值:{loss}")


if __name__ == "__main__":
    test_cross_entropy_loss()