import torch

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
