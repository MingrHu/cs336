import torch
import torch.nn as nn
from einops import rearrange, einsum

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


class MR_multihead_self_attention(nn.Module):
    def __init__(self,d_model: int,num_heads: int,q_weight:torch.Tensor,
                 k_weight:torch.Tensor,v_weight:torch.Tensor,device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.d = d_model // num_heads
        self.q_size,self.k_size,self.v_size = q_weight.size(0),k_weight.size(0),v_weight.size(0)
        # [...,d_in]
        self.qkv = torch.cat([q_weight,k_weight,v_weight],dim = -2)

    def forward(self,x:torch.Tensor,o_weight:torch.Tensor) -> torch.Tensor:
        # [...,s,d_in] @ [d_combine,d_in].T
        # [...,s,d_combine]
        QKV = x @ self.qkv.T
        # [...,s,dq/dk/dv]
        q,k,v = torch.split(QKV,[self.q_size,self.k_size,self.v_size],dim = - 1)
        # single_head
        per_q = rearrange(q,"... seq (h per_dq) -> ... seq h per_dq",h = self.num_heads)
        per_k = rearrange(k,"... seq (h per_dk) -> ... seq h per_dk",h = self.num_heads)
        per_v = rearrange(v,"... seq (h per_dv) -> ... seq h per_dv",h = self.num_heads)

        s_q = rearrange(per_q,"... s h d -> ... h s d")
        s_k = rearrange(per_k,"... s h d -> ... h s d")
        s_v = rearrange(per_v,"... s h d -> ... h s d")

        ret = scaled_dot_product_attention(s_q,s_k,s_v)
        ret = rearrange(ret,"... h s per_dv -> ... s (h per_dv)")
        # [...,s,d_v] @ [d_model,d_v].T
        return ret @ o_weight.T 