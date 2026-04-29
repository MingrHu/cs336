import torch
import torch.nn as nn
import numpy as np

class MR_Model_linear(nn.Module):
    # shape: (batch_size,sequence_size,dim)
    # 构建线性变换模块
    def __init__(self, in_features, out_features, device=None, dtype=None):
        # 继承初始化父类
        super().__init__()
        self.device = device
        self.dtype = dtype
        weight = torch.zeros((out_features,in_features),device=device, dtype=dtype)
        std = np.sqrt(2 / (in_features + out_features))
        # 参考Task的初始化方法
        weight = nn.init.trunc_normal_(weight,0,std,-3 * std,3 * std)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class MR_Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        embedding = torch.zeros((num_embeddings,embedding_dim),device=device, dtype=dtype)
        embedding = nn.init.trunc_normal_(embedding,0,1,-1,1)
        self.embedding = nn.Parameter(embedding)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]

class MR_RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.eps = eps
        self.dmodel = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        x_temp = x ** 2
        rmsa = torch.sqrt(x_temp.mean(dim = -1,keepdim= True) + self.eps)
        ret = x / rmsa
        return ret.to(in_type)

class MR_SwiGLU(nn.Module):
    def __init__(self,d_model: int,dff:int,weight1: torch.Tensor,weight2: torch.Tensor,weight3: torch.Tensor, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.dmodel = d_model
        self.dff = dff
        self.w1 = weight1
        self.w2 = weight2
        self.w3 = weight3

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        # shape = (...,dff)
        w1_x = x @ self.w1.T
        silu = w1_x * torch.sigmoid(w1_x)
        # shape = (...,dff)
        w3_x = x @ self.w3.T
        glu = silu * w3_x
        return glu @ self.w2.T
    

class MR_RoPE():
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) :
        """
            Construct the RoPE module and create buffers if needed.
            theta: float Θ value for the RoPE
            d_k: int dimension of query and key vectors
            max_seq_len: int Maximum sequence length that will be inputted
            device: torch.device | None = None Device to store the buffer on
        """
        self.device = device
        self.const_theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.cos_table = torch.zeros((max_seq_len,d_k // 2))
        self.sin_table = torch.zeros((max_seq_len,d_k // 2))
        for i in range(max_seq_len):


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
            Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
            Note that you should tolerate x with an arbitrary number of batch dimensions. You should
            assume that the token positions are a tensor of shape (..., seq_len) specifying the token
            positions of x along the sequence dimension.
            You should use the token positions to slice your (possibly precomputed) cos and sin tensors
            along the sequence dimension.
        """