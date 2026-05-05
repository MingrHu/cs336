import torch
import torch.nn as nn

class MR_Model_linear(nn.Module):
    # shape: (batch_size,sequence_size,dim)
    # 构建线性变换模块
    def __init__(self, in_features:int, out_features:int, device=None, dtype=None):
        # 继承初始化父类
        super().__init__()
        self.device = device
        self.dtype = dtype
        weight = torch.zeros((out_features,in_features),device=device, dtype=dtype)
        std = torch.sqrt(2 / torch.tensor(in_features + out_features)).item()
        # 参考Task的初始化方法
        weight = nn.init.trunc_normal_(weight,0,std,-3 * std,3 * std)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class MR_Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device=None, dtype=None):
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
        # rmsa按照d_model的列求和做均值
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
        # 参考公式实现
        self.device = device
        self.d_k = d_k
        # 提前存表计算 TODO:可不作为模型参数 参考论文用register
        self.cos_table = torch.zeros((max_seq_len + 1,d_k // 2 + 1),device = device)
        self.sin_table = torch.zeros((max_seq_len + 1,d_k // 2 + 1),device = device)
        for i in range(max_seq_len):
            for k in range(1,d_k // 2 + 1):
                theta_ik = i / (theta ** ((2 * k - 2) / d_k))
                theta_tensor = torch.tensor(theta_ik)
                self.cos_table[i][k] = torch.cos(theta_tensor)
                self.sin_table[i][k] = torch.sin(theta_tensor)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

        # 不能在原始的x上直接改
        out = x.clone()
        for k in range(1,self.d_k // 2 + 1):
            vec_x = x[...,2 * k - 2]
            vec_y = x[...,2 * k - 1]
            
            new_x = vec_x * self.cos_table[token_positions,k] - vec_y * self.sin_table[token_positions,k]
            new_y = vec_y * self.cos_table[token_positions,k] + vec_x * self.sin_table[token_positions,k]

            out[...,2 * k - 2] = new_x
            out[...,2 * k - 1] = new_y
        return out
                



        