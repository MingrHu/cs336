import torch
import torch.nn as nn
import numpy as np

class MR_Model_linear(nn.Module):
    # shape: (batch_size,sequence_size,dim)
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        构建线性变换模块
        参数：
            in_features: int：输入的最后一维维度
            out_features: int：输出的最后一维维度
            device: torch.device | None = None：参数存储的设备
            dtype: torch.dtype | None = None：参数的数据类型
        """
        super().__init__()
        weight = torch.zeros((out_features,in_features),device=device, dtype=dtype)
        std = np.sqrt(2 / (in_features + out_features))
        weight = nn.init.trunc_normal_(weight,0,std,-3 * std,3 * std)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class MR_Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
            an embedding module. This function should accept the following parameters:
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        embedding = torch.zeros((num_embeddings,embedding_dim),device=device, dtype=dtype)
        embedding = nn.init.trunc_normal_(embedding,0,1,-1,1)
        self.embedding = nn.Parameter(embedding)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]

class MR_RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.dmodel = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        x_temp = x ** 2
        rmsa = torch.sqrt(x_temp.mean(dim = -1,keepdim= True) + self.eps)
        ret = x / rmsa
        return ret.to(in_type)

