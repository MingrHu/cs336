import torch
import torch.nn as nn
from einops import rearrange

# @Author: MingrHu
# @Date: 2026-05-06
# @Description: 构建线性变换模块
# @Param in_features: 输入特征维度
# @Param out_features: 输出特征维度
# @Param device: CPU/GPU
# @Param dtype: 数据类型
# @Return: 线性变换模块
# @FLOPs: 2 * m * n * p
class MR_Model_linear(nn.Module):
    # shape: (batch_size,sequence_size,dim)
    # 构建线性变换模块
    def __init__(self, in_features:int, out_features:int, device=None, dtype=None):
        # 继承初始化父类
        super().__init__()
        self.device = device
        self.dtype = dtype
        weight = torch.zeros((out_features,in_features),device = device, dtype = dtype)
        std = torch.sqrt(2 / torch.tensor(in_features + out_features)).item()
        # 参考Task的初始化方法
        weight = nn.init.trunc_normal_(weight,0,std,-3 * std,3 * std)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

# @Author: MingrHu
# @Date: 2026-05-06
# @Description: 构建嵌入层
# @Param num_embeddings: 嵌入维度
# @Param embedding_dim: 嵌入维度
# @Param device: CPU/GPU
# @Param dtype: 数据类型
# @Return: 嵌入层
# @FLOPs: m * n
class MR_Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device = None, dtype = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        embedding = torch.zeros((num_embeddings,embedding_dim),device = device, dtype = dtype)
        embedding = nn.init.trunc_normal_(embedding,0,1,-1,1)
        self.embedding = nn.Parameter(embedding)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]

# @Author: MingrHu
# @Date: 2026-05-06
# @Description: 构建RMS归一化层
# @Param d_model: 输入特征维度
# @Param eps: 小常量
# @Param device: CPU/GPU
# @Param dtype: 数据类型
# @Return: RMS归一化层
# @FLOPS: m * n
class MR_RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.eps = eps
        self.dmodel = d_model

    def forward(self, x: torch.Tensor,weights: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        x_temp = x ** 2
        # rmsa按照d_model的列求和做均值
        rmsa = torch.sqrt(x_temp.mean(dim = -1,keepdim= True) + self.eps)
        ret = x / rmsa
        return ret.to(in_type) * weights.T

# @Author: MingrHu
# @Date: 2026-05-06
# @Description: 构建SwiGLU层
# @Param d_model: 输入特征维度
# @Param dff: 中间层维度
# @Param weight1: 第一个权重矩阵
# @Param weight2: 第二个权重矩阵
# @Param weight3: 第三个权重矩阵
# @Param device: CPU/GPU
# @Param dtype: 数据类型
# @Return: SwiGLU层
# @FLOPs: 
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
    

# @Author: MingrHu
# @Date: 2026-05-06
# @Description: 构建RoPE层
# @Param theta: RoPE参数
# @Param d_model: 输入特征维度
# @Param max_seq_len: 最大序列长度
# @Param device: CPU/GPU
# @Return: RoPE层
class MR_RoPE(nn.Module):
    def __init__(self, theta: float, d_model: int, max_seq_len: int, device=None) :
        # 参考公式实现
        self.device = device
        self.d_k = d_model
        # 提前存表计算 TODO:可不作为模型参数 参考论文用register
        # shape [mx_seq_len,d]
        self.cos_table = torch.zeros((max_seq_len + 1,d_model // 2 + 1),device = device)
        self.sin_table = torch.zeros((max_seq_len + 1,d_model // 2 + 1),device = device)
        for i in range(max_seq_len):
            for k in range(1,d_model // 2 + 1):
                theta_ik = i / (theta ** ((2 * k - 2) / d_model))
                theta_tensor = torch.tensor(theta_ik)
                self.cos_table[i][k] = torch.cos(theta_tensor)
                self.sin_table[i][k] = torch.sin(theta_tensor)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

        # 不能在原始的x上直接改
        out = x.clone()
        for k in range(1,self.d_k // 2 + 1):
            vec_x = x[...,2 * k - 2]
            vec_y = x[...,2 * k - 1]
            # token_pos shape [batch,seq_len]
            new_x = vec_x * self.cos_table[token_positions,k] - vec_y * self.sin_table[token_positions,k]
            new_y = vec_y * self.cos_table[token_positions,k] + vec_x * self.sin_table[token_positions,k]

            out[...,2 * k - 2] = new_x
            out[...,2 * k - 1] = new_y
        return out
                

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


# @Author: MingrHu
# @Date: 2026-05-06
# @Description: 构建多头自注意力层
# @Param d_model: 输入特征维度
# @Param num_heads: 头数
# @Param q_weight: 查询权重矩阵
# @Param k_weight: 键权重矩阵
# @Param v_weight: 值权重矩阵
# @Param max_seq_len: 最大序列长度
# @Param theta: RoPE参数
# @Param token_positions: 位置编码张量
# @Param device: CPU/GPU
# @Param dtype: 数据类型
# @Return: 多头自注意力层
class MR_multihead_self_attention(nn.Module):
    def __init__(self,d_model: int,num_heads: int,q_weight:torch.Tensor,k_weight:torch.Tensor,v_weight:torch.Tensor,
                 max_seq_len: int = 1024,theta:float | None = None,device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        
        # mask 下三角矩阵 
        self.mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype = torch.bool))
        self.q_size,self.k_size,self.v_size = q_weight.size(0),k_weight.size(0),v_weight.size(0)
        # [...,d_combine,d_in]
        self.qkv = torch.cat([q_weight,k_weight,v_weight],dim = -2)
        self.theta = theta

    def forward(self,x:torch.Tensor,o_weight:torch.Tensor,token_positions:torch.Tensor | None = None) -> torch.Tensor:
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
        d_k = s_k.shape[-1]
        # RoPE
        if self.theta is not None and token_positions is not None:
            rope = MR_RoPE(self.theta,d_k,self.max_seq_len,device = self.device)
            s_q = rope.forward(s_q,token_positions)
            s_k = rope.forward(s_k,token_positions)

        mask = self.mask[:x.size(-2),:x.size(-2)]
        ret = scaled_dot_product_attention(s_q,s_k,s_v,mask)
        ret = rearrange(ret,"... h s per_dv -> ... s (h per_dv)")
        # [...,s,d_v] @ [d_model,d_v].T
        return ret @ o_weight.T 


# @Author: MingrHu
# @Date: 2026-05-06
# @Description: 构建Transformer Block
# @Param d_model: 输入特征维度
# @Param num_heads: 头数
# @Param ffn_dim: FFN维度
# @Param max_seq_len: 最大序列长度
# @Param theta: RoPE参数
# @Param q_weight: 查询权重矩阵
# @Param k_weight: 键权重矩阵
# @Param v_weight: 值权重矩阵
# @Param mut_out_weight: 多头自注意力输出权重矩阵
# @Param ln_weight1: 第一个LayerNorm权重矩阵
# @Param ln_weight2: 第二个LayerNorm权重矩阵
# @Param ffn_weight1: 第一个FFN权重矩阵
# @Param ffn_weight2: 第二个FFN权重矩阵
# @Param ffn_weight3: 第三个FFN权重矩阵
# @Param device: CPU/GPU
# @Param dtype: 数据类型
# @Return: Transformer Block
class MR_transformer_block(nn.Module):
    def __init__(self,d_model:int,num_heads:int,ffn_dim:int,max_seq_len:int,theta:float,
                 q_weight:torch.Tensor,k_weight:torch.Tensor,v_weight:torch.Tensor,mut_out_weight:torch.Tensor,
                 ln_weight1:torch.Tensor,ln_weight2:torch.Tensor,ffn_weight1:torch.Tensor,ffn_weight2:torch.Tensor,
                 ffn_weight3:torch.Tensor,device = None, dtype = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.mut_out_weight = mut_out_weight
        self.ln_weight1 = ln_weight1
        self.ln_weight2 = ln_weight2
        # 一些子模块初始化
        self.ln1 = MR_RMSNorm(d_model)
        self.ln2 = MR_RMSNorm(d_model)
        self.mha = MR_multihead_self_attention(d_model,num_heads,q_weight,k_weight,v_weight,max_seq_len,theta,device,dtype)
        self.ffn = MR_SwiGLU(d_model,ffn_dim,ffn_weight1,ffn_weight2,ffn_weight3,device,dtype)

    # x shape (batch sequence_length d_model)
    # Return shape (batch sequence_length d_model)
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        # 1 LayerNorm1
        fx1 = self.ln1.forward(x,self.ln_weight1)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        # 生成token_positions
        token_positions = torch.arange(seq_len,device = self.device,dtype = self.dtype)
        token_positions = token_positions.repeat(batch_size,1)
        # 2 MHA
        mha_out = self.mha.forward(fx1,self.mut_out_weight,token_positions)
        # Add
        y1 = mha_out + x
        # 3 LayerNorm2
        fx2 = self.ln2.forward(y1,self.ln_weight2)
        # 4 FFN
        ffn_out = self.ffn.forward(fx2)
        # Add
        y2 = ffn_out + y1
        return y2
        


class MR_transformer_lm(nn.Module):
    def __init__(self,vocab_size:int,context_length:int,d_model:int,num_layers:int,num_heads:int,d_ff:int,rope_theta:float,
                in_indices:torch.Tensor,weights:dict[str,torch.Tensor],device = None, dtype = None):
        super().__init__()
        token_embeddings = weights["token_embeddings.weight"]
        # [vocab_size,d_model]  [batch,seq_len] = [batch,seq_len,d_model]
        in_features = token_embeddings[in_indices]

        for i in range(num_layers):
            q_weight = weights[f"layers.{i}.attn.q_proj.weight"]
            k_weight = weights[f"layers.{i}.attn.k_proj.weight"]
            v_weight = weights[f"layers.{i}.attn.v_proj.weight"]
            mut_out_weight = weights[f"layers.{i}.attn.output_proj.weight"]
            ln_weight1 = weights[f"layers.{i}.ln1.weight"]
            ln_weight2 = weights[f"layers.{i}.ln2.weight"]
            ffn_weight1 = weights[f"layers.{i}.ffn.w1.weight"]
            ffn_weight2 = weights[f"layers.{i}.ffn.w2.weight"]
            ffn_weight3 = weights[f"layers.{i}.ffn.w3.weight"]
            transformer_block = MR_transformer_block(d_model,num_heads,d_ff,context_length,rope_theta,q_weight,k_weight,v_weight,
                                                         mut_out_weight,ln_weight1,ln_weight2,ffn_weight1,ffn_weight2,ffn_weight3)
            in_features = transformer_block.forward(in_features)
    
        self.in_features = in_features
        self.ln_final_weight = weights["ln_final.weight"]
        self.out_weight = weights["lm_head.weight"]
        self.ln_final_fx = MR_RMSNorm(d_model)
        

    def forward(self)->torch.Tensor:
        ln_final = self.ln_final_fx.forward(self.in_features,self.ln_final_weight)  
        # [b,s,d] @ [vocab_size,d_model].T
        liner_ret = ln_final @ self.out_weight.T
        return liner_ret