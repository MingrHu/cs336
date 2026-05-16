import torch
import torch.nn as nn
from einops import rearrange
from cs336_basics.helper import scaled_dot_product_attention

# @Author: MingrHu
# @Date: 2026-05-06
# @Description: 构建线性变换模块
# @Param in_features: 输入特征维度
# @Param out_features: 输出特征维度
# @Param device: CPU/GPU
# @Param dtype: 数据类型
# @Return: 线性变换模块
class MR_Model_linear(nn.Module):
    # shape: (batch_size,sequence_size,dim)
    # 构建线性变换模块
    def __init__(self, in_features:int, out_features:int, device = None, dtype = None):
        # 继承初始化父类
        super().__init__()
        self.device = device
        self.dtype = dtype
        weight = torch.zeros((out_features,in_features),device = device, dtype = dtype)
        std = torch.sqrt(2 / torch.tensor(in_features + out_features)).item()
        # 参考Task的初始化方法
        weight = nn.init.trunc_normal_(weight,0,std,-3 * std,3 * std)
        # register
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
class MR_Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device = None, dtype = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        weight = torch.zeros((num_embeddings,embedding_dim),device = device, dtype = dtype)
        weight = nn.init.trunc_normal_(weight,0,1,-1,1)
        # register
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


# @Author: MingrHu
# @Date: 2026-05-06
# @Description: 构建RMS归一化层
# @Param d_model: 输入特征维度
# @Param eps: 小常量
# @Param device: CPU/GPU
# @Param dtype: 数据类型
# @Return: RMS归一化层
class MR_RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.eps = eps
        self.dmodel = d_model
        # register
        self.weight = nn.Parameter(torch.ones(d_model,device = device, dtype = dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        x_temp = x ** 2
        # rmsa按照d_model的列求和做均值
        rmsa = torch.sqrt(x_temp.mean(dim = -1,keepdim= True) + self.eps)
        ret = x / rmsa
        return ret.to(in_type) * self.weight.T


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
class MR_SwiGLU(nn.Module):
    def __init__(self,d_model: int,dff:int, device = None, dtype = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.dmodel = d_model
        self.dff = dff
        # register
        self.w1 = MR_Model_linear(d_model,dff,device = device, dtype = dtype)
        self.w2 = MR_Model_linear(dff,d_model,device = device, dtype = dtype)
        self.w3 = MR_Model_linear(d_model,dff,device = device, dtype = dtype)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        # (...,s,d_model) @ (dff,d_model).T
        w1_x = self.w1(x)
        silu = w1_x * torch.sigmoid(w1_x)
        # (...,s,d_model) @ (dff,d_model).T
        w3_x = self.w3(x)
        glu = silu * w3_x
        # (...,s,dff) @ (d_model,dff).T
        return self.w2(glu)
    
    
# @Author: MingrHu
# @Date: 2026-05-06
# @Description: 构建RoPE层
# @Param theta: RoPE参数
# @Param d_model: 输入特征维度
# @Param max_seq_len: 最大序列长度
# @Param device: CPU/GPU
class MR_RoPE(nn.Module):
    def __init__(self, theta: float, d_model: int, max_seq_len: int, device = None, dtype = None) :
        super().__init__()
        # 参考公式实现
        self.device = device
        self.dtype = dtype
        self.d_k = d_model

        inv_freq = 1.0 / (theta ** (torch.arange(0,d_model,2).float() / d_model))
        seq_i = torch.arange(max_seq_len).float()
        # (max_seq_len,1) @ (1,d_model // 2)
        theta_ik = seq_i.reshape(-1,1) @ inv_freq.reshape(1,-1)

        self.cos_table = torch.cos(theta_ik)
        self.sin_table = torch.sin(theta_ik)

        # 移动到device 不注册参数
        self.cos_table = self.cos_table.to(device)
        self.sin_table = self.sin_table.to(device)

    def forward(self, x: torch.Tensor,token_positions:torch.Tensor) -> torch.Tensor:

        # 不能在原始的x上直接改
        out = x.clone()
        # TODO 性能问题 后续优化
        for k in range(0,self.d_k // 2):
            vec_x = x[...,2 * k]
            vec_y = x[...,2 * k + 1]
            # token_pos shape [batch,seq_len]
            cos = self.cos_table[token_positions, k].unsqueeze(-2)
            sin = self.sin_table[token_positions, k].unsqueeze(-2)
            new_x = vec_x * cos - vec_y * sin
            new_y = vec_y * cos + vec_x * sin

            out[...,2 * k] = new_x
            out[...,2 * k + 1] = new_y
        return out


# @Author: MingrHu
# @Date: 2026-05-06
# @Description: 构建多头自注意力层
# @Param d_model: 输入特征维度
# @Param num_heads: 头数
# @Param max_seq_len: 最大序列长度
# @Param theta: RoPE参数
# @Param token_positions: 位置编码张量
# @Param device: CPU/GPU
# @Param dtype: 数据类型
# @Return: 多头自注意力层
class MR_multihead_self_attention(nn.Module):
    def __init__(self,d_model: int,num_heads: int,max_seq_len: int = 1024,theta:float | None = None,device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.d_k = d_model // num_heads
        
        # mask 下三角矩阵 
        self.mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype = torch.bool,device = device))

        # 权重
        self.q_proj = MR_Model_linear(d_model,d_model)
        self.k_proj = MR_Model_linear(d_model,d_model)
        self.v_proj = MR_Model_linear(d_model,d_model)
        self.output_proj = MR_Model_linear(d_model,d_model)

        if theta is not None:
            self.rope = MR_RoPE(theta,self.d_k,self.max_seq_len,device,dtype)
        else:
            self.rope = None

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor | None = None) -> torch.Tensor:
        # (...,s,dq/dk/dv) 矩阵投影
        q,k,v = self.q_proj(x),self.k_proj(x),self.v_proj(x)
        # single_head_attention
        per_q = rearrange(q,"... seq (h per_dq) -> ... seq h per_dq",h = self.num_heads)
        per_k = rearrange(k,"... seq (h per_dk) -> ... seq h per_dk",h = self.num_heads)
        per_v = rearrange(v,"... seq (h per_dv) -> ... seq h per_dv",h = self.num_heads)

        s_q = rearrange(per_q,"... s h d -> ... h s d")
        s_k = rearrange(per_k,"... s h d -> ... h s d")
        s_v = rearrange(per_v,"... s h d -> ... h s d")

        # RoPE
        if self.rope is not None and token_positions is not None:
            s_q = self.rope.forward(s_q,token_positions)
            s_k = self.rope.forward(s_k,token_positions)

        mask = self.mask[:x.size(-2),:x.size(-2)]
        ret = scaled_dot_product_attention(s_q,s_k,s_v,mask)
        ret = rearrange(ret,"... h s per_dv -> ... s (h per_dv)")
        # [...,s,d_v] @ [d_model,d_v].T
        return self.output_proj(ret)


# @Author: MingrHu
# @Date: 2026-05-06
# @Description: 构建Transformer Block
# @Param d_model: 输入特征维度
# @Param num_heads: 头数
# @Param ffn_dim: FFN维度
# @Param max_seq_len: 最大序列长度
# @Param theta: RoPE参数
# @Param device: CPU/GPU
# @Param dtype: 数据类型
# @Return: Transformer Block
class MR_transformer_block(nn.Module):
    def __init__(self,d_model:int,num_heads:int,d_ff:int,max_seq_len:int,theta:float,device = None, dtype = None):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # 一些子模块初始化
        self.ln1 = MR_RMSNorm(d_model,device = device,dtype = dtype)
        self.ln2 = MR_RMSNorm(d_model,device = device,dtype = dtype)
        self.attn = MR_multihead_self_attention(d_model,num_heads,max_seq_len,theta,device = device,dtype = dtype)
        self.ffn = MR_SwiGLU(d_model,d_ff,device = device,dtype = dtype)

    # x shape (batch sequence_length d_model)
    # Return shape (batch sequence_length d_model)
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        # 1 LayerNorm1
        fx1 = self.ln1(x)

        # 生成token_positions
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        token_positions = torch.arange(seq_len,device = self.device,dtype = self.dtype)
        token_positions = token_positions.repeat(batch_size,1)

        # 2 MHA
        mha_out = self.attn(fx1,token_positions)

        # Add
        y1 = mha_out + x
        
        # 3 LayerNorm2
        fx2 = self.ln2(y1)

        # 4 FFN
        ffn_out = self.ffn(fx2)
        
        # Add
        y2 = ffn_out + y1
        return y2
        


class MR_transformer_lm(nn.Module):
    def __init__(self,vocab_size:int,context_length:int,d_model:int,num_layers:int,num_heads:int,d_ff:int,rope_theta:float,device = None, dtype = None):
        super().__init__()
        self.num_layers = num_layers
        # 基础组件
        self.lm_head = MR_Model_linear(d_model,vocab_size)
        self.ln_final = MR_RMSNorm(d_model)
        self.token_embeddings = MR_Embedding(vocab_size,d_model)

        # 这里用了ModuleList方法
        self.layers = nn.ModuleList([
            MR_transformer_block(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        

    def forward(self,in_indices:torch.Tensor)->torch.Tensor:
        # 1 获取输入
        # (vocab_size,d_model)  (batch,seq_len) = (batch,seq_len,d_model)
        x = self.token_embeddings(in_indices)

        # 2 执行block
        for layer in self.layers:
            x = layer(x)
        
        # 3 补充一个归一化
        x = self.ln_final(x)

        # 4 输出lm head
        return self.lm_head(x)