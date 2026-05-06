import torch

# ====================== 1. 张量初始化 ======================
# 自定义数据
t = torch.tensor([[1., 2.], [3., 4.]])

# 全0 / 全1
zeros = torch.zeros(3, 5)
ones = torch.ones(3, 5)

# 固定值
full = torch.full((2, 4), fill_value=10)

# 单位矩阵
eye = torch.eye(3)

# 随机
rand = torch.rand(2, 3)       # [0,1) 均匀分布
randn = torch.randn(2, 3)     # 标准正态 N(0,1)
randint = torch.randint(0, 10, (2, 3))

# 序列
arange = torch.arange(0, 10, 1)
linspace = torch.linspace(0, 1, 5)

# 下三角为1其余为0的矩阵
seq_len = 8
mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

# ====================== 2. 形状/维度操作 ======================
x = torch.randn(4, 6)

# 重塑
reshape_x = x.reshape(2, -1)
flatten_x = x.flatten()

# 增加/压缩维度
unsq = x.unsqueeze(0)   # (1,4,6)
sq = unsq.squeeze(0)

# 转置
t1 = x.T
t2 = x.transpose(0, 1)

# 维度交换
permute_x = torch.randn(2, 3, 4).permute(2, 0, 1)

# [2,8,512] → [2,8,8,64]
x_reshape = x.reshape(batch, seq_len, num_heads, -1)
print(x_reshape.shape)  # torch.Size([2, 8, 8, 64])
# [batch, seq_len, heads, dk] 
# → [batch, heads, seq_len, dk]
x_perm = x_reshape.permute(0, 2, 1, 3)
print(x_perm.shape)  # torch.Size([2, 8, 8, 64])

# ====================== 3. 逐元素运算 ======================
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

add = a + b
sub = a - b
mul_elem = a * b        # 逐元素相乘
div = a / b
pow2 = a ** 2   # 所有数平方
pow2_2 = torch.square(a)
sqrt_a = torch.sqrt(a)
abs_a = torch.abs(a)

exp_a = torch.exp(a)
log_a = torch.log(a)

# 符号、取整
sign_a = torch.sign(a)
floor_a = torch.floor(a)
ceil_a = torch.ceil(a)

# ====================== 4. 矩阵/线性代数运算 ======================
# 矩阵乘法
matmul1 = a @ b
matmul2 = torch.matmul(a, b)

# 批量矩阵乘法 (B, N, M) @ (B, M, P)
batch_a = torch.randn(8, 32, 16)
batch_b = torch.randn(8, 16, 24)
batch_matmul = torch.bmm(batch_a, batch_b)

# 内积、行列式、逆
dot = torch.dot(a.flatten(), b.flatten())
inv_a = torch.linalg.inv(a + torch.eye(2)*1e-6)

# ====================== 5. 统计聚合 ======================
sum_all = a.sum()
sum_dim = a.sum(dim=0, keepdim=True)

mean_all = a.mean()
max_val, max_idx = a.max(dim=1)
min_val, min_idx = a.min(dim=1)

argmax_a = torch.argmax(a, dim=1)
argmin_a = torch.argmin(a, dim=1)

std_a = a.std()
var_a = a.var()

# 归一化
norm_a = (a - a.mean(dim=1, keepdim=True)) / a.std(dim=1, keepdim=True)

# ====================== 6. 拼接 & 拆分 ======================
# 拼接
cat_row = torch.cat([a, b], dim=0)
cat_col = torch.cat([a, b], dim=1)

# 堆叠(新增维度)
stack_x = torch.stack([a, b], dim=0)

# 拆分
chunk_x = torch.chunk(a, chunks=2, dim=1)
split_x = torch.split(a, split_size_or_sections=[1,1], dim=1)

# ====================== 7. 索引 & 掩码 ======================
# 基础索引
item = a[0, 1]
row = a[0, :]

# 条件筛选
mask = a > 2
a_mask = a[mask]

# .where 条件赋值
a_where = torch.where(a > 2, torch.tensor(10.), a)

# ====================== 8. 自动求导 核心 ======================
w = torch.randn(3, 2, requires_grad=True)
x = torch.randn(2, 1)

y = w @ x
loss = y.sum()

loss.backward()     # 反向传播
grad_w = w.grad     # 取出梯度

# 清空梯度
w.grad.zero_()

# 禁止梯度推理
with torch.no_grad():
    out = w @ x

# ====================== 9. 常用激活函数 ======================
relu_x = torch.relu(a)
sigmoid_x = torch.sigmoid(a)
tanh_x = torch.tanh(a)
leaky_relu_x = torch.nn.functional.leaky_relu(a, 0.2)

# ====================== 10. 设备 & 类型转换 ======================
# 类型转换
float_a = a.float()
int_a = a.int()

# 设备切换
cuda_a = a.cuda()
cpu_a = cuda_a.cpu()

# 原地修改（慎用）
a.add_(1.)