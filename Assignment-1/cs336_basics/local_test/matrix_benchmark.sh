#!/bin/bash

# 矩阵大小（越大越接近真实算力，太小不准）
MATRIX_SIZE=4096

echo "===== CPU 矩阵计算 实测浮点算力测试 ====="
echo "CPU: Intel Xeon 8260 (64核 AVX512)"
echo "测试矩阵大小：${MATRIX_SIZE}x${MATRIX_SIZE} 双精度浮点"
echo "理论峰值：4915.2 GFLOPS"
echo "========================================="

# 用 Python + numpy 测（科学计算最标准、最常用）
# 如果没装numpy会自动安装
python3 -m pip install numpy -q -i https://pypi.tuna.tsinghua.edu.cn/simple

# 执行测试
python3 - <<END
import numpy as np
import time

# 设置线程数 = 你的CPU核心数（64）
import os
os.environ["OMP_NUM_THREADS"] = "64"
os.environ["OPENBLAS_NUM_THREADS"] = "64"
os.environ["MKL_NUM_THREADS"] = "64"
os.environ["VECLIB_MAXIMUM_THREADS"] = "64"
os.environ["NUMEXPR_NUM_THREADS"] = "64"

# 生成随机矩阵（双精度）
n = $MATRIX_SIZE
A = np.random.randn(n, n)
B = np.random.randn(n, n)

# 预热（让CPU提速、缓存加载）
C = A @ B

# 正式计时
start = time.time()
C = A @ B
end = time.time()

sec = end - start
flop = 2 * (n ** 3)        # 矩阵乘法浮点运算量公式
gflops = flop / sec / 1e9  # 换算成 GFLOPS
ratio = gflops / 4915.2    # 相对于理论峰值的比例

print(f"计算耗时：{sec:.3f} 秒")
print(f"实际算力：{gflops:.1f} GFLOPS")
print(f"理论折扣：{ratio:.1%}")

if ratio >= 0.6:
    print("✅ 性能优秀：接近优化库极限")
elif ratio >= 0.4:
    print("🆗 性能正常：标准科学计算水平")
else:
    print("⚠️  性能偏低：可能线程/优化没开足")
END