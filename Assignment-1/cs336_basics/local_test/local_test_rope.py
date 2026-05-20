import time,torch
from cs336_basics import transformer


def test_rope_performance():

    batch_size = 256
    num_heads = 16
    context_length = 256
    d_model = 512
    token_positions = torch.arange(context_length)
    token_positions = token_positions.repeat(batch_size,1)
    print(token_positions.shape)

    s = time.time()
    rope = transformer.MR_RoPE(10000.0,d_model,context_length)
    x = torch.randn(batch_size,num_heads,context_length,d_model)
    rope(x,token_positions)
    e = time.time()

    elapsed = e - s
    print(f"耗时：{elapsed:.4f} 秒")

if __name__ == "__main__":
    test_rope_performance()