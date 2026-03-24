import os
# gpt2使用的正则表达式 分割tokens
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir,"output")
os.makedirs(output_dir,exist_ok=True)