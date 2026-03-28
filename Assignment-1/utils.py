import os
import multiprocessing
import regex as re
from typing import BinaryIO
# gpt2使用的正则表达式 分割tokens
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir,"output")
os.makedirs(output_dir,exist_ok=True)



#####################Function#####################
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            
            found_at = -1
            # Find the special token in the mini chunk
            # min_pos
            for sp_token in split_special_token:
                pos = mini_chunk.find(sp_token)
                if pos != - 1:
                    found_at = pos if found_at == -1 or found_at > pos else found_at
            
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


## Usage
# with open(..., "rb") as f: # type: ignore
#     num_processes = 4
#     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

#     # The following is a serial implementation, but you can parallelize this
#     # by sending each start/end pair to a set of processes.
#     for start, end in zip(boundaries[:-1], boundaries[1:]):
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         # Run pre-tokenization on your chunk and store the counts for each pre-token




def get_freq_dic(chunk:str,special_tokens:list[str])->dict[tuple[bytes,...],int]:
    # 参考实验手册 特殊token的匹配模式
    pattern:str = "|".join(map(re.escape,special_tokens))
    # 先按特殊token分为不同的text段落 一个段落有多个单词 分别以空格划分
    docs = re.split(pattern,chunk)
    word_counts:dict[tuple[bytes,...],int] = {}
    # 获取以bytes格式的词元频率
    for part in docs:
        # 按照实验的规则分割tokens
        tokens = re.findall(PAT, part)
        # tokens = part.split()
        # print(tokens)
        for token in tokens:
            text_bytes = token.encode("utf-8")
            key = tuple(bytes([b]) for b in text_bytes)
            word_counts[key] = word_counts.get(key, 0) + 1
    return word_counts

def handle_func(input_path:str,start:int,end:int,sp_tokens:list[str],q:multiprocessing.Queue):
    with open(input_path,'rb') as f:
        f.seek(start)
        data = f.read(end - start)
        chunk = data.decode("utf-8", errors="ignore")
        q.put(get_freq_dic(chunk,sp_tokens))