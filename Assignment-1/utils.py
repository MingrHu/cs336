import os
import multiprocessing
import regex as re
from typing import BinaryIO
# gpt2使用的正则表达式 分割tokens
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir,"output")
os.makedirs(output_dir,exist_ok=True)



#####################Preprocess#####################
# 必须输入长度降序排列的special_tokens
# 该函数只是尽可能的去切 不保证每个区间只有一个特殊token
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    if not split_special_token or desired_num_chunks <= 1:
        return [0, file.seek(0, os.SEEK_END)]

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
    max_token_len = len(split_special_token[0])
    overlap = max_token_len - 1 if max_token_len > 0 else 0

    for bi in range(1, len(chunk_boundaries) - 1):
        # 防止正好切到特殊token
        initial_position = max(0, chunk_boundaries[bi] - overlap)
        file.seek(initial_position)  # Start at boundary guess

        current_pos = initial_position
        overlap_buffer = b""
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            search_buffer = overlap_buffer + mini_chunk
            
            found_at = -1
            matched_len = 0
            # Find the special token in the mini chunk
            # min_pos
            for sp_token in split_special_token:
                pos = search_buffer.find(sp_token)
                if pos != - 1:
                    if found_at == -1 or pos < found_at or (pos == found_at and len(sp_token) > matched_len):
                        found_at = pos
                        matched_len = len(sp_token)
            
            # 找到了就停
            if found_at != -1:
                absolute_pos = (current_pos - len(overlap_buffer)) + found_at + matched_len
                chunk_boundaries[bi] = absolute_pos
                break

            overlap_buffer = search_buffer[-overlap:] if overlap > 0 else b""
            current_pos += len(mini_chunk)

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


## Usage for Preprocess
# with open(..., "rb") as f: # type: ignore
#     num_processes = 4
#     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

#     # The following is a serial implementation, but you can parallelize this
#     # by sending each start/end pair to a set of processes.
#     for start, end in zip(boundaries[:-1], boundaries[1:]):
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         # Run pre-tokenization on your chunk and store the counts for each pre-token


#####################BPE multiple process#####################
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

def handle_bpe_func(input_path:str,start:int,end:int,sp_tokens:list[str],q:multiprocessing.Queue):
    with open(input_path,'rb') as f:
        f.seek(start)
        data = f.read(end - start)
        chunk = data.decode("utf-8", errors="ignore")
        q.put(get_freq_dic(chunk,sp_tokens))


#####################Tokenizer multiple process#####################
def exec_tokenizer_func(special_tokens:list[str],text:str,dic_token_id:dict[bytes,int],max_len:int)->list[int]:
    tokens:list[bytes] = []
    parts:list[str] = []
    sp_tokens = special_tokens

    # 特殊词元列表为空时不应该拆分文本
    if special_tokens != []:
        pattern:str = "|".join(map(re.escape,sp_tokens))
        parts = re.split(f'({pattern})', text) 
        # 按顺序拿出所有的特殊token
        sp_tokens = re.findall(pattern,text)
    else:
        parts.append(text)
    # 把文本先预处理 拆分为每个token
    for part in parts:
        # 如果是特殊token 则直接添加占位
        if part in special_tokens:
            tokens.append(b"")
            continue
        passage = re.findall(PAT,part)
        for token in passage:
            tokens.append(token.encode("utf-8"))
    
    ret:list[int] = []

    # 需要按照生成的合并规则的顺序进行应用
    # 复杂度 假设每个token的长度为k 有m个token 则复杂度为O(m*k2)
    sp_idx = 0
    for token in tokens:
        if token == b"":
            ret.append(dic_token_id[sp_tokens[sp_idx].encode("utf-8")])
            sp_idx += 1
            continue
        text_bytes = tuple(bytes([b]) for b in token)
        while True:
            max_level = max_len 
            merge_rule:bytes = b""
            for idx in range(len(text_bytes) - 1):
                pair = text_bytes[idx] + text_bytes[idx+1]
                if dic_token_id.get(pair) == None:
                    continue
                if dic_token_id[pair] < max_level:
                    max_level = dic_token_id[pair]
                    merge_rule = pair
            if merge_rule == b"":
                break
            new_text_bytes:list[bytes] = []
            j = 0
            while j < len(text_bytes):
                if j < len(text_bytes) - 1 and text_bytes[j] + text_bytes[j+1] == merge_rule:
                    new_text_bytes.append(merge_rule)
                    j += 2
                else:
                    new_text_bytes.append(text_bytes[j])
                    j += 1
            text_bytes = tuple(new_text_bytes)
        for b in text_bytes:
            ret.append(dic_token_id[b])
    return ret

def handle_tokenizer_func(input_path:str,start:int,end:int,sp_tokens:list[str],
                          dic_token_id:dict[bytes,int],vocab_size:int,q:multiprocessing.Queue,chunk_idx:int):
    with open(input_path,'rb') as f:
        f.seek(start)
        data = f.read(end - start)
        chunk = data.decode("utf-8", errors = "ignore")
        q.put((chunk_idx,exec_tokenizer_func(sp_tokens,chunk,dic_token_id,vocab_size)))
