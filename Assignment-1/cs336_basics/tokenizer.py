import regex as re
import json

from typing import Iterable, Iterator
from cs336_basics.utils import PAT

class MR_Tokenizer:
    def __init__(self, vocab:dict[int,bytes], merges:list[tuple[bytes,bytes]], special_tokens:list[str] | None = None):
        """
        从给定的词汇表、合并规则列表和（可选的）特殊词元列表构建分词器
        参数：
            vocab: dict[int, bytes]
            merges: list[tuple[bytes, bytes]]
            special_tokens: list[str] | None = None
        """
        self.vocab = vocab
        self.merges = merges
        self.dic_token_id:dict[bytes,int] = {}
        if special_tokens != None:
            self.special_tokens:list[str] = special_tokens
        else:
            self.special_tokens:list[str] = []
        # token对id的映射
        for id,token in vocab.items():
            self.dic_token_id[token] = id
        

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens:list[str] | None = None):
        """
        类方法：从序列化的词汇表、合并规则文件（与BPE训练代码的输出格式一致）和（可选的）特殊词元列表构建并返回Tokenizer
        额外参数：
            vocab_filepath: str
            merges_filepath: str
            special_tokens: list[str] | None = None
        """
        vocab:dict[int,bytes] = {}
        merges:list[tuple[bytes,bytes]] = []
        spec_tokens:list[str] = []
        if special_tokens != None:
            spec_tokens = special_tokens
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges_data = json.load(f)

        vocab = {int(k): bytes.fromhex(v) for k, v in vocab_data.items()}
        merges = [(bytes.fromhex(a), bytes.fromhex(b))for a, b in merges_data]

        return cls(vocab, merges, spec_tokens)
        


    def encode(self, text: str) -> list[int]:
        """将输入文本编码为词元ID序列"""
        if text == "":
            return []
        tokens:list[bytes] = []
        parts:list[str] = []
        sp_tokens:list[str] = sorted(self.special_tokens,key = len,reverse=True)
        
        # 特殊词元列表为空时不应该拆分文本
        if self.special_tokens != []:
            pattern:str = "|".join(map(re.escape,sp_tokens))
            parts = re.split(f'({pattern})', text) 
            sp_tokens = re.findall(pattern,text)
        else:
            parts.append(text)
        # 把文本先预处理 拆分为每个token
        for part in parts:
            # 如果是特殊token 则直接添加占位
            if part in self.special_tokens:
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
                ret.append(self.dic_token_id[sp_tokens[sp_idx].encode("utf-8")])
                sp_idx += 1
                continue
            text_bytes = tuple(bytes([b]) for b in token)
            while True:
                max_level = len(self.vocab)
                merge_rule:bytes = b""
                for idx in range(len(text_bytes) - 1):
                    pair = text_bytes[idx] + text_bytes[idx+1]
                    if self.dic_token_id.get(pair) == None:
                        continue
                    if self.dic_token_id[pair] < max_level:
                        max_level = self.dic_token_id[pair]
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
                ret.append(self.dic_token_id[b])
        return ret


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        接收字符串可迭代对象（如Python文件句柄），返回一个生成器，惰性生成词元ID
        用于对无法直接加载到内存的大型文件进行内存高效的分词
        """
        # 可以去了解一下python的迭代器类型
        for it_str in iterable:
            yield from self.encode(it_str)

    def decode(self, ids: list[int]) -> str:
        """将词元ID序列解码为文本"""
        byte_buffer:bytes = b""
        for id in ids:
            if id in self.vocab:
                byte_buffer += self.vocab[id]
            else:
                raise ValueError(f"Invalid token ID: {id}")
            
        return byte_buffer.decode("utf-8",errors = "replace")

if __name__ == "__main__":
    tokenizer = MR_Tokenizer({},[])
    tokenizer.encode("s")
   
