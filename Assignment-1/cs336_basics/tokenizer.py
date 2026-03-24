import re
import json

from typing import Iterable, Iterator
from utils import PAT

class MR_Tokenizer:
    def __init__(self, vocab, merges, special_tokens:list[str] | None = None):
        """
        从给定的词汇表、合并规则列表和（可选的）特殊词元列表构建分词器
        参数：
            vocab: dict[int, bytes]
            merges: list[tuple[bytes, bytes]]
            special_tokens: list[str] | None = None
        """
        self.vacab = vocab
        self.merges = merges
        if special_tokens != None:
            self.special_tokens:list[str] = special_tokens
        else:
            self.special_tokens:list[str] = []

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
        pattern:str = "|".join(map(re.escape,self.special_tokens))
        patrs = re.split(pattern,text)
        # 把文本先预处理 拆分为每个token
        for part in patrs:
            passage = re.findall(PAT,part)
            for token in passage:
                tokens.append(token)
        
        




    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        接收字符串可迭代对象（如Python文件句柄），返回一个生成器，惰性生成词元ID
        用于对无法直接加载到内存的大型文件进行内存高效的分词
        """

    def decode(self, ids: list[int]) -> str:
        """将词元ID序列解码为文本"""
       