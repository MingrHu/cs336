import regex as re
import json
import multiprocessing  

from typing import Iterable, Iterator
from utils import (PAT,
                   find_chunk_boundaries,handle_tokenizer_func,
                   exec_tokenizer_func)

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
        ret:list[int] = exec_tokenizer_func(self.special_tokens,text,self.dic_token_id,self.vocab)
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
    
    def _multiple_encode(self,input_path:str)->list[int]:
        """针对大文件的多进程程编码"""
        q = multiprocessing.Queue()
        process_ins:list[multiprocessing.Process] = []
        ret:list[int] = []
        boundaries:list[int] = []

        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f,8,[b.encode('utf-8') for b in self.special_tokens])

        for start,end in zip(boundaries[:-1],boundaries[1:]):
            p = multiprocessing.Process(
                target = handle_tokenizer_func,
                args =(input_path,start,end,
                       self.special_tokens,self.dic_token_id,self.vocab,q))
            process_ins.append(p)
            p.start()

        size = len(process_ins)
        print(f"---------共有{size}个进程运行，获得{size}个reduce-----------")
        for idx in range(size):
            print(f"🤔#####开始merge######🤓 当前处于第{idx + 1}个")
            # 每个reduce的结果都是一个list[int]
            for val in q.get():
                ret.append(val)

        for p in process_ins:
            p.join()

        print(f"✅ map-reduce 结束 返回预处理后的词元ID序列")
        return ret

if __name__ == "__main__":
    tokenizer = MR_Tokenizer({},[])
    tokenizer.encode("s")
   
