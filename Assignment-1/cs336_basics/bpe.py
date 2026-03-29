import regex as re
import json
import os
import multiprocessing

from typing import BinaryIO
from utils import (PAT,output_dir,current_dir,
    find_chunk_boundaries,
    handle_bpe_func,
    get_freq_dic
)

class MR_BPE:

    def __init__(self,input_path:str | os.PathLike,vocab_size:int,special_tokens:list[str]) -> None:
        # print("--------BPE init-------")
        # 路径对象转为str
        input_path = str(input_path) 
        self.input_path = input_path
        self.special_tokens = special_tokens
        self.vocab_size = vocab_size
        self.dic:dict[tuple[bytes,...],int] = {}
        self.merges:list[tuple[bytes,bytes]] = []
        self.vocab:dict[int,bytes] = {}
        self.vocab_cur_size = 0
        self.text:str = ""

        # 初始化初始词汇表
        for idx,sp_token in enumerate(special_tokens):
            self.vocab[idx] = self.vocab.get(idx,sp_token.encode("utf-8"))
        self.vocab_cur_size = len(self.vocab)
        # 加入0～255
        for i in range(256):
            self.vocab[self.vocab_cur_size] = self.vocab.get(self.vocab_cur_size,bytes([i]))
            self.vocab_cur_size += 1

    def pre_process_text(self,max_memory:int = 1 << 32):
        # 文本 → 按特殊 token 切分成大段 → 每段用 PAT 抽词 → 每个词转 bytes 元组 → 统计词频 dict[tuple, int]
        with open(self.input_path,"rb") as f:
            f.seek(0,os.SEEK_END)
            size = f.tell()
        if size > max_memory:
            self.dic = self._multiple_pre_process_text() # type: ignore
        else:
            with open(self.input_path, "r", encoding="utf-8") as f:
                self.text = f.read()
            self.dic = get_freq_dic(self.text,self.special_tokens)
        # print(self.dic)
        return
        
    def _multiple_pre_process_text(self,num_processes:int = 8)->dict[tuple[bytes,...],int]:
        q = multiprocessing.Queue()
        process_ins:list[multiprocessing.Process] = []
        ret:dict[tuple[bytes,...],int] = {}
        boundaries:list[int] = []

        with open(self.input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f,num_processes,[b.encode('utf-8') for b in self.special_tokens])

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            p = multiprocessing.Process(
                target = handle_bpe_func,
                args =(self.input_path,start,end,self.special_tokens,q))
            process_ins.append(p)
            p.start()

        size = len(process_ins)
        print(f"---------共有{size}个进程运行，获得{size}个reduce-----------")
        for idx in range(size):
            print(f"🤔#####开始merge######🤓 当前处于第{idx + 1}个")
            for k,v in q.get().items():
                ret[k] = ret.get(k,0) + v

        for p in process_ins:
            p.join()
        print(f"✅ map-reduce 结束 返回预处理freq词典")
        return ret

    def __find_max_freq(self,dic:dict[tuple[bytes,bytes],dict[int,int]])->tuple[bytes,bytes]:
        max_freq = -1
        merge_rule:tuple[bytes,bytes] = (b"",b"")
        for k,v in dic.items():
            sm = sum(v.values())
            if sm > max_freq:
                max_freq = sm
                merge_rule = k
            elif sm == max_freq:
                # 频率相同时，选择字典序更大的键
                merge_rule = k if merge_rule < k else merge_rule
        self.merges.append(merge_rule)
        new_token = merge_rule[0] + merge_rule[1]
        self.vocab.setdefault(self.vocab_cur_size,new_token) 
        self.vocab_cur_size += 1
        # print(f"max bi = {bi} and freq = {max_freq}")
        return merge_rule
        
    def train_bpe(self):
        print("😎开始bpe训练 请等待......")
        # {low: 5, lower: 2, widest: 3, newest: 6}
        # {['l','o','w']:5,['l','o','w','r']:2,......}
        # {lo:7, ow:7, we:8, er:2, wi:3, id:3, de:3, es:9, st:9, ne:6, ew:6}
        # {(l,o,w):5, (l,o,w,e,r):2, (w,i,d,e,st):3, (n,e,w,e,st):6}
        temp_dic:dict[tuple[bytes,bytes],dict[int,int]] = {}
        idx_dic:dict[int,tuple[bytes,...]] = {}
        idx_count:dict[int,int] = {}
        idx = -1
        for key_tuple,freq in self.dic.items():
            idx += 1
            if len(key_tuple) == 1:
                continue
            for i in range(len(key_tuple) - 1):
                next_i = i + 1
                merge_rule = (key_tuple[i],key_tuple[next_i])
                # 添加当前cur对应的位置和idx索引 idx对应到具体的key_tuple i对应位置
                temp_dic.setdefault(merge_rule,{}).setdefault(idx,0)
                temp_dic.setdefault(merge_rule,{})[idx] += freq
            # idx <--> key_tuple idx唯一映射到key_tuple 方便后续根据idx值找对应的key_tuple
            idx_dic[idx] = idx_dic.get(idx,key_tuple)
            idx_count[idx] = idx_count.get(idx,0) + freq
        rounds = 0

        while True:
            # 停止条件
            if len(temp_dic) == 0 or self.vocab_cur_size >= self.vocab_size:
                break
            # 拿到最大的合并规则
            bi = self.__find_max_freq(temp_dic)
            # 拿[idx,freq]的list
            posItem = temp_dic.get(bi,{})
            idx_list = list(posItem.keys())
            for idx in idx_list:
                j,cnt = 0,idx_count[idx]
                old_tokens = idx_dic[idx]
                new_tokens:list[bytes] = []

                while j < len(old_tokens):
                    if j + 1 < len(old_tokens) and old_tokens[j] == bi[0] and old_tokens[j + 1] == bi[1]:
                        new_tokens.append(bi[0] + bi[1])
                        j += 2
                    else:
                        new_tokens.append(old_tokens[j])
                        j += 1
                # check
                for i in range(len(old_tokens) - 1):
                    pair = (old_tokens[i],old_tokens[i + 1])
                    temp_dic[pair][idx] -= cnt
                    if temp_dic[pair][idx] == 0:
                        temp_dic[pair].pop(idx)
                        if len(temp_dic[pair]) == 0:
                            temp_dic.pop(pair)
                # new
                for i in range(len(new_tokens) - 1):
                    pair = (new_tokens[i],new_tokens[i + 1])
                    temp_dic.setdefault(pair,{}).setdefault(idx,0)
                    temp_dic.setdefault(pair,{})[idx] += cnt
                new_key = tuple(new_tokens)
                idx_dic.pop(idx)
                idx_dic[idx] = idx_dic.get(idx,new_key)

        # print(self.dic)


    def get_vocab(self)->dict[int,bytes]:    
        return self.vocab

    def get_merges(self)->list[tuple[bytes,bytes]]:
        return self.merges

    #保存为可读的json文件
    @classmethod
    def serialize(cls, vocab:dict[int, bytes],
                  merges:list[tuple[bytes, bytes]],
                  vocab_filepath:str,
                  merges_filepath:str):
        print("👏👏👏 恭喜！开始序列化字典和合并规则到磁盘 请注意查收")
        if len(vocab) == 0 or len(merges) == 0:
            return None
        
        vocab_data = {str(token_id): byte_str.hex() for token_id, byte_str in vocab.items() }
        with open(vocab_filepath, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)

        merges_data = [[a.hex(), b.hex()] for a, b in merges]
        with open(merges_filepath, "w", encoding="utf-8") as f:
            json.dump(merges_data, f, indent=2, ensure_ascii=False)
    
    # 从json文件中读取
    @classmethod
    def deserialize(cls, vocab_filepath: str,
                    merges_filepath: str,
                    need_print = False
                    )->tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges_data = json.load(f)
        vocab = {int(k): bytes.fromhex(v) for k, v in vocab_data.items()}
        merges = [(bytes.fromhex(a), bytes.fromhex(b))for a, b in merges_data]
        if need_print:
            print(vocab)
            print(merges)
        return (vocab,merges)
        