import regex as re
import os
# 
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
class MR_BPE:

    def __init__(self,input_path:str | os.PathLike,vocab_size:int,special_tokens:list[str]) -> None:
        print("--------BPE init-------")
        # 路径对象转为str
        input_path = str(input_path) 
        with open(input_path,"r",encoding = "utf-8") as f:
            self.text = f.read()
        self.special_tokens = special_tokens
        self.vocab_size = vocab_size
        self.dic:dict[tuple[bytes,...],int] = {}
        self.merges:list[tuple[bytes,bytes]] = []
        self.vocab:dict[int,bytes] = {}
        self.vocab_cur_size = 0

        # 初始化初始词汇表
        for idx,sp_token in enumerate(special_tokens):
            self.vocab[idx] = self.vocab.get(idx,sp_token.encode("utf-8"))
        self.vocab_cur_size = len(self.vocab)
        # 加入0～255
        for i in range(256):
            self.vocab[self.vocab_cur_size] = self.vocab.get(self.vocab_cur_size,bytes([i]))
            self.vocab_cur_size += 1

    def pre_process_text(self):
        pattern:str = "|".join(map(re.escape,self.special_tokens))

        # 先按特殊token分为不同的text段落 一个段落有多个单词 分别以空格划分
        docs = re.split(pattern,self.text)
        word_counts:dict[tuple[bytes,...],int] = {}
        # 获取以bytes格式的词元频率
        for part in docs:
            tokens = re.findall(PAT, part)
            # tokens = part.split()
            for token in tokens:
                text_bytes = token.encode("utf-8")
                key = tuple(bytes([b]) for b in text_bytes)
                word_counts[key] = word_counts.get(key, 0) + 1
                
        self.dic = word_counts
        return
    
    def get_pair(self):
        return

    def find_max_freq(self,dic:dict[tuple[bytes,bytes],list[list[int]]])->tuple[bytes,bytes]:
        max_freq = -1
        merge_rule:tuple[bytes,bytes] = (b"",b"")
        for k,v in dic.items():
            sm = sum(j[1] for j in v)
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
        # {low: 5, lower: 2, widest: 3, newest: 6}
        # {['l','o','w']:5,['l','o','w','r']:2,......}
        # {lo:7, ow:7, we:8, er:2, wi:3, id:3, de:3, es:9, st:9, ne:6, ew:6}
        # {(l,o,w):5, (l,o,w,e,r):2, (w,i,d,e,st):3, (n,e,w,e,st):6}
        print("-----Start to train bpe-----")
        while True:
            temp_dic:dict[tuple[bytes,bytes],list[list[int]]] = {}
            idx_dic:dict[int,tuple[bytes,...]] = {}
            idx = -1
            for key_tuple,freq in self.dic.items():
                idx += 1
                if len(key_tuple) == 1:
                    continue
                for i in range(len(key_tuple) - 1):
                    next_i = i + 1
                    merge_rule = (key_tuple[i],key_tuple[next_i])
                    # 添加当前cur对应的位置和idx索引 idx对应到具体的key_tuple
                    temp_dic.setdefault(merge_rule,[]).append([idx,freq])
                # idx <--> key_tuple idx唯一映射到key_tuple 方便后续根据idx值找对应的key_tuple
                idx_dic[idx] = idx_dic.get(idx,key_tuple)
        
            # 停止条件
            if len(temp_dic) == 0 or self.vocab_cur_size >= self.vocab_size:
                break
            # 拿到最大的合并规则
            bi = self.find_max_freq(temp_dic)
            # 拿[idx,freq]的list
            posItem = temp_dic.get(bi,[])
            for pos_info in posItem:
                # 可能会有多个相同的idx
                idx,_ = pos_info
                j = 0
                tokens = idx_dic[idx]
                new_tokens:list[bytes] = []
                while j < len(tokens):
                    if j + 1 < len(tokens) and tokens[j] == bi[0] and tokens[j + 1] == bi[1]:
                        new_tokens.append(bi[0] + bi[1])
                        j += 2
                    else:
                        new_tokens.append(tokens[j])
                        j += 1
                new_key = tuple(new_tokens)
                v = self.dic.get(tokens,None)
                if v is None:
                    continue
                self.dic.pop(tokens)
                self.dic[new_key] = v
        # print(self.dic)


    def get_vocab(self)->dict[int,bytes]:    
        return self.vocab

    def get_merges(self)->list[tuple[bytes,bytes]]:
        return self.merges
        



# def run_test_bpe():
#     special_tokens = ["<|endoftext|>", "<|startoftext|>"]
#     bpe = MR_BPE("/Users/hmr/Desktop/AI/cs336/my_test.txt",10000,special_tokens)
#     bpe.pre_process_text()
#     bpe.train_bpe()
#     print(bpe.get_vocab())
#     print(bpe.get_merges())

# if __name__ == '__main__':
#     run_test_bpe()
