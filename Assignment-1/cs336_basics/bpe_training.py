import regex as re
# 
# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
class MR_BPE:

    def __init__(self,input_path:str,vocab_size:int,special_tokens:list[str]) -> None:
        print("BPE init")
        with open(input_path,"r",encoding = "utf-8") as f:
            self.text = f.read()
        self.special_tokens = special_tokens
        self.vocab_size = vocab_size
        self.freq_dic: list[dict[str,int]] = []
        self.dic:dict[tuple[bytes,...],int] = {}
        self.merges:list[tuple[bytes,bytes]] = []

    def pre_process_text(self):
        pattern:str = "|".join(map(re.escape,self.special_tokens))

        docs = re.split(pattern,self.text)
        split_text = [s.strip() for s in docs if s.strip()]

        for text in split_text:
            list_str = text.split()
            temp_dic:dict[str,int] = {}
            for s in list_str:
                if s in temp_dic:
                    temp_dic[s] += 1
                else:
                    temp_dic[s] = 1
            self.freq_dic.append(temp_dic)

        for temp_dic in self.freq_dic:
            for chars, count in temp_dic.items():
                key_tuple = tuple(c.encode("utf-8") for c in chars)
                self.dic[key_tuple] = self.dic.get(key_tuple, 0) + count
        print(self.dic)
        return
    
    def get_pair(self):
        return

    def find_max_freq(self,dic:dict[tuple[bytes,bytes],list[list[int]]])->tuple[bytes,bytes]:
        max_freq = -1
        merge_rule:tuple[bytes,bytes] = (b"",b"")
        for k,v in dic.items():
            sm = sum(j[2] for j in v)
            if sm > max_freq:
                max_freq = sm
                merge_rule = k
            elif sm == max_freq:
                # 频率相同时，选择字典序更大的键
                merge_rule = k if merge_rule < k else merge_rule
        self.merges.append(merge_rule)
        # print(f"max bi = {bi} and freq = {max_freq}")
        return merge_rule

        
    def train_bpe(self):
        # {low: 5, lower: 2, widest: 3, newest: 6}
        # {lo:7, ow:7, we:8, er:2, wi:3, id:3, de:3, es:9, st:9, ne:6, ew:6}
        # {(l,o,w):5, (l,o,w,e,r):2, (w,i,d,e,st):3, (n,e,w,e,st):6}
        print("-----Start to train bpe-----")
        while True:
            temp_dic:dict[tuple[bytes,bytes],list[list[int]]] = {}
            idx_dic:dict[int,tuple[bytes,...]] = {}
            idx = -1
            for key_tuple,v in self.dic.items():
                idx += 1
                if len(key_tuple) == 1:
                    continue
                for i in range(len(key_tuple) - 1):
                    next_i = i + 1
                    if next_i < len(key_tuple):
                        merge_rule = (key_tuple[i],key_tuple[next_i])
                        # 添加当前cur对应的位置和idx索引 idx对应到具体的key_tuple
                        temp_dic.setdefault(merge_rule,[]).append([i,idx,v])
                # idx <--> key_tuple
                idx_dic[idx] = idx_dic.get(idx,key_tuple)
            if len(temp_dic) == 0 or len(self.merges) > self.vocab_size:
                break
            # 拿到最大的合并规则
            bi = self.find_max_freq(temp_dic)
            posItem = temp_dic.get(bi,[])
            for pos_info in posItem:
                i,idx,_ = pos_info
                j = 0
                old_key = idx_dic[idx]
                list_key:list[bytes] = []
                while j < len(old_key):
                    cur = old_key[j]
                    if j == i and j + 1 < len(old_key):
                        cur = old_key[j] + old_key[j + 1]
                        j += 2
                    else:
                        j += 1
                    list_key.append(cur)
                new_key = tuple(list_key)
                v = self.dic.pop(old_key)
                self.dic[new_key] = v

        print(self.dic)


    def get_vocab(self)->dict[int,bytes]:    
        ret: dict[int,bytes] = {

        }

        return ret

    def get_merges(self)->list[tuple[bytes,bytes]]:
        print(self.merges)
        return self.merges
        



def run_test_bpe():
    special_tokens = ["<|endoftext|>", "<|startoftext|>"]
    bpe = MR_BPE("/Users/hmr/Desktop/AI/cs336/my_test.txt",10000,special_tokens)
    bpe.pre_process_text()
    bpe.train_bpe()
    bpe.get_merges()

if __name__ == '__main__':
    run_test_bpe()
