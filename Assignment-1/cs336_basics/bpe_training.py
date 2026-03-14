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
        self.freq_dic: list[dict[bytes,int]] = []

    def pre_process_text(self):
        pattern:str = "|".join(map(re.escape,self.special_tokens))

        docs = re.split(pattern,self.text)
        split_text = [s.strip() for s in docs if s.strip()]

        for text in split_text:
            list_str = text.split()
            temp_dic:dict[bytes,int] = {}
            for s in list_str:
                if s in temp_dic:
                    temp_dic[s.encode()] += 1
                else:
                    temp_dic[s.encode()] = 1
            self.freq_dic.append(temp_dic)
        for dic in self.freq_dic:
            print(dic)
        return
        
    def train_bpe(self):
        print("-----Start to train bpe-----")



    def get_vocab(self)->dict[int,bytes]:    
        ret: dict[int,bytes] = {

        }

        return ret

    def get_merges(self)->list[tuple[bytes,bytes]]:
        ret:list[tuple[bytes,bytes]] = []

        return ret
        



def run_test_bpe():
    special_tokens = ["<|endoftext|>", "<|startoftext|>"]
    bpe = MR_BPE("/Users/hmr/Desktop/AI/cs336/my_test.txt",10000,special_tokens)
    bpe.pre_process_text()

if __name__ == '__main__':
    run_test_bpe()
