from cs336_basics.bpe import MR_BPE
from utils import output_dir,current_dir

def train_bpe_tinystories():
    input_path =  f"{current_dir}/data/TinyStoriesV2-GPT4-train.txt"
    bpe = MR_BPE(input_path,10000,special_tokens = ["<|endoftext|>"])
    bpe.pre_process_text(10)
    bpe.train_bpe()
    # 序列化到磁盘
    bpe.serialize(bpe.vocab,bpe.merges,
        f"{output_dir}/tiny_stories_vocab.json",f"{output_dir}/tiny_stories_merges.json")
    

def train_bpe_expts_owt():
    input_path = "/home/humingrui/cs336/Assignment-1//data/owt_train.txt"
    bpe = MR_BPE(input_path,32000,special_tokens = ["<|endoftext|>"])
    bpe.pre_process_text()
    bpe.train_bpe()
    bpe.serialize(bpe.vocab,bpe.merges,
        f"{output_dir}/owt_vocab.json",f"{output_dir}/owt_merges.json")
    # bpe.deserialize(f"{output_dir}/owt_vocab.json",f"{output_dir}/owt_merges.json",need_print=False)

# def run_test_bpe():
#     special_tokens = ["<|endoftext|>", "<|startoftext|>"]
#     bpe = MR_BPE("/Users/hmr/Desktop/AI/cs336/my_test.txt",10000,special_tokens)
#     bpe.pre_process_text()
#     bpe.train_bpe()
#     print(bpe.get_vocab())
#     print(bpe.get_merges())
if __name__ == '__main__':
    # train_bpe_tinystories()
    train_bpe_expts_owt()