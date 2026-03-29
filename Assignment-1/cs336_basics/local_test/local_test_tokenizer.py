from cs336_basics.tokenizer import MR_Tokenizer
from cs336_basics.bpe import MR_BPE
from utils import current_dir,output_dir
import sys,os,struct

def tokenizer_experiments():
    input_file_path = f"{current_dir}/data/TinyStoriesV2-GPT4-train.txt"
    file_size = 0
    with open(input_file_path,'rb') as f:
        f.seek(0,os.SEEK_END)
        file_size = f.tell()
    
    vocab,merges = MR_BPE.deserialize(f"{current_dir}/output/tiny_stories_vocab.json", 
                               f"{current_dir}/output/tiny_stories_merges.json")
    tokenizer = MR_Tokenizer(vocab,merges,["<|endoftext|>"])
    token_id_list = tokenizer._multiple_encode(input_file_path)
    data = b''.join(struct.pack('I', x) for x in token_id_list)
    elemnts_size = len(data)

    print(f"🥸 文本转token id序列后所占字节大小:{elemnts_size}")
    print(f"🤔 编码后压缩比为{file_size/elemnts_size}")

if __name__ == "__main__":
    tokenizer_experiments()
    