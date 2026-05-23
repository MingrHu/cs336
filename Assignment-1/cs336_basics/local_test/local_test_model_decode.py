import torch
import torch.nn as nn
from cs336_basics.local_test.local_test_train_model import load_model_and_weights
from cs336_basics import helper,tokenizer,bpe
from utils import output_dir,current_dir

def model_decode():
    checkpoint_path = f"{output_dir}/checkpoint_v0.pt"
    model,_ = load_model_and_weights(checkpoint_path)
    model.eval()

    vocab,merges = bpe.MR_BPE.deserialize(f"{current_dir}/output/tiny_stories_vocab.json", 
                               f"{current_dir}/output/tiny_stories_merges.json")
    endof = "<|endoftext|>"
    tk = tokenizer.MR_Tokenizer(vocab,merges,[endof])
    endof_id = tk.dic_token_id[endof.encode("utf-8")]

    while True:
        prompt = input("请输入 prompt: ")
        if prompt.lower() == "exit":
            break

        token_ids = tk.encode(prompt)
        token_ids = torch.tensor(token_ids, dtype=torch.long, device="cpu").unsqueeze(0)

        with torch.no_grad():
            for token_id in helper.lm_decode_stream(model, token_ids,50,endof_id,'cpu',tmp = 0.7,top_p = 0.9):
                print(tk.decode([token_id]), end="", flush=True)

if __name__ == "__main__":
    model_decode()