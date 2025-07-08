import torch
from transformers import AutoTokenizer
from models_transformer import Transformer


def main_test():
    # Load Qwen tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")

    # Define special tokens manually (if not set by tokenizer)
    special_tokens = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>"  # use eos as pad to avoid undefined pad_token
    }
    tokenizer.add_special_tokens(special_tokens)

    # Input example
    src_text = "纽约是一座城市"
    tgt_text = "New York is a city"

    # Tokenize input and target
    src_tokens = tokenizer.encode(src_text, return_tensors="pt")
    tgt_tokens = tokenizer.encode(tgt_text, return_tensors="pt")

    # Initialize model and resize embedding if needed
    vocab_size = len(tokenizer)
    model = Transformer(vocab_size)

    # Prepare decoder input: prepend <|im_start|> manually
    bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    tgt_input = torch.cat([torch.tensor([[bos_token_id]]), tgt_tokens[:, :-1]], dim=1)

    # Forward pass
    output = model(src_tokens, tgt_input)

    # Decode output
    print("Output shape:", output.shape)
    pred_ids = output.argmax(dim=-1)
    decoded = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
    print("Prediction:", decoded)


if __name__ == '__main__':
    main_test()
