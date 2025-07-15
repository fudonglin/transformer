import torch
from torch import nn
from transformers import AutoTokenizer
from models_transformer import Transformer

@torch.no_grad()
def greedy_decode(model, tokenizer, src_tokens, max_len=50):
    """
    Autoregressive greedy decoding from Transformer model.
    :param model: Trained Transformer model.
    :param tokenizer: Hugging Face tokenizer.
    :param src_tokens: Tokenized source input (1, src_seq_len).
    :param max_len: Maximum length of generated sequence.
    :return: Generated target token IDs.
    """

    model.eval()
    device = src_tokens.device
    src = src_tokens.to(device)

    # Encode source
    src_embed = model.pos_encoder(model.embedding(src))
    for layer in model.encoder_layers:
        src_embed = layer(src_embed, mask=None)

    # Start with BOS token
    bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    generated = torch.tensor([[bos_token_id]], device=device)

    for _ in range(max_len):
        tgt_embed = model.pos_encoder(model.embedding(generated))
        tgt_mask = model.generate_square_subsequent_mask(generated.size(1)).to(device)

        # Decode step-by-step
        tgt_output = tgt_embed
        for layer in model.decoder_layers:
            tgt_output = layer(tgt_output, src_embed, src_mask=None, tgt_mask=tgt_mask)

        logits = model.fc_out(tgt_output[:, -1:, :])  # (1, 1, vocab_size)
        next_token = logits.argmax(dim=-1)  # (1, 1)
        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return generated


def main_inference():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
    special_tokens = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>"
    }
    tokenizer.add_special_tokens(special_tokens)

    # Input (Chinese)
    src_text = "纽约是一座城市"
    src_tokens = tokenizer.encode(src_text, return_tensors="pt")

    # Initialize model
    vocab_size = len(tokenizer)
    model = Transformer(vocab_size)
    model.embedding = nn.Embedding(vocab_size, model.embedding.embedding_dim)  # ensure match if resized
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Inference
    output_ids = greedy_decode(model, tokenizer, src_tokens)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Source:", src_text)
    print("Prediction:", output_text)


if __name__ == '__main__':
    main_inference()
