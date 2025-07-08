import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Positional Encoding as described in "Attention is All You Need" paper.
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Positional Encoding as described in "Attention is All You Need" paper.
        :param d_model: Dimension of the model (embedding size).
        :param max_len:  Maximum length of the input sequence for which positional encodings are computed.
        """

        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)

        # Compute the division term, i.e., \frac{10000}{-2i * d_{model}}
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # (d_model/2,)

        # Fill even-numbered dimension with sine values
        pe[:, 0::2] = torch.sin(position * div_term)
        # Fill odd-numbered dimension with cosine values
        pe[:, 1::2] = torch.cos(position * div_term)

        # non-learnable buffer
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        """
        Forward pass to get positional encoding for the input tensor.
        :param x: input tensor of shape (batch_size, seq_len, d_model).
        :return: positional encoding tensor of shape (1, seq_len, d_model).
        """

        _, N, _ = x.shape
        pos_embed = self.pe[:, :N, :].to(x.device)

        return x + pos_embed


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
               Multi-Head Attention as described in "Attention is All You Need" paper.
               :param d_model: dimension of the model (embedding size).
               :param num_heads: number of attention heads.
        """

        super().__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """
               Forward pass through the multi-head attention block.
               :param x: input tensor of shape (batch_size, seq_len, d_model).
               :param mask: attention mask to prevent attending to certain positions (optional).
               :return: output tensor of shape (batch_size, seq_len, d_model).
        """

        B, T_q, _ = q.size()
        _, T_k, _ = k.size()
        _, T_v, _ = v.size()

        # Project and reshape: (B, T, C) -> (B, num_heads, T, d_k)
        q = self.q_linear(q).view(B, T_q, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(B, T_v, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)  # (B, num_heads, T_q, T_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)  # (B, num_heads, T_q, T_k)
        context = attn @ v  # (B, num_heads, T_q, d_k)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(B, T_q, self.num_heads * self.d_k)  # (B, T_q, d_model)

        return self.out(context)



# Position-wise Feedforward Network
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        """
        Position-wise Feedforward Network as described in "Attention is All You Need" paper.
        :param d_model: dimension of the model (embedding size).
        :param d_ff: dimension of the feedforward network, default value: 4*d_model.
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        Forward pass through the feedforward network.
        :param x: input tensor of shape (batch_size, seq_len, d_model).
        :return: output tensor of shape (batch_size, seq_len, d_model).
        """
        return self.linear2(F.relu(self.linear1(x)))

# Encoder Layer = MH-Attention + FeedForward + LayerNorm
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        """
        Encoder Layer as described in "Attention is All You Need" paper.
        :param d_model: dimension of the model (embedding size).
        :param num_heads: number of attention heads.
        :param d_ff: dimension of the feedforward network, default value: 4*d_model.
        """

        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x2 = self.norm1(x + self.attn(x, x, x, mask))
        return self.norm2(x2 + self.ff(x2))

# Decoder Layer = masked MH-Attn + cross-attn + FeedForward
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        """
        Decoder Layer as described in "Attention is All You Need" paper.
        :param d_model: dimension of the model (embedding size).
        :param num_heads:  number of attention heads.
        :param d_ff: dimension of the feedforward network, default value: 4*d_model.
        """

        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)

        self.ff = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        """
        Forward pass through the decoder layer.
        :param x: input tensor of shape (batch_size, tgt_seq_len, d_model).
        :param enc_out: input tensor from the encoder of shape (batch_size, src_seq_len, d_model).
        :param src_mask: source mask to prevent attending to certain positions in the encoder output.
        :param tgt_mask: target mask to prevent attending to future tokens in the decoder input.
        :return: output tensor of shape (batch_size, tgt_seq_len, d_model).
        """

        x2 = self.norm1(x + self.self_attn(x, x, x, tgt_mask))
        x3 = self.norm2(x2 + self.cross_attn(x2, enc_out, enc_out, src_mask))

        return self.norm3(x3 + self.ff(x3))

# Transformer architecture combining encoder and decoder
class Transformer(nn.Module):

    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048):
        """
        Transformer model as described in "Attention is All You Need" paper.
        :param vocab_size: vocabulary size for the input and output.
        :param d_model: dimension of the model (embedding size).
        :param num_heads: number of attention heads.
        :param num_layers: number of encoder/decoder layers.
        :param d_ff: dimension of the feedforward network, default value: 4*d_model.
        """

        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.pos_encoder = PositionalEncoding(d_model)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])

        self.fc_out = nn.Linear(d_model, vocab_size)  # Final projection to vocab

    def generate_square_subsequent_mask(self, sz):
        """
        Generate a square mask for the decoder to prevent attending to future tokens.
        :param sz: size of the mask (sequence length).
        :return: a lower triangular mask of shape (1, 1, sz, sz).
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.tril(torch.ones((sz, sz), device=device)).unsqueeze(0).unsqueeze(0)


    def forward(self, src, tgt):
        """
        Forward pass through the Transformer model.
        :param src: source input tensor of shape (batch_size, src_seq_len).
        :param tgt: target input tensor of shape (batch_size, tgt_seq_len).
        :return: token predictions of shape (batch_size, tgt_seq_len, vocab_size).
        """

        src_mask = None  # No masking on encoder side for now
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))

        src = self.pos_encoder(self.embedding(src))
        tgt = self.pos_encoder(self.embedding(tgt))

        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        for layer in self.decoder_layers:
            tgt = layer(tgt, src, src_mask, tgt_mask)

        return self.fc_out(tgt)