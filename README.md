# Implement Transformers from Scratch

This repository offers an **unofficial PyTorch implementation of the Transformer architecture** from scratch. If you're new to Transformers or want a refresher, I highly recommend reading the original paper, [Attention Is All You Need](https://arxiv.org/pdf/1706.03762), as well as my previous [blog post](https://fudonglin.github.io/2025/07/04/transformer.html), which walks through each component step-by-step.



## Overview

![Transformer](https://github.com/fudonglin/transformer/blob/main/images/transformers.png?raw=t)

Figure 1: Transformer model architecture.



This repository implements the Transformer model for machine translation, following the original design. As an example task, we translate the Chinese sentence `"纽约是一座城市"` into its English equivalent: `"New York is a city"`.

Figure 1 highlights the three main building blocks of the model: **Positional Encoding**, **Encoder**, and **Decoder**:

- **Positional Encoding** adds order-awareness to the otherwise order-agnostic attention mechanism.
- The **Encoder** contains **Multi-Head Self-Attention** and **Feedforward Networks**, learning contextual representations from the source language.
- The **Decoder** comprises:
  1. **Masked Multi-Head Self-Attention** to enforce autoregressive decoding,
  2. **Cross-Attention** to align the source and target sequences,
  3. A **Feedforward Network** to apply non-linear transformations and enhance representational power.



## Positional Encoding

Transformers are inherently **order-agnostic** because self-attention treats all tokens equally, regardless of position. To introduce sequence order, the authors use Positional Encoding, which encodes both relative and absolute position information using sinusoidal functions.

The original paper uses fixed, non-learnable encodings:

```math
\begin{gathered}
PE(pos, 2i) = \sin(\frac{pos}{10000^{2i/d_{\textrm{model}}}}), \\
PE(pos, 2i+1) = \cos(\frac{pos}{10000^{2i/d_{\textrm{model}}}}).
\end{gathered}
```

Here, $pos$ denotes the position of a word (or token) in the sequence, $d\_{\textrm{model}}$ is the dimensionality of the model's embeddings (e.g., $d_{\textrm{model}} = 512$ used in the paper), and $i \in \{0, 1, 2, \dots, d_{\textrm{model}} - 1 \}$ indexes the embedding dimensions. 



```python
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

        # Non-learnable buffer
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        """
        Forward pass to get positional encoding for the input tensor.
        :param x: input tensor of shape (batch_size, seq_len, d_model).
        :return: sum of input and positional encoding  tensors, with its shape (1, seq_len, d_model).
        """

        _, N, _ = x.shape
        pos_embed = self.pe[:, :N, :].to(x.device)

        return x + pos_embed
```



## Multi-Head Attention

![MHA](https://github.com/fudonglin/transformer/blob/main/images/attention.png?raw=true)

**Multi-Head Attention (MHA)** is designed to help the model capture different types of relationships between tokens (e.g., words) in a sequence. It enables the model to attend to information from multiple representation subspaces simultaneously:

```math
\begin{gathered}
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}^O, \\
\text{head}_i = \text{Attention}(\mathbf{Q} \mathbf{W}_i^Q, \mathbf{K} \mathbf{W}_i^K, \mathbf{V} \mathbf{W}_i^V), \\
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}.
\end{gathered}
```

Here, $\mathbf{Q}$, $\mathbf{K}$, and $\mathbf{V}$ are the query, key, and value in the Attention. $\mathbf{W}^{Q}, \mathbf{W}^{K}, \mathbf{W}^{V}$,  and $\mathbf{W}^{O}$ are four learnable matrices.  $i$ denote the $i$-th attention head. $\sqrt{d_{k}}$  is a scale factor that helps produce a smoother distribution of attention weights.



```python
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
            :param q: query tensor of shape (batch_size, seq_len, d_model).
            :param k: key tensor of shape (batch_size, seq_len, d_model).
            :param v: value tensor of shape (batch_size, seq_len, d_model).
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
```





## Position-wise Feedforward Network

The **Position-wise Feedforward Network (FFN)** is applied to each token independently and identically. Unlike the attention mechanism, which enables interaction between tokens, the FFN introduces **non-linearity** and **depth**, allowing the model to learn more abstract features.

> 🔍 The non-linearity — typically ReLU or GELU — is crucial; without it, the FFN would collapse into a single linear transformation, reducing its ability to capture complex patterns.

```math
\text{FFN}(\mathbf{x}) = \mathbf{W}_2 \cdot \phi(\mathbf{W}_1 \cdot \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2.
```

Here, $\mathbf{x}$ is the input token representation at a given position. $\mathbf{W}_{i}$ and $\mathbf{b}_{i}$ are the weights and bias at the $i$-th linear layer. $\phi ( \cdot )$ is the non-linear activation function, typically ReLU or GELU.



```python
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
```



## Encoder

The **Encoder** learns a contextual representation of the input sequence. Each encoder layer consists of:

- Multi-Head Self-Attention
- Position-wise Feedforward Network
- Residual Connections and Layer Normalization



```python
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
```



## Decoder

The **Decoder** generates the target sequence, one token at a time, using both past predictions and encoded source representations. Each decoder layer contains:

1. Masked Multi-Head Self-Attention (to prevent access to future tokens),
2. Cross-Attention (to attend to the encoder output),
3. Feedforward Network (for non-linear transformation).



```python
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
```



## Transformer

Finally, the **Transformer** class ties everything together — embedding, positional encoding, stacked encoder and decoder layers, and the final linear projection to the output vocabulary.



```python
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
```

