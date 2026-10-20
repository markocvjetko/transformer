"""
TODO -
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

"""
generate(): no eval(), no trimming to seq_len, top_p ignored, no stop at EOS, breaks on right padding.
Config: vocab_size is 48008 at the top level but 48006 in model.
**kwargs: silently swallows typos in config keys.
Missing checks: n_heads % gqa_factor, even head_dim, dq == dk == dv.
RoPE in bf16: the tables get cast if the model is cast (mild); the angles break if the default dtype is bf16 (severe). Fix: compute in fp32 and keep the tables fp32.
SDPA with left padding: rows with nothing to attend to may give NaN, depending on backend and version.
Docstrings and comments: several shapes and descriptions are outdated.
"""


class SwishGatedLinearUnit(nn.Module):
    def __init__(self, d_model, d_ff, bias=False):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        gate_output = F.silu(self.w_gate(x)) * self.w_up(x)
        return self.w_down(gate_output)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, base, head_dim, seq_len):
        super().__init__()
        self.base = base
        self.head_dim = head_dim
        self.seq_len = seq_len

        token_pos = torch.arange(seq_len, dtype=torch.float32)
        d_pairs = torch.arange(0, head_dim // 2, dtype=torch.float32)
        thetas = torch.pow(base, -2 * d_pairs / head_dim)

        cos_matrix = torch.cos(torch.outer(token_pos, thetas))
        cos_matrix = cos_matrix[None, :, None, :]
        sin_matrix = torch.sin(torch.outer(token_pos, thetas))
        sin_matrix = sin_matrix[None, :, None, :]
        self.register_buffer("cos_matrix", cos_matrix, persistent=False)
        self.register_buffer("sin_matrix", sin_matrix, persistent=False)

    def forward(self, x):
        """
        Args:
            x: should be B, some_len, emb_dim
        """

        cos = self.cos_matrix[:, : x.shape[1]]
        sin = self.sin_matrix[:, : x.shape[1]]

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        rot_even = cos * x_even - sin * x_odd
        rot_odd = sin * x_even + cos * x_odd

        out = torch.empty_like(x)
        out[..., ::2] = rot_even
        out[..., 1::2] = rot_odd
        return out


class TorchSelfAttention(nn.Module):
    """
    Scaled dot-product multi-head attention.
    """

    def __init__(
        self,
        d_input,
        dq,
        dk,
        dv,
        do,
        p_dropout,
        n_heads,
        use_gqa=False,
        gqa_factor=4,
        rope=None,
        use_bias=False,
    ):
        super().__init__()
        self.d_input = d_input
        self.dq = dq
        self.dk = dk  # these dims are placeholders
        self.dv = dv  # to check if all share the same dim
        self.do = do
        self.p_dropout = p_dropout
        self.n_heads = n_heads
        self.use_gqa = use_gqa
        self.gqa_factor = gqa_factor
        self.rope = rope
        self.use_bias = use_bias

        self.W_Q = nn.Linear(d_input, dq, bias=use_bias)
        if use_gqa:
            self.W_K = nn.Linear(d_input, dk // gqa_factor, bias=use_bias)
            self.W_V = nn.Linear(d_input, dv // gqa_factor, bias=use_bias)
        else:
            self.W_K = nn.Linear(d_input, dk, bias=use_bias)
            self.W_V = nn.Linear(d_input, dv, bias=use_bias)
        self.W_O = nn.Linear(dv, do, bias=use_bias)  # Projects concatenated heads (dv) to output (do)

    def forward(self, Q, K, V, causal_mask=None, padding_mask=None):

        Q = self.W_Q(Q).view(Q.shape[0], Q.shape[1], self.n_heads, self.dq // self.n_heads)
        if self.use_gqa:
            K = self.W_K(K).view(K.shape[0], K.shape[1], self.n_heads // self.gqa_factor, self.dk // self.n_heads)
            V = self.W_V(V).view(V.shape[0], V.shape[1], self.n_heads // self.gqa_factor, self.dv // self.n_heads)
        else:
            K = self.W_K(K).view(K.shape[0], K.shape[1], self.n_heads, self.dk // self.n_heads)
            V = self.W_V(V).view(V.shape[0], V.shape[1], self.n_heads, self.dv // self.n_heads)

        if self.rope is not None:
            Q = self.rope(Q)
            K = self.rope(K)

        is_causal = causal_mask is not None and padding_mask is None
        attn_mask = None
        if padding_mask is not None:
            attn_mask = ~padding_mask[:, None, None, :]
            if causal_mask is not None:
                attn_mask = attn_mask & ~causal_mask[: Q.shape[1], : Q.shape[1]]

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Apply causal mask (for decoder self-attention)
        dropout_p = self.p_dropout if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, enable_gqa=self.use_gqa
        )

        # Merge the 'n_heads' and 'head_dim' dimensions back into a single dimension after transposing
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(attn_output.size(0), attn_output.size(1), -1)

        # Apply output projection
        return self.W_O(attn_output)


class MySelfAttention(nn.Module):
    """
    Scaled dot-product multi-head attention.
    """

    def __init__(
        self,
        d_input,
        dq,
        dk,
        dv,
        do,
        p_dropout,
        n_heads,
        use_gqa=False,
        gqa_factor=4,
        rope=None,
        use_bias=False,
    ):
        super().__init__()
        self.d_input = d_input
        self.dq = dq
        self.dk = dk  # these dims are placeholders
        self.dv = dv  # to check if all share the same dim
        self.do = do
        self.dropout = nn.Dropout(p=p_dropout)
        self.n_heads = n_heads
        self.use_gqa = use_gqa
        self.gqa_factor = gqa_factor
        self.rope = rope
        self.use_bias = use_bias

        self.W_Q = nn.Linear(d_input, dq, bias=use_bias)
        if use_gqa:
            self.W_K = nn.Linear(d_input, dk // gqa_factor, bias=use_bias)
            self.W_V = nn.Linear(d_input, dv // gqa_factor, bias=use_bias)
        else:
            self.W_K = nn.Linear(d_input, dk, bias=use_bias)
            self.W_V = nn.Linear(d_input, dv, bias=use_bias)

        self.W_O = nn.Linear(dv, do, bias=use_bias)  # Projects concatenated heads (dv) to output (do)

    def forward(self, Q, K, V, causal_mask=None, padding_mask=None):

        Q = self.W_Q(Q).view(Q.shape[0], Q.shape[1], self.n_heads, self.dq // self.n_heads)
        if self.use_gqa:
            K = self.W_K(K).view(K.shape[0], K.shape[1], self.n_heads // self.gqa_factor, self.dk // self.n_heads)
            V = self.W_V(V).view(V.shape[0], V.shape[1], self.n_heads // self.gqa_factor, self.dv // self.n_heads)
        else:
            K = self.W_K(K).view(K.shape[0], K.shape[1], self.n_heads, self.dk // self.n_heads)
            V = self.W_V(V).view(V.shape[0], V.shape[1], self.n_heads, self.dv // self.n_heads)

        if self.rope is not None:
            Q = self.rope(Q)
            K = self.rope(K)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        if self.use_gqa:
            K = K.repeat_interleave(self.gqa_factor, dim=1)
            V = V.repeat_interleave(self.gqa_factor, dim=1)

        attn_score = torch.matmul(Q, K.transpose(-2, -1))
        attn_score /= math.sqrt(Q.shape[-1])

        L = attn_score.shape[-1]
        mask = None
        if causal_mask is not None:
            mask = causal_mask[:L, :L]  # (L, L)
        if padding_mask is not None:
            pad = padding_mask[:, None, None, :]  # (B, 1, 1, L)
            mask = pad if mask is None else mask | pad

        if mask is not None:
            attn_score = attn_score.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(attn_score, dim=-1)
        if mask is not None:
            # A query that cannot see any key (e.g. left pad token, gets Nans).
            # Line below sets the weights to 0, keeping the gradients clean
            attn_weights = attn_weights.masked_fill(mask.all(dim=-1, keepdim=True), 0.0)

        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)

        # Merge the 'n_heads' and 'head_dim' dimensions back into a single dimension after transposing
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(attn_output.size(0), attn_output.size(1), -1)

        # Apply output projection
        return self.W_O(attn_output)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        p_dropout,
        n_heads,
        use_gqa=False,
        gqa_factor=4,
        rope=None,
        norm_type="rms",  # layer or rms
        ff_type="swiglu",  # swiglu or gelu
        use_bias=False,
        use_torch_sdpa=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads

        if norm_type == "rms":
            self.norm_mha = nn.RMSNorm(d_model)
            self.norm_ff = nn.RMSNorm(d_model)
        elif norm_type == "layer":
            self.norm_mha = nn.LayerNorm(d_model)
            self.norm_ff = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        attn_cls = TorchSelfAttention if use_torch_sdpa else MySelfAttention
        self.mha = attn_cls(
            d_input=d_model,
            dq=d_model,
            dk=d_model,
            dv=d_model,
            do=d_model,
            p_dropout=p_dropout,
            n_heads=n_heads,
            use_gqa=use_gqa,
            gqa_factor=gqa_factor,
            rope=rope,
            use_bias=use_bias,
        )

        self.dropout = nn.Dropout(p=p_dropout)

        if ff_type == "gelu":
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff, bias=use_bias),
                nn.GELU(approximate="tanh"),
                nn.Linear(d_ff, d_model, bias=use_bias),
            )  # applied to each token individually
        elif ff_type == "swiglu":
            self.ff = SwishGatedLinearUnit(d_model, d_ff, bias=use_bias)
        else:
            raise ValueError(f"Unknown ff_type: {ff_type}")

    def forward(self, x, causal_mask=None, padding_mask=None):
        x_ln = self.norm_mha(x)
        x = x + self.dropout(self.mha(x_ln, x_ln, x_ln, causal_mask, padding_mask))
        x = x + self.dropout(self.ff(self.norm_ff(x)))
        return x


class DecoderLM(nn.Module):
    """
    Transformer model for sequence to sequence tasks.
    Args:
        vocab_size: int, the size of the vocabulary
        seq_len: int, the length of the input sequence
        d_model: int, the dimension of the model
        d_ff: int, the dimension of the feedforward layer
        n_heads: int, the number of heads in the multi-head attention
        N: int, the number of encoder and decoder blocks
    """

    def __init__(
        self,
        vocab_size=50257,
        seq_len=1024,
        d_model=1024,
        d_ff=4096,
        n_heads=16,
        use_gqa=False,
        gqa_factor=2,
        N=24,
        pad_token_id=0,
        eos_token_id=2,
        p_dropout=0.1,
        pos_emb_type="rope",  # rope or learned
        rope_base=10_000,
        norm_type="rms",  # layer or rms
        ff_type="swiglu",  # gelu or swiglu
        use_bias=False,
        use_torch_sdpa=False,
        **kwargs,
    ):
        super().__init__()

        if ff_type not in {"swiglu", "gelu"}:
            raise ValueError("ff_type must be swiglu or gelu")
        if d_model % n_heads != 0:
            raise ValueError("n_heads must divide d_model")

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.N = N
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.p_dropout = p_dropout
        self.rope_base = rope_base
        self.norm_type = norm_type
        self.ff_type = ff_type

        self.embedding_layer = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(p_dropout)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(
                    (seq_len, seq_len),
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
            persistent=False,
        )
        if pos_emb_type == "rope":
            rope = RotaryPositionalEmbedding(base=rope_base, head_dim=d_model // n_heads, seq_len=seq_len)
            self.embedding_pos = None
        elif pos_emb_type == "learned":
            rope = None
            self.embedding_pos = nn.Embedding(seq_len, d_model)
        else:
            raise ValueError(f"Unknown pos_emb_type: {pos_emb_type}")

        self.decoders = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    p_dropout=p_dropout,
                    n_heads=n_heads,
                    use_gqa=use_gqa,
                    gqa_factor=gqa_factor,
                    rope=rope,
                    norm_type=norm_type,
                    ff_type=ff_type,
                    use_bias=use_bias,
                    use_torch_sdpa=use_torch_sdpa,
                )
                for _ in range(N)
            ]
        )

        if norm_type == "rms":
            self.norm_final = nn.RMSNorm(d_model)
        elif norm_type == "layer":
            self.norm_final = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        self.ff_output = nn.Linear(d_model, vocab_size, bias=False)

        # the output layer and the embedding layer share weights.
        self.ff_output.weight = self.embedding_layer.weight

        self.apply(self._init_weights)

        for name, params in self.named_parameters():
            if any(k in name for k in ("ff.2.weight", "w_down.weight", "mha.W_O.weight")):
                nn.init.normal_(params, mean=0.0, std=0.02 / math.sqrt(2 * N))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x, padding_mask=None):
        """
        Args:
            x: torch.tensor, shape (B, N, vocab_size)
            y: torch.tensor, shape (B, M, vocab_size)
            Where M, N are sequence lengths.
        Return:
            Tensor of shape (B, M, vocab_size)
        """

        x = self.embedding_layer(x)
        if self.embedding_pos is not None:
            pos = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
            x = x + self.embedding_pos(pos)

        decoder_output = self.dropout(x)
        for decoder in self.decoders:
            decoder_output = decoder(decoder_output, self.causal_mask, padding_mask)

        return self.ff_output(self.norm_final(decoder_output))

    def generate(self, x: torch.Tensor, temperature=1.0, seq_len=256, top_k=50, top_p=1.0):
        # TODO: ADD top_p SUPPORT, place model in EVAL?
        with torch.no_grad():
            while x.shape[-1] < seq_len:
                logits = self(x, padding_mask=x == self.pad_token_id)[:, -1, :]
                logits /= temperature
                probs = torch.softmax(logits, dim=-1)
                top_k_vals, top_k_idx = torch.topk(probs, top_k, dim=-1)
                cumsum_masked = top_k_vals
                samples = torch.multinomial(cumsum_masked, num_samples=1)

                x = torch.cat((x, torch.gather(top_k_idx, dim=1, index=samples)), dim=-1)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer = DecoderLM(
        vocab_size=48006,
        d_model=512,
        d_ff=1408,
        n_heads=8,
        N=30,
        seq_len=512,
    ).to(device)
    num_params = sum(p.numel() for p in transformer.parameters())
    print(f"Number of parameters: {num_params}")

    batch_size = 2
    seq_len = 2
    x = torch.randint(low=1, high=16, size=(batch_size, seq_len)).to(device)
    output = transformer(x)
    output = transformer.generate(x, seq_len=4, temperature=1, top_p=0.5)
