import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Scaled dot-product multi-head attention.
    """

    def __init__(
        self,
        d_input,
        n_heads,
        dq,
        dk,
        dv,
        do,
        p_dropout,
        rope=None,
        use_bias=False,
        use_gqa=False,
        gqa_factor=2,
    ):
        super().__init__()
        self.d_input = d_input
        self.n_heads = n_heads
        self.dq = dq
        self.dk = dk  # these dims are placeholders
        self.dv = dv  # to check if all share the same dim
        self.do = do
        self.dropout = nn.Dropout(p_dropout)
        self.rope = rope
        self.use_bias = use_bias
        self.use_gqa = use_gqa
        self.gqa_factor = gqa_factor
        if use_gqa:
            self.W_Q = nn.Linear(d_input, dq, bias=use_bias)
            self.W_K = nn.Linear(d_input, dk // self.gqa_factor, bias=use_bias)
            self.W_V = nn.Linear(d_input, dv // self.gqa_factor, bias=use_bias)
        else:
            self.W_Q = nn.Linear(d_input, dq, bias=use_bias)
            self.W_K = nn.Linear(d_input, dk, bias=use_bias)
            self.W_V = nn.Linear(d_input, dv, bias=use_bias)
        self.W_O = nn.Linear(dv, do, bias=use_bias)  # Projects concatenated heads (dv) to output (do)

    def forward(self, Q, K, V, causal_mask=None, padding_mask=None):

        if self.use_gqa:
            gqa_heads = self.n_heads // self.gqa_factor
            Q = self.W_Q(Q).view(Q.shape[0], Q.shape[1], self.n_heads, self.dq // self.n_heads)
            K = self.W_K(K).view(K.shape[0], K.shape[1], gqa_heads, self.dk // self.n_heads)
            V = self.W_V(V).view(V.shape[0], V.shape[1], gqa_heads, self.dv // self.n_heads)

            K = K.repeat_interleave(self.gqa_factor, dim=2)
            V = V.repeat_interleave(self.gqa_factor, dim=2)
        else:
            Q = self.W_Q(Q).view(Q.shape[0], Q.shape[1], self.n_heads, self.dq // self.n_heads)
            K = self.W_K(K).view(K.shape[0], K.shape[1], self.n_heads, self.dk // self.n_heads)
            V = self.W_V(V).view(V.shape[0], V.shape[1], self.n_heads, self.dv // self.n_heads)
        if self.rope is not None:
            Q = self.rope(Q)
            K = self.rope(K)

        print("First 5 Q values:", Q.flatten()[:5])
        print("First 5 K values:", K.flatten()[:5])
        print("First 5 V values:", V.flatten()[:5])

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        attn_score = torch.matmul(Q, K.transpose(-2, -1))
        attn_score /= math.sqrt(Q.shape[-1])

        # Apply causal mask (for decoder self-attention)
        if causal_mask is not None:
            attn_score = attn_score.masked_fill(
                causal_mask[: attn_score.shape[-1], : attn_score.shape[-1]], float("-inf")
            )

        if padding_mask is not None:
            attn_score = attn_score.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn_weights = torch.softmax(attn_score, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)

        # Merge the 'n_heads' and 'head_dim' dimensions back into a single dimension after transposing
        attn_output = attn_output.transpose(1, 2).contiguous()
        print("attn_output as text:", attn_output[-1, 1, -1, :9])
        attn_output = attn_output.view(attn_output.size(0), attn_output.size(1), -1)

        # Apply output projection
        return self.W_O(attn_output)


class SelfAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads=8,
        gqa_factor=2,
        p_dropout=0.0,
        is_causal=True,
        use_bias=True,
        enable_gqa=True,
    ):
        super().__init__()

        self.p_dropout = p_dropout
        self.n_heads = n_heads
        self.gqa_factor = gqa_factor
        self.enable_gqa = enable_gqa
        self.is_causal = is_causal

        self.W_q = nn.Linear(d_model, d_model, bias=use_bias)

        self.W_k = nn.Linear(d_model, d_model // gqa_factor, bias=use_bias)
        self.W_v = nn.Linear(d_model, d_model // gqa_factor, bias=use_bias)

        self.W_o = nn.Linear(d_model, d_model, bias=use_bias)

        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        Q = self.W_q(x).view(x.shape[0], x.shape[1], self.n_heads, x.shape[2] // self.n_heads)
        K = self.W_k(x).view(x.shape[0], x.shape[1], self.n_heads // self.gqa_factor, x.shape[2] // self.n_heads)
        V = self.W_v(x).view(x.shape[0], x.shape[1], self.n_heads // self.gqa_factor, x.shape[2] // self.n_heads)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        output = F.scaled_dot_product_attention(Q, K, V, enable_gqa=self.enable_gqa, is_causal=self.is_causal)
        output = output.transpose(1, 2).contiguous().view(output.shape[0], output.shape[2], -1)
        return self.W_o(output)


# Set shared parameters
B = 2  # batch size
T = 8  # sequence length
d_model = 512  # embedding size
n_heads = 16  # number of heads
gqa_factor = 4
enable_gqa = True
use_bias = False
is_causal = True

# Make dummy input
x = torch.randn(B, T, d_model)

# ----- Make parameter initialization identical -----
# Torch manual seed for full determinism, if wanted for demonstration
seed = 42
torch.manual_seed(seed)

# Dummy modules to copy params
mha = MultiHeadAttention(
    d_input=d_model,
    n_heads=n_heads,
    dq=d_model,
    dk=d_model,
    dv=d_model,
    do=d_model,
    p_dropout=0.0,  # no dropout for matching
    rope=None,
    use_bias=use_bias,
    use_gqa=enable_gqa,
    gqa_factor=gqa_factor,
)
self_attn = SelfAttention(
    d_model=d_model,
    n_heads=n_heads,
    gqa_factor=gqa_factor,
    use_bias=use_bias,
    enable_gqa=enable_gqa,
    is_causal=is_causal,
)

# Copy all relevant parameters from mha to self_attn (q/k/v/o)
with torch.no_grad():
    # Copy W_Q -> W_q, W_K -> W_k, W_V -> W_v, W_O -> W_o
    self_attn.W_q.weight.copy_(mha.W_Q.weight)
    self_attn.W_k.weight.copy_(mha.W_K.weight)
    self_attn.W_v.weight.copy_(mha.W_V.weight)
    self_attn.W_o.weight.copy_(mha.W_O.weight)
    if use_bias:
        self_attn.W_q.bias.copy_(mha.W_Q.bias)
        self_attn.W_k.bias.copy_(mha.W_K.bias)
        self_attn.W_v.bias.copy_(mha.W_V.bias)
        self_attn.W_o.bias.copy_(mha.W_O.bias)
# ----- End identical params section -----

print("MultiHeadAttention parameters:")
for name, param in mha.named_parameters():
    print(name, param.shape)
print("\nSelfAttention parameters:")
for name, param in self_attn.named_parameters():
    print(name, param.shape)

# Example forward pass, to verify shape compatibility
# Build a causal mask of shape (T, T) where T is sequence length\
if is_causal:
    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
else:
    causal_mask = None
with torch.no_grad():
    out_mha = mha(x, x, x, causal_mask=causal_mask)
    out_selfattn = self_attn(x)
    print("\nMultiHeadAttention output shape:", out_mha.shape)
    print("SelfAttention output shape:", out_selfattn.shape)
    # Directly compare values (allowing for floating point error)
    print("Output difference (L2 norm):", torch.norm(out_mha - out_selfattn).item())
    print("Are outputs (close)?", torch.allclose(out_mha, out_selfattn, atol=1e-6))
