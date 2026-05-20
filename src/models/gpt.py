import torch
import math
from torch import nn


class MultiHeadAttention(nn.Module):
    """
    Scaled dot-product multi-head attention.
    """

    def __init__(self, d_input, n_heads, dq, dk, dv, do, causal_mask, p_dropout):
        super().__init__()
        self.d_input = d_input
        self.n_heads = n_heads
        self.dq = dq
        self.dk = dk    # these dims are placeholders
        self.dv = dv    # to check if all share the same dim
        self.do = do
        self.causal_mask = causal_mask
        self.dropout = nn.Dropout(p_dropout)

        self.W_Q = nn.Linear(d_input, dq)
        self.W_K = nn.Linear(d_input, dk)
        self.W_V = nn.Linear(d_input, dv)
        self.W_O = nn.Linear(dv, do)  # Projects concatenated heads (dv) to output (do)

    def forward(self, q, k, v, padding_mask=None):
        
        Q = self.W_Q(q).view(q.shape[0], q.shape[1], self.n_heads, self.dq // self.n_heads)
        K = self.W_K(k).view(k.shape[0], k.shape[1], self.n_heads, self.dk // self.n_heads)
        V = self.W_V(v).view(v.shape[0], v.shape[1], self.n_heads, self.dv // self.n_heads)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        attn_score = torch.matmul(Q, K.transpose(-2, -1))
        attn_score /= math.sqrt(Q.shape[-1])
        
        # Apply causal mask (for decoder self-attention)
        if self.causal_mask:
            device = attn_score.device
            causal = torch.triu(
                torch.ones((attn_score.shape[-2], attn_score.shape[-1]), dtype=torch.bool, device=device)
            , diagonal=1)
            attn_score = attn_score.masked_fill(causal, float('-inf'))
        
        # Apply key padding mask (to ignore padding tokens)
        #print("key mask", key_padding_mask.shape)
        #print("attn score", attn_score.shape)
        if padding_mask is not None:
            # key_padding_mask: (batch, key_len) -> (batch, 1, 1, key_len)
            attn_score = attn_score.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights = torch.softmax(attn_score, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)

        # Merge the 'n_heads' and 'head_dim' dimensions back into a single dimension after transposing
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(attn_output.size(0), attn_output.size(1), -1)
        
        # Apply output projection
        return self.W_O(attn_output)


class DecoderBlock(nn.Module):
    
    def __init__(self, d_model, d_ff, n_heads, p_dropout):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads        

        self.ln_mha = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads, d_model, d_model, d_model, d_model, True, p_dropout)
        self.dropout = nn.Dropout(p=p_dropout)
        self.ff = nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model)) #applied to each token individually

        self.ln_ff = nn.LayerNorm(d_model)


    def forward(self, x, padding_mask=None):
        x_ln = self.ln_mha(x)
        x = x + self.dropout(self.mha(x_ln, x_ln, x_ln, padding_mask=padding_mask))
        x = x + self.dropout(self.ff(self.ln_ff(x)))
        return x

class GPT2(nn.Module):
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
        N=24,
        pad_token_id=0,
        eos_token_id=2,
        p_dropout=0.1,
        n_beams=4,
        alpha=0.6
        ):
        super().__init__()

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

        self.n_beams = n_beams
        self.alpha = alpha

        self.embedding_layer = nn.Embedding(vocab_size, d_model)
        self.embedding_pos = nn.Embedding(seq_len, d_model)  # buffer auto-moves with .to(device)
        self.dropout = nn.Dropout(p_dropout)
        self.decoders = nn.ModuleList([DecoderBlock(d_model, d_ff, n_heads, p_dropout) for _ in range(N)])
        self.ln_final = nn.LayerNorm(d_model)
        self.ff_output = nn.Linear(d_model, vocab_size, bias=False)

        #the output layer and the embedding layer share weights.
        self.ff_output.weight = self.embedding_layer.weight
        
        self.apply(self._init_weights)
    
        for name, params in self.named_parameters():
            if "ff.0.weight" in name:
                print(name)
                nn.init.normal_(params, mean=0.0, std=0.02 / math.sqrt(2*N))
            if "ff.2.weight" in name:
                print(name)
                nn.init.normal_(params, mean=0.0, std=0.02 / math.sqrt(2*N))
            if "mha.W_O.weight" in name:
                print(name)
                nn.init.normal_(params, mean=0.0, std=0.02 / math.sqrt(2*N))


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

    

    def forward(self, x):
        """
        Args:
            x: torch.tensor, shape (B, N, vocab_size)
            y: torch.tensor, shape (B, M, vocab_size)
            Where M, N are sequence lengths.
        Return:
            Tensor of shape (B, M, vocab_size)
        # """
        # print("in forward")
        # print(x.shape, y.shape)

        padding_mask = (x == self.pad_token_id)



        # while not y.shape[-1] >= self.seq_len:
        x = self.embedding_layer(x)
        pos = torch.arange(0, x.shape[1]).unsqueeze(0)
        x = x + self.embedding_pos(pos)

        decoder_output = self.dropout(x)
        for decoder in self.decoders:
            decoder_output = decoder(decoder_output, padding_mask)
        # output = F.softmax(self.ff_output(decoder_output), dim=-1)
        #     output = torch.argmax(output, dim=-1)
        #     y = torch.cat([y, output], dim=-1)

        return self.ff_output(self.ln_final(decoder_output))
    
        


if __name__ == "__main__":

    transformer = GPT2()
    num_params = sum(p.numel() for p in transformer.parameters())
    print(f"Number of parameters: {num_params}")
    batch_size = 8
    seq_len = 512

    x = torch.randint(low=0, high=512, size=(batch_size, seq_len))

    output = transformer(x)


    print("Output shape:", output.shape)
    print("Output:", output)
