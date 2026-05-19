from unittest import removeHandler
import torch
import math
import torch.nn.functional as F
from torch import nn


class EmbeddingLayer(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embeddings(x)


class PositionalEmbedding(nn.Module):

    def __init__(self, seq_len, d_model):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Register as buffer so it moves with .to(device) but isn't a learnable parameter
        self.register_buffer('pos_emb', self._precompute_embeddings())

    def _precompute_embeddings(self):
        """
        Generates a positional embedding based on
        PE(pos, 2i) = sin(pos/10000^(2i/d_model))       # EVEN dimensions
        PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))     # ODD dimensions
        
        #the result should be of shape (seq_len, d_model)
        """
        pos = torch.arange(0, self.seq_len).unsqueeze(1)  # (seq_len, 1)
        dim = torch.arange(0, self.d_model, 2)  # (d_model/2,) - even dimension indices
        
        # Compute the divisor: 10000^(2i/d_model)
        div_term = 10000 ** (dim / self.d_model)
        
        out = torch.empty(self.seq_len, self.d_model)
        out[:, 0::2] = torch.sin(pos / div_term)  # even dimensions get sin
        out[:, 1::2] = torch.cos(pos / div_term)  # odd dimensions get cos
        return out

    def forward(self, x):
        #to check are input token sequences the same length and dim as the embeddings?
        return x + self.pos_emb[None, :x.shape[1], :]


class MultiHeadAttention(nn.Module):
    """
    Scaled dot-product multi-head attention.
    """

    def __init__(self, d_input, n_heads, dq, dk, dv, do, causal_mask):
        super().__init__()
        self.d_input = d_input
        self.n_heads = n_heads
        self.dq = dq
        self.dk = dk    # these dims are placeholders
        self.dv = dv    # to check if all share the same dim
        self.do = do
        self.causal_mask = causal_mask

        self.W_Q = nn.Linear(d_input, dq)
        self.W_K = nn.Linear(d_input, dk)
        self.W_V = nn.Linear(d_input, dv)
        self.W_O = nn.Linear(dv, do)  # Projects concatenated heads (dv) to output (do)

    def forward(self, q, k, v, key_padding_mask=None):
        
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
        if key_padding_mask is not None:
            # key_padding_mask: (batch, key_len) -> (batch, 1, 1, key_len)
            attn_score = attn_score.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights = torch.softmax(attn_score, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Merge the 'n_heads' and 'head_dim' dimensions back into a single dimension after transposing
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(attn_output.size(0), attn_output.size(1), -1)
        
        # Apply output projection
        return self.W_O(attn_output)


class EncoderBlock(nn.Module):

    def __init__(self, d_model=512, d_ff=2048, n_heads=8, p_dropout=0.1): 
        super().__init__()

        self.mha = MultiHeadAttention(d_model, n_heads, d_model, d_model, d_model, d_model, False)
        self.ln_mha = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
                                nn.Linear(d_model, d_ff),
                                nn.ReLU(),
                                nn.Linear(d_ff, d_model)) #applied to each token individually
        self.dropout = nn.Dropout(p=p_dropout)
        self.ln_ff = nn.LayerNorm(d_model)

    def forward(self, x, src_padding_mask=None):
        x = self.ln_mha(self.dropout(self.mha(x, x, x, key_padding_mask=src_padding_mask)) + x) #shape is B, text_len, d_model
        x = self.ln_ff(x + self.dropout(self.ff(x)))
        return x



class DecoderBlock(nn.Module):
    
    def __init__(self, d_model, d_ff, n_heads, p_dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads        

        self.mha_1 = MultiHeadAttention(d_model, n_heads, d_model, d_model, d_model, d_model, True)
        self.ln_mha_1 = nn.LayerNorm(d_model)
        self.mha_2 = MultiHeadAttention(d_model, n_heads, d_model, d_model, d_model, d_model, False)
        self.ln_mha_2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.ReLU(),
                    nn.Linear(d_ff, d_model)) #applied to each token individually
        self.dropout = nn.Dropout(p=p_dropout)
        self.ln_ff = nn.LayerNorm(d_model)


    def forward(self, x, encoder_output, key_padding_mask=None, memory_padding_mask=None):
        x = self.ln_mha_1(self.dropout(self.mha_1(x, x, x, key_padding_mask=key_padding_mask)) + x)
        x = self.ln_mha_2(self.dropout(self.mha_2(x, encoder_output, encoder_output, key_padding_mask=memory_padding_mask)) + x)
        x = self.ln_ff(x + self.dropout(self.ff(x)))
        return x

class Transformer(nn.Module):
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
        vocab_size=37000,
        seq_len=256,
        d_model=512,
        d_ff=2048,
        n_heads=8,
        N=6,
        pad_token_id=0,
        eos_token_id=2,
        p_dropout=0.1,
        n_beams=4,
        alpha=0.6
        ):
        super().__init__()

        if d_ff // n_heads != 0:
            raise ValueError("n_heads must divide d_ff")

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

        self.embedding_layer_encoder = EmbeddingLayer(vocab_size, d_model)
        self.embedding_layer_decoder = EmbeddingLayer(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(seq_len, d_model)  # buffer auto-moves with .to(device)
        self.dropout = nn.Dropout(p_dropout)
        self.encoders = nn.ModuleList([EncoderBlock(d_model, d_ff, n_heads) for _ in range(N)])
        self.decoders = nn.ModuleList([DecoderBlock(d_model, d_ff, n_heads) for _ in range(N)])
        self.ff_output = nn.Linear(d_model, vocab_size)

        #the output layer and the embedding layer share weights.
        #embedding layer(token_id) -> weights
        #output layer(weights) -> token
        self.ff_output.weight = self.embedding_layer_decoder.embeddings.weight
        
    def forward(self, x, y):
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

        src_padding_mask = (x == self.pad_token_id)
        tgt_padding_mask = (y == self.pad_token_id)

        #pass through embedding layer
        x = self.embedding_layer_encoder(x) * math.sqrt(self.d_model)
        x = self.positional_embedding(x)
        x = self.dropout(x)

        for encoder in self.encoders:
            x = encoder(x, src_padding_mask=src_padding_mask)    

        # while not y.shape[-1] >= self.seq_len:
        y_embedding = self.embedding_layer_decoder(y) * math.sqrt(self.d_model)
        decoder_output = self.positional_embedding(y_embedding)
        decoder_output = self.dropout(decoder_output)
        for decoder in self.decoders:
            decoder_output = decoder(decoder_output, x, tgt_padding_mask, src_padding_mask)
        # output = F.softmax(self.ff_output(decoder_output), dim=-1)
        #     output = torch.argmax(output, dim=-1)
        #     y = torch.cat([y, output], dim=-1)

        return self.ff_output(decoder_output)
    
    def translate_beam_search(self, x: torch.Tensor, y: torch.Tensor):
        """
        Args:
            x: torch.Tensor (B, M)
            y: torch.Tensor (B, N)
        """

        with torch.no_grad():
            batch_size = x.shape[0]
            

            ### ENCODER
            src_padding_mask = (x == self.pad_token_id)
            x_enc = self.embedding_layer_encoder(x) * math.sqrt(self.d_model)
            x_enc = self.positional_embedding(x_enc)
            x_enc = self.dropout(x_enc)
            for encoder in self.encoders:
                x_enc = encoder(x_enc, src_padding_mask=src_padding_mask)    
            
            ### BEAM SEARCH PREP
            src_padding_mask = src_padding_mask.unsqueeze(1).repeat(1, self.n_beams, 1)
            x_enc = x_enc.unsqueeze(1).repeat(1, self.n_beams, 1, 1)
            #print("x.shape", x.shape)    
            #print("y.shape", y.shape)
            first_log_probs = F.log_softmax(self.forward(x, y), dim=-1)  # (B, N, vocab_size)
            
            #print(first_log_probs)
            topk_vals, topk_indices = torch.topk(first_log_probs, k=self.n_beams, dim=-1)
            # print("initial topk vals", topk_vals)
            # print("initial topk vals shape", topk_vals.shape)
            # print("initial topk indices", topk_indices)
            # print("initial topk indices shape", topk_indices.shape) # 1, 1, 2
            
            
            y = y.unsqueeze(1).repeat(1, self.n_beams, 1)
            #print(y.shape) #1, 2, 1
            y = torch.cat((y, topk_indices.transpose(1, 2)), dim=-1)
            #print(y.shape)
                

            #print("y.shape", y.shape)

            #print("topkvals shape", topk_vals.shape)
            
            log_probs = torch.zeros((batch_size, self.n_beams))  #B, n_beams
            #print("logprobs", log_probs.shape)
            log_probs = log_probs.to(next(self.parameters()).device)
            active_beams = torch.ones((batch_size, self.n_beams), dtype=torch.bool)
            active_beams = active_beams.to(next(self.parameters()).device)
            active_beams = active_beams & ~(y[:, :, -1] == self.eos_token_id)

            remaining_batches = list(range(x_enc.size(0)))

            while remaining_batches and y.shape[1] < self.seq_len:

                x_remaining = x_enc[remaining_batches]
                # print(f"x_remaining original shape: {x_remaining.shape}")
                x_remaining = x_remaining.view(-1, *x_remaining.shape[2:]) 
        
                # print(f"x_remaining after view: {x_remaining.shape}")

                y_remaining = y[remaining_batches]
                # print(f"y_remaining original shape: {y_remaining.shape}")
                y_remaining = y_remaining.view(-1, y_remaining.shape[-1]) #flatten the beams    
                # print(f"y_remaining after view: {y_remaining.shape}")


                # print("src_padding_mask.shape", src_padding_mask.shape)
                tgt_padding_mask = (y_remaining == self.pad_token_id)
                y_embedding = self.embedding_layer_decoder(y_remaining) * math.sqrt(self.d_model)
                decoder_output = self.positional_embedding(y_embedding)
                decoder_output = self.dropout(decoder_output)
                # print("decoder_output.shape", decoder_output.shape)
                i = 0
                for decoder in self.decoders:
                    #print("decoder", i)
                    i+=1
                    # print("decoder_output.shape (input):", decoder_output.shape)
                    # print("x_remaining.shape:", x_remaining.shape)
                    # print("tgt_padding_mask.shape:", tgt_padding_mask.shape)
                    # print("src_padding_mask[remaining_batches].view(-1, *x_remaining.shape[2:]).shape:",
                    #       src_padding_mask[remaining_batches].view(-1, *x_remaining.shape[2:]).shape)
                    
                    decoder_output = decoder(
                        decoder_output,
                        x_remaining,
                
                        tgt_padding_mask,
                        src_padding_mask[remaining_batches].view(-1, *x_remaining.shape[2:]))
                # print("---------------------------------------------------------")
                # print("decoder_output[:, -1].shape:", decoder_output[:, -1].shape)
                next_token = self.ff_output(decoder_output[:, -1])
                #print("next_token.shape:", next_token.shape)
                token_logprobs = F.log_softmax(next_token)
                #print("token_logprobs.shape:", token_logprobs.shape)
                token_logprobs = token_logprobs.reshape(x_enc.shape[0], self.n_beams, -1)
                #print(token_logprobs.shape)
                #print("max token_logprobs per beam:", token_logprobs.max(dim=-1).values)
        
                #always select a finished beam 
                token_logprobs[active_beams[remaining_batches] == False] = float('-inf') 
                token_logprobs[active_beams[remaining_batches] == False, self.pad_token_id] = 0.0
                
                token_logprobs[active_beams[remaining_batches], self.pad_token_id] = float('-inf')

                token_logprobs += log_probs[remaining_batches].unsqueeze(-1)
                flat_token_logprobs = token_logprobs.view(batch_size, -1)
                topk_vals, topk_indices = torch.topk(flat_token_logprobs, k=self.n_beams, dim=-1)

                log_probs[remaining_batches] = topk_vals
                #print("log_probs cumulative", log_probs)
                # Map flat indices back to (beam, token) pairs
                beam_indices = topk_indices // self.vocab_size
                token_indices = topk_indices % self.vocab_size
                
                y_remaining = y_remaining.view(len(remaining_batches), self.n_beams, -1)
                batch_idx = torch.arange(y_remaining.shape[0]).unsqueeze(1)
                
                # print("------------------------------------")
                # print("y_remaining.shape (before indexing):", y_remaining.shape)
                # print("remaining_indices.shape:", batch_idx.shape)
                # print("beam_indices.shape:", beam_indices.shape)
                y_remaining = y_remaining[batch_idx, beam_indices, :]
                # print("y_remaining.shape (after indexing):", y_remaining.shape)
        
                y_remaining = torch.cat((y_remaining, token_indices.unsqueeze(-1)), dim=-1)

                # print("y_rem.shape", y_remaining.shape)
                # print("y.shape", y.shape)

                y = torch.cat((y, torch.full([y.shape[0], y.shape[1], 1], self.pad_token_id).cuda()), dim=-1)
                # print("y.shape", y.shape)
                y[remaining_batches] = y_remaining

                active_beams = active_beams[batch_idx, beam_indices]
                just_finished = token_indices == self.eos_token_id
                active_beams = active_beams & ~just_finished
                if not active_beams.any():
                    break     

            #NORMALIZE AND RETURN HIGHEST LOGPROB BEAM PER BATCH BASED ON NORMALIZATION
            lengths = y.shape[-1]
            num_pad = (y == self.pad_token_id).sum(dim=-1)
            normalized_lengths = (lengths - num_pad).clamp(min=1)
            log_probs = log_probs / (normalized_lengths.float() ** self.alpha)
            #print("normalized log_probs", log_probs)

                
            return y


    def translate(self, x, y):
        """
        TODO - expects a tokenized sentence as encoder input and BOS token as decoder input
        Args:
            x: torch.tensor, shape (B, input_seq_len) (typically (B, self.seq_len))
            y: torch.tensor, shape (B, output_seq_len) (typically (B, 1))
        Return:
            Tensor of shape B, seq_len?
        """

        src_padding_mask = (x == self.pad_token_id)

        #pass through embedding layer
        x = self.embedding_layer_encoder(x) * math.sqrt(self.d_model)
        x = self.positional_embedding(x)
        x = self.dropout(x)
        for encoder in self.encoders:
            x = encoder(x, src_padding_mask=src_padding_mask)    

        #tracks uncompleted sequences (those without <EOS> token)
        remaining = list(range(x.size(0)))

        while remaining and y.shape[1] < self.seq_len:
            
            tgt_padding_mask = (y == self.pad_token_id)
            y_embedding = self.embedding_layer_decoder(y) * math.sqrt(self.d_model)
            decoder_output = self.positional_embedding(y_embedding)
            decoder_output = self.dropout(decoder_output)
            
            # Select current part of decoder_output based on remaining_sentences
            decoder_output = decoder_output[remaining]
            
            for decoder in self.decoders:
                decoder_output = decoder(
                    decoder_output,
                    x[remaining],
                    tgt_padding_mask[remaining],
                    src_padding_mask[remaining])

            next_token = self.ff_output(decoder_output[:, -1])
            next_token = torch.argmax(next_token, dim=-1, keepdim=True)

            token_column = torch.full((y.shape[0], 1), self.pad_token_id, dtype=torch.long, device=y.device)
            token_column[remaining] = next_token
            next_token = token_column 
            y = torch.cat((y, next_token), dim=-1)

            #update incomplete sequence tracker
            eos_mask = (next_token.squeeze(-1) == self.eos_token_id)
            remaining = [idx for idx, is_eos in zip(remaining, eos_mask) if not is_eos]
        
            
        return y
        


# if __name__ == "__main__":

#     transformer = Transformer(
#         vocab_size=512,
#         seq_len=64,
#         d_model=128,
#         d_ff=256,
#         n_heads=1,
#         N=4
#     )

#     batch_size = 8
#     seq_len = 64

#     x = torch.randint(low=0, high=512, size=(batch_size, seq_len))
#     y = torch.zeros(batch_size, 1, dtype=torch.long)

#     output = transformer(x, y)


#     print("Output shape:", output.shape)
#     print("Output:", output)

if __name__ == "__main__":
    from src.tokenizers.BPE import BytePairEncoding
    from src.train.lit_transformer import LitTransformer
    import torch

    # You need to instantiate the model used by LitTransformer and pass it in as the argument.
    # Assume you know (or can specify) the Transformer architecture / args used at training.
    # Here is an example (adjust paths/params as needed):

    # Example: Load model config -- in production, load those from the checkpoint or your experiment config!
    vocab_size = 37000          # Use actual vocab_size used for training
    d_model = 256               # actual d_model
    d_ff = 512                  # actual d_ff
    n_heads = 8                 # actual n_heads
    N = 6                       # actual N
    seq_len = 256               # actual seq_len

    # Create the underlying Transformer model -- adjust the imports and values as needed
    transformer = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        N=N,
        seq_len=seq_len,
    )
    num_params = sum(p.numel() for p in transformer.parameters())
    print("Number of parameters in Transformer:", num_params)

    transformer.eval()

    # Now load the LitTransformer with the transformer argument
    lit_model = LitTransformer.load_from_checkpoint(
        "/home/mcvjetko/phd/projects/transformer/experiments/en-hr/last.ckpt",
        transformer=transformer
    )
    transformer = lit_model.transformer
    transformer.cuda()
    bpe_tokenizer = BytePairEncoding.from_file("/home/mcvjetko/phd/projects/transformer/experiments/en-hr-tokenizers/tokenizer_37000.json")
    

    raw =  "Once upon a midnight dreary, while I pondered weak and weary over many a quaint and curious volume of forgotten lore."
    text = bpe_tokenizer.tokenize(raw, max_length=256, add_special=True, pad=True)
    #print("decoded", bpe_tokenizer.decode(text))
    text = torch.tensor(text).to("cuda").unsqueeze(0)
    #text = text.repeat(2, 1)
    #print(text)Julien is running often. -> Julien često trči.
    # print(text.shape)
    y = torch.full((1, 1), 1, dtype=torch.long).to("cuda")
    #y = y.repeat(2, 1)
    print(text.shape, y.shape)
    # print(y)
    # print(raw + " -> " + bpe_tokenizer.decode(transformer.translate(text, y)[1].tolist()))
    # print(raw + " -> " + bpe_tokenizer.decode(transformer.translate_beam_search(text, y)[0][0].tolist()))
    # print(raw + " -> " + bpe_tokenizer.decode(transformer.translate_beam_search(text, y)[0][1].tolist()))
    output = transformer.translate_beam_search(text, y)
    for o in output:
        for k in o:
            print(bpe_tokenizer.decode(k))