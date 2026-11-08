import torch
from torch import nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self,
                 embedding_dim : int = 384,
                 num_heads : int = 6,
                 dropout : float = 0.1,):
        super().__init__()

        self.ln1 = nn.LayerNorm(normalized_shape=embedding_dim)

        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                               num_heads=num_heads,
                                               dropout=dropout,
                                               batch_first=True)

        self.ln2 = nn.LayerNorm(normalized_shape=embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim*4),
            nn.GELU(),
            nn.Linear(in_features=embedding_dim*4, out_features=embedding_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self,x:torch.Tensor,causal_mask:torch.Tensor):
        x_norm = self.ln1(x)

        attn_output,_ = self.attention(query=x_norm,
                                       key=x_norm,
                                       value=x_norm,
                                       attn_mask=causal_mask,
                                       is_causal=False)

        x = x + attn_output

        x = self.mlp(self.ln2(x)) + x

        return x


class GPT(nn.Module):
    def __init__(self,
                 vocab_size : int = 473,
                 embedding_dim : int = 384,
                 num_heads : int = 6,
                 num_layers : int = 6,
                 dropout : float = 0.1,
                 block_size : int = 256,
                 ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=block_size,
            embedding_dim=embedding_dim
        )

        self.dropout = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoder(embedding_dim=embedding_dim,
                                   num_heads=num_heads,
                                   dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = nn.LayerNorm(normalized_shape=embedding_dim)

        self.output_proj = nn.Linear(
            in_features=embedding_dim,
            out_features=vocab_size
        )

        self.loss_fn = nn.CrossEntropyLoss()


        causal_mask = torch.triu(
            torch.ones(block_size, block_size,dtype=torch.bool),
            diagonal=1
        )

        self.register_buffer('causal_mask', causal_mask)

        self.apply(self._init_weights)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total params: {total_params}")

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self,input_ids: torch.Tensor,targets: torch.Tensor=None):

        batch_size,seq_length = input_ids.shape
        device = input_ids.device

        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_length,device=device)
        pos_emb = self.position_embedding(positions)

        #loook up table for pos_embedding
        #[0.001,0.24,0.115,0.112] -> 0
        #[0.001,0.24,0.115,0.112] -> 1
        #[0.001,0.24,0.115,0.112] -> 2
        #[0.001,0.24,0.115,0.112] -> 3
        #[0.001,0.24,0.115,0.112] -> 4

        x = self.dropout(token_emb + pos_emb)
        mask = self.causal_mask[:seq_length,:seq_length]

        for block in self.blocks:
            x = block(x,mask)

        x = self.ln_final(x)

        logits = self.output_proj(x)
        #print(f"Logits shape: {logits.shape}")
        # (4,256,65) 4 -> batch_size,256 -> block_size , 65 -> vocab_size
        # logit shape (batch_size,seq_len,vocab_size)
        # target shape (batch_size,seq_len)

        # (batch_size * seq_len,vocab_size)
        # (batch_size,seq_len, )

        loss = None
        if targets is not None:

            batch_size,seq_length,vocab_size = logits.shape
            logits_flat = logits.reshape(batch_size * seq_length,vocab_size)
            targets_flat = targets.reshape(batch_size * seq_length)
            loss = self.loss_fn(logits_flat,targets_flat)
        #print(f"Loss: {loss}")

        return logits,loss

    @torch.no_grad()
    def generate(self,input_ids:torch.Tensor,max_new_tokens:int,temperature:float=0.5):
        for _ in range(max_new_tokens):

            if input_ids.size(1) <= self.block_size:
                current_input = input_ids
            else:
                current_input = input_ids[:,-self.block_size:]

            logits,_ = self.forward(current_input)

            #print(f"generate logits shape: {logits.shape}")

            last_logits = logits[:,-1,:]

            #print(f"last logits shape: {last_logits.shape}")

            last_logits = last_logits/temperature

            probs = F.softmax(last_logits,dim=-1)
            #print(f"probs: {probs}")
            next_token = torch.multinomial(probs,num_samples=1)

            if next_token.item() == 0:
                break

            input_ids = torch.cat([input_ids,next_token],dim=1)
            #print(f"input_ids shape: {input_ids.shape}")

        return input_ids


if __name__ == '__main__':
    '''
    gpt = GPT()
    batch_size = 1
    seq_length = 256
    dummy_input = torch.randint(0,65,(batch_size,seq_length))
    dummy_target = torch.randint(0, 65, (batch_size, seq_length))
    b,s = dummy_input.shape
    pos = gpt.position_embedding(torch.arange(s))
    print(f"pos: {pos}")
    print(f"pos shape: {pos.shape}")

    print(f"dummy_input.shape: {dummy_input.shape}")
    dummy_embedd = gpt.token_embedding(dummy_input)
    print(f"dummy_input.shape: {dummy_embedd.shape}")

    random_input = torch.zeros((2,2,2))
    print(f"random input: {random_input}")
    print(f"random_input.shape: {random_input.shape}")
    my_input = torch.ones((2,2))
    print(f"my_input: {my_input}")
    print(f"my_input.shape: {my_input.shape}")
    total_input = random_input + my_input
    print(f"total_input: {total_input}")
    print(f"total_input.shape: {total_input.shape}")
    gpt(input_ids = dummy_input,targets=dummy_target)
    gpt.generate(dummy_input,max_new_tokens=1,temperature=0.5)
    '''
    from dataset import load_data

    train_data, test_data, tokenizer = load_data()
    gpt = GPT()
    prompt = "efe\nbaba"
    prompt_ids = torch.tensor(tokenizer.encode(prompt),dtype=torch.long).unsqueeze(0)
    print(f"prompt_ids: {prompt_ids}")
    print(f"prompt_ids.shape: {prompt_ids.shape}")
    input_ids = gpt.generate(prompt_ids,max_new_tokens=20,temperature=0.5)
    print(tokenizer.decode(input_ids[0]))





