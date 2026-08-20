import torch
from torch import nn

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=6,
                 block_size=256, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, x):
        B,T=x.shape
        if T > self.block_size:
            x=x[:,-self.block_size:]
            T=x.shape[1]
        pos=torch.arange(T,device=x.device)
        h=self.token_emb(x)+self.pos_emb(pos)[None,:,:]
        mask=torch.triu(torch.ones(T,T,device=x.device,dtype=torch.bool),1)
        h=self.encoder(h, mask=mask)
        return self.lm_head(self.norm(h))
