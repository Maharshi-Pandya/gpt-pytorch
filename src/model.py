# todo: implement the model

import torch
import torch.nn as nn
import einops

from utils import ConfigGPT


class CausalSelfAttention(nn.Module):
    """
    masked self attention layer
    """

    def __init__(self, config: ConfigGPT):
        super().__init__()
        assert config.d_embed % config.n_attn_heads == 0

        # projection matrix for q, k, v in one go
        self.w_proj = nn.Linear(config.d_embed, 3 * config.d_embed)
        # projection matrix for o
        self.o_proj = nn.Linear(config.d_embed, config.d_embed)

        # for regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resd_drop = nn.Dropout(config.resd_pdrop)

        # attention mask (tokens preceding current token)
        self.attn_mask = torch.tril(
            torch.ones(config.context_size, config.context_size)
        )
        self.attn_mask = einops.rearrange(self.attn_mask, "h w -> () () h w")   # (1, 1, h, w)

        self.n_heads = config.n_attn_heads
        self.d_embed = config.d_embed

    def forward(self, x: torch.Tensor):
        assert len(x.size()) == 3, f"Cannot forward tensor of shape {x.size()}"
        # batch size, sequence length, embedding dimension = d_embed
        bs, sl, ed = x.size()

        # compute q, k, v for each head 
        # nn.Linear input (bs, sl, d_embed) and output (bs, sl, 3 * d_embed)
        q, k, v = self.w_proj(x).split(self.d_embed, dim=2)
        
        # resulting shape (bs, nh, sl, ed // nh)
        q = einops.rearrange(q, "bs sl (nh ed) -> bs nh sl ed", nh=self.n_heads)
        k = einops.rearrange(k, "bs sl (nh ed) -> bs nh sl ed", nh=self.n_heads)
        v = einops.rearrange(v, "bs sl (nh ed) -> bs nh sl ed", nh=self.n_heads)
