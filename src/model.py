# todo: implement the model

import math
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
        q: torch.Tensor = einops.rearrange(q, "bs sl (nh ed) -> bs nh sl ed", nh=self.n_heads)
        k: torch.Tensor = einops.rearrange(k, "bs sl (nh ed) -> bs nh sl ed", nh=self.n_heads)
        v: torch.Tensor = einops.rearrange(v, "bs sl (nh ed) -> bs nh sl ed", nh=self.n_heads)

        # masked self-attention: softmax(q * kt / sqrt(dk)) * v
        kT: torch.Tensor = einops.rearrange(k, "bs nh sl ed -> bs nh ed sl")

        att = (q @ kT) * (1 / math.sqrt(k.size(-1)))    # (bs, nh, sl, sl)
        att = att.masked_fill(self.attn_mask[:, :, :sl, :sl] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v     # (bs, nh, sl, sl) * (bs, nh, sl, ed) -> (bs, nh, sl, ed)
        y = einops.rearrange(y, "bs nh sl ed -> bs sl (nh ed)")     # concat heads

        output = self.resd_drop(self.o_proj(y))
        return output


class Block(nn.Module):
    """
    1 block of layernom1 + attention + residual + layernorm2 + ffn + residual
    """

    def __init__(self, config: ConfigGPT):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.d_embed)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_embed)
        self.ffn = nn.ModuleDict(dict(
            i_proj = nn.Linear(config.d_embed, 4 * config.d_embed),
            o_proj = nn.Linear(4 * config.d_embed, config.d_embed),
            act = nn.GELU(),
            drop = nn.Dropout(config.resd_pdrop)
        ))

        ffn = self.ffn
        # forward pass of ffn
        self.ffnf = lambda x: ffn.drop(ffn.o_proj(ffn.act(ffn.i_proj(x))))

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffnf(self.ln2(x))
        return x
    

class GPT2(nn.Module):
    def __init__(self, config: ConfigGPT):
        super().__init__()

        self.context_size = config.context_size

        self.ln1 = nn.LayerNorm(config.d_embed)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_embed),
            wpe = nn.Embedding(self.context_size, config.d_embed),
            drop = nn.Dropout(config.embed_pdrop),
            attn = nn.ModuleList([Block(config) for _ in range(config.n_attn_heads)]),
            lnf = nn.LayerNorm(config.d_embed)
        ))
        # one hot ?
        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)

        # todo: weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        with N(0, 0.02)
        """