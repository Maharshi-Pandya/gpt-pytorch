import torch


def get_device() -> str:
    """
    return cpu/cuda/mps if available
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class ConfigGPT:
    "config params for GPT"

    def __init__(self) -> None:
        self.vocab_size = None
        self.d_embed = 768
        self.context_size = 1024
        self.n_attn_layers = 12
        self.n_attn_heads = 12
        self.resd_pdrop = 0.1
        self.embed_pdrop = 0.1
        self.attn_pdrop = 0.1
    
    @staticmethod
    def get_default(cls):
        return cls()
