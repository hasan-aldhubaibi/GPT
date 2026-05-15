import torch
import torch.nn as nn

from model.transformer_block import TransformerBlock


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.vocab_size = config["vocab_size"]
        self.block_size = config["block_size"]

        self.n_layer = config["n_layer"]
        self.d_model = config["d_model"]

        # =====================================================
        # TOKEN EMBEDDING
        # =====================================================

        self.token_emb = nn.Embedding(
            self.vocab_size,
            self.d_model
        )

        # =====================================================
        # TRANSFORMER BLOCKS
        # =====================================================

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                n_head=config["n_head"],
                dropout=config["dropout"],
                block_size=self.block_size
            )
            for _ in range(self.n_layer)
        ])

        # =====================================================
        # FINAL LAYER NORM
        # =====================================================

        self.ln_f = nn.LayerNorm(self.d_model)

        # =====================================================
        # LM HEAD
        # =====================================================

        self.head = nn.Linear(
            self.d_model,
            self.vocab_size,
            bias=False
        )

        # =====================================================
        # WEIGHT TYING
        # =====================================================

        self.head.weight = self.token_emb.weight

        # =====================================================
        # INIT
        # =====================================================

        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, idx, return_attn=False):

        B, T = idx.shape

        x = self.token_emb(idx)

        attn_weights = None

        for i, block in enumerate(self.blocks):

            if return_attn and i == len(self.blocks) - 1:
                x, attn_weights = block(
                    x,
                    return_attn=True
                )
            else:
                x = block(x)

        x = self.ln_f(x)

        logits = self.head(x)

        if return_attn:
            return logits, attn_weights

        return logits