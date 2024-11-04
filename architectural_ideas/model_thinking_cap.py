import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, input.shape[-1:], self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # Flash attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        # Calculate query, key, values for all heads
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Reshape and transpose for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Causal self-attention
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, hs)
        # Reshape back to (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    d_hidden: int = 384  # Hidden size for the MLP processing distributions

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Transformer components
        self.transformer = nn.ModuleDict(dict(
            # No token embeddings anymore
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Encoder for distributions
        self.dist_mlp = nn.Sequential(
            nn.Linear(config.vocab_size, config.d_hidden),
            nn.ReLU(),
            nn.Linear(config.d_hidden, config.n_embd),
        )

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, distributions=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # (t)
        pos_emb = self.transformer.wpe(pos)  # (t, n_embd)

        # Distributions
        if distributions is None:
            # Convert idx to one-hot distributions
            distributions = F.one_hot(idx, num_classes=self.config.vocab_size).float()  # (b, t, vocab_size)
        else:
            # Use the provided distributions
            pass  # distributions is provided, shape (b, t, vocab_size)

        # Encode distributions
        dist_emb = self.dist_mlp(distributions)  # (b, t, n_embd)

        # Combine embeddings
        x = dist_emb + pos_emb  # (b, t, n_embd)

        x = self.transformer.drop(x)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (b, t, vocab_size)

        if targets is not None:
            # Training phase
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            # Inference phase
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, entropy_threshold=3.0, max_forward_passes=10):
        device = idx.device
        batch_size = idx.size(0)
        # Initialize distributions with one-hot vectors for idx
        distributions = F.one_hot(idx, num_classes=self.config.vocab_size).float()  # (b, t, vocab_size)

        for _ in range(max_new_tokens):
            # Initialize thinking masks and placeholders
            thinking_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            steps = 0
            dist_new_tokens = torch.zeros(batch_size, 1, self.config.vocab_size, device=device)

            # Initialize temporary distributions for sequences still thinking
            dist_thinking = distributions.clone()

            while thinking_mask.any() and steps < max_forward_passes:
                steps += 1

                # Prepare dist_cond
                dist_cond = dist_thinking

                # Crop sequences if necessary
                if dist_cond.size(1) > self.config.block_size:
                    dist_cond = dist_cond[:, -self.config.block_size:, :]

                # Forward pass
                idx_cond = torch.zeros(batch_size, dist_cond.size(1), dtype=torch.long, device=device)  # Dummy idx
                logits, _ = self(idx_cond, distributions=dist_cond)
                logits = logits[:, -1, :] / temperature  # (b, vocab_size)

                # Optionally apply top_k
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                # Compute probabilities
                probs = F.softmax(logits, dim=-1)  # (b, vocab_size)

                # Compute entropy
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # (b)

                # Determine which sequences will output tokens
                output_mask = ((entropy < entropy_threshold) | (steps >= max_forward_passes)) & thinking_mask

                # Sequences that will output tokens
                if output_mask.any():
                    dist_new_tokens[output_mask, 0, :] = probs[output_mask]
                    thinking_mask[output_mask] = False

                # Sequences still thinking
                if thinking_mask.any():
                    # Update distributions for sequences still thinking
                    dist_thinking = torch.cat([dist_thinking, probs.unsqueeze(1)], dim=1)
                else:
                    break  # All sequences have output tokens

            # Append new distributions to distributions
            distributions = torch.cat([distributions, dist_new_tokens], dim=1)

            # Convert the output distributions to token indices (e.g., by taking argmax)
            idx_next = dist_new_tokens.squeeze(1).argmax(dim=-1, keepdim=True)  # (b, 1)

            # Append new tokens to idx
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Create optim groups
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
