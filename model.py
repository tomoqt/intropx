"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

# This model enhances a standard autoregressive transformer decoder by introducing an uncertainty-aware token
# that captures the model's confidence in previous predictions. A "distribution context" is maintained alongside
# the token context, tracking the model's probability distributions over the vocabulary. The distribution context
# is initialized with special distributions and evolves in sync with the token context, ensuring that uncertainty 
# information is used to inform future predictions.
#
# 1. Autoregressive Decoder Setup:
#    The model operates as a decoder-only transformer, generating tokens one at a time, where each token depends 
#    only on the previous tokens in the sequence. A distribution context is maintained for the last L tokens.
# 
# 2. Distribution Context Initialization and Evolution:
#    - Initial state: The distribution context is initialized with Dirac delta functions for each token in the 
#      context, representing certainty in the known tokens.
#    - Padding: If fewer tokens are present than the context size L, the context is padded.
#    - Update: New probability distributions generated during the forward pass are inserted into the context, 
#      replacing the oldest, mimicking how tokens are added to the token context.
#
# 3. Processing the Distribution Context:
#    The stored probability distributions are processed by a CNN to extract patterns. The output is pooled and passed
#    through an MLP to produce an embedding representing the uncertainty across the context window.
#
# 4. Appending the Uncertainty-Aware Token:
#    This uncertainty-aware embedding is treated as token L+1 and appended to the sequence of token embeddings,
#    resulting in a tensor of size (B, L+1, d_model).
#
# 5. Causal Autoregressive Rollout:
#    The model respects causality by attending only to past tokens. A causal mask ensures that no future tokens are
#    accessed during self-attention. The uncertainty-aware token is included in this attention.
#
# 6. Transformer Input and Token Generation:
#    The enriched sequence, including the uncertainty-aware token, is passed through the transformer to generate 
#    the next token. This structure allows the model to factor in its own uncertainty during token generation.


import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
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
    block_size: int = 1024  # Ensure this is large enough for your use case
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    use_sparse_distributions: bool = False
    distribution_context_size: int = 1024  # or any other default value
    d_hidden: int = 128  # Add this line

# {{ Add DistributionEmbedder class }}
class DistributionEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=config.vocab_size, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(128, config.n_embd)

    def forward(self, distribution_context):
        # distribution_context shape: (batch_size, context_size, vocab_size)
        x = distribution_context.permute(0, 2, 1)  # (batch_size, vocab_size, context_size)
        x = self.conv(x)  # (batch_size, 128, context_size)
        x = self.relu(x)
        x = x.mean(dim=2)  # Global average pooling over context_size: (batch_size, 128)
        x = self.fc(x)  # (batch_size, n_embd)
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # Weight tying

        # Initialize all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

        self.distribution_context = None
        self.use_sparse_distributions = config.use_sparse_distributions
        self.distribution_context_size = config.distribution_context_size

        # Initialize the distribution context with zeros
        self.distribution_context = torch.zeros(1, self.distribution_context_size, config.vocab_size)

        # {{ Initialize the DistributionEmbedder }}
        self.distribution_embedder = DistributionEmbedder(config)
        
        # {{ Adjust token embedding size for concatenation }}
        # Remove or modify this line as we'll be appending a new token, not concatenating embeddings
        # self.transformer.wte = nn.Embedding(config.vocab_size, config.n_embd * 2)
        self.transformer.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        # {{ Initialize components for Uncertainty-Aware Token }}
        self.cnn = nn.Conv1d(in_channels=config.vocab_size, out_channels=128, kernel_size=3, padding=1)  # Adjust channels as needed
        self.relu = nn.ReLU()
        self.mlp = nn.Sequential(
            nn.Linear(128, config.d_hidden),  # config.d_hidden should match model embedding dimension
            nn.GELU(),
            nn.Linear(config.d_hidden, config.n_embd)
        )
        
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

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # Initialize distribution context if it's None or not properly shaped
        if self.distribution_context is None or self.distribution_context.size(0) != b:
            self._init_distribution_context(idx)
        
        # Generate additional embedding from distribution context
        distribution_emb = self.distribution_embedder(self.distribution_context)  # Shape: (b, n_embd)
        distribution_emb = distribution_emb.unsqueeze(1)  # Shape: (b, 1, n_embd)
        
        # Generate token embeddings
        tok_emb = self.transformer.wte(idx)  # Shape: (b, t, n_embd)
        
        # Generate position embeddings (only for the input tokens, not for the uncertainty token)
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)  # Shape: (t, n_embd)
        
        # Combine token embeddings and positional embeddings
        x = tok_emb + pos_emb
        
        # Append uncertainty-aware embedding (without positional encoding)
        x = torch.cat((x, distribution_emb), dim=1)  # Shape: (b, t+1, n_embd)
        
        # Apply dropout
        x = self.transformer.drop(x)
        
        # Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training phase
            logits = self.lm_head(x[:, :-1])  # Exclude the last token (uncertainty token)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            # Inference phase
            logits = self.lm_head(x[:, [-2]])  # Use the second-to-last token (last actual token, not uncertainty token)
            return logits, None

    # {{ Add CNN + MLP pipeline for Uncertainty-Aware Token }}
    def generate_uncertainty_token(self):
        # Input to CNN: distribution_context (batch_size, L, V)
        x = self.distribution_context  # Shape: (batch_size, L, V)
        x = x.permute(0, 2, 1)  # Shape: (batch_size, V, L)
        x = self.cnn(x)  # Shape: (batch_size, 128, L)
        x = self.relu(x)
        x = x.mean(dim=2)  # Global average pooling: (batch_size, 128)
        uncertainty_emb = self.mlp(x)  # Shape: (batch_size, n_embd)
        return uncertainty_emb  # This will be appended as token L+1

    def _init_distribution_context(self, idx):
        """
        Initialize the distribution context with Dirac delta distributions for the tokens
        in the context window, and zeros for the remaining slots.
        """
        b, t = idx.size()
        context_size = min(t, self.distribution_context_size)

        # Initialize with zeros
        self.distribution_context = torch.zeros(b, self.distribution_context_size, self.config.vocab_size, device=idx.device)

        # Set Dirac deltas for the last 'context_size' tokens
        dirac_deltas = torch.zeros(b, context_size, self.config.vocab_size, device=idx.device)
        dirac_deltas.scatter_(2, idx[:, -context_size:].unsqueeze(-1), 1)

        self.distribution_context[:, -context_size:, :] = dirac_deltas

    def _update_distribution_context(self, new_distributions):
        """
        Update the distribution context by shifting left and appending new distributions.

        Args:
            new_distributions (Tensor): Tensor of shape (b, t_new, vocab_size)
        """
        b, t_new, vocab_size = new_distributions.size()
        assert vocab_size == self.config.vocab_size, "Vocabulary size mismatch in distribution context."

        if t_new >= self.distribution_context_size:
            # Replace entire context with the last 'distribution_context_size' distributions
            self.distribution_context = new_distributions[:, -self.distribution_context_size:, :]
        else:
            # Shift left by 't_new' and append new distributions
            shifted_context = self.distribution_context[:, t_new:, :] # shape (b, c - t_new, vocab_size)
            self.distribution_context = torch.cat([shifted_context, new_distributions], dim=1) # shape (b, c, vocab_size)

        if self.use_sparse_distributions:
            self.distribution_context = self._to_sparse(self.distribution_context)

    def _to_sparse(self, dense_tensor):
        # Convert dense tensor to sparse
        return dense_tensor.to_sparse()

    def get_distribution_context(self):
        """
        Retrieve the current distribution context.

        Returns:
            Tensor: Distribution context of shape (b, c, vocab_size)
        """
        if self.use_sparse_distributions and self.distribution_context is not None:
            return self.distribution_context.to_dense()
        return self.distribution_context

    def crop_block_size(self, block_size):
        """
        Model surgery to decrease the block size if necessary.

        Args:
            block_size (int): New block size to crop to.
        """
        assert block_size <= self.config.block_size, "New block size must be less than or equal to the current block size."
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens by iteratively sampling from the model's output distributions.

        Args:
            idx (Tensor): Tensor of shape (b, t) containing input token IDs.
            max_new_tokens (int): Number of tokens to generate.
            temperature (float): Scaling factor for logits.
            top_k (int, optional): Truncate distributions to top k tokens.

        Returns:
            Tensor: Generated token IDs of shape (b, t + max_new_tokens)
        """
        # Initialize distribution context with Dirac deltas for the input sequence
        if self.distribution_context is None:
            self._init_distribution_context(idx)

        for _ in range(max_new_tokens):
            # Adjust the sequence cropping to leave room for the uncertainty-aware token
            idx_cond = idx if idx.size(1) < self.config.block_size else idx[:, -(self.config.block_size-1):]
            # Forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

            # Update distribution context with the new distribution
            self._update_distribution_context(probs.unsqueeze(1))  # Shape: (b, 1, vocab_size)

        return idx
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
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
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx













