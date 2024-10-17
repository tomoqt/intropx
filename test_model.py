import torch
import pytest
from model import GPT, GPTConfig

@pytest.fixture
def gpt_model():
    config = GPTConfig(
        block_size=128,
        vocab_size=1000,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1,
        bias=True,
        use_sparse_distributions=False,
        distribution_context_size=64,
        d_hidden=64  # Make sure this matches the value in the original fixture
    )
    return GPT(config)

def test_model_initialization(gpt_model):
    assert isinstance(gpt_model, GPT)
    assert gpt_model.config.block_size == 128
    assert gpt_model.config.vocab_size == 1000
    assert gpt_model.distribution_context is not None
    assert gpt_model.distribution_context.shape == (1, 64, 1000)

def test_distribution_context_initialization(gpt_model):
    idx = torch.randint(0, 1000, (1, 32))
    gpt_model._init_distribution_context(idx)
    context = gpt_model.get_distribution_context()
    assert context.shape == (1, 64, 1000)
    assert torch.all(context[:, :32, :].sum(dim=-1) == 0)
    assert torch.all(context[:, 32:, :].sum(dim=-1) == 1)

def test_distribution_context_update(gpt_model):
    new_distributions = torch.rand(1, 10, 1000)
    new_distributions = new_distributions / new_distributions.sum(dim=-1, keepdim=True)
    gpt_model._update_distribution_context(new_distributions)
    context = gpt_model.get_distribution_context()
    assert context.shape == (1, gpt_model.distribution_context_size, 1000)
    assert torch.allclose(context[:, -10:, :], new_distributions)

def test_uncertainty_aware_token_generation(gpt_model):
    uncertainty_token = gpt_model.generate_uncertainty_token()
    assert uncertainty_token.shape == (1, gpt_model.config.n_embd)

def test_forward_pass(gpt_model):
    idx = torch.randint(0, 1000, (1, 64))
    logits, loss = gpt_model(idx)
    # The output should be a distribution over the vocabulary for the next token
    assert logits.shape == (1,1,1000)  # (batch_size, vocab_size)
    assert loss is None

def test_forward_pass_with_targets(gpt_model):
    idx = torch.randint(0, 1000, (1, 64))
    targets = torch.randint(0, 1000, (1, 64))
    logits, loss = gpt_model(idx, targets=targets)
    assert logits.shape == (1, 64, 1000)  # (batch_size, sequence_length, vocab_size)
    assert loss is not None
    assert loss.item() > 0

def test_generate(gpt_model):
    idx = torch.randint(0, 1000, (1, 10))
    generated = gpt_model.generate(idx, max_new_tokens=20)
    assert generated.shape == (1, 30)
    assert torch.all(generated[:, :10] == idx)

def test_crop_block_size(gpt_model):
    original_block_size = gpt_model.config.block_size
    gpt_model.crop_block_size(64)
    assert gpt_model.config.block_size == 64
    assert gpt_model.transformer.wpe.weight.shape[0] == 64

def test_causal_mask(gpt_model):
    idx = torch.randint(0, 1000, (1, 64))
    logits, _ = gpt_model(idx)
    assert logits.shape == (1, 1, 1000)
    # Check if the model is not attending to future tokens
    # This is an indirect way to test the causal mask
    # A more direct test would involve inspecting the attention weights

def test_distribution_embedder(gpt_model):
    distribution_context = torch.rand(1, 64, 1000)
    embedding = gpt_model.distribution_embedder(distribution_context)
    assert embedding.shape == (1, 128)

def test_sparse_distributions(gpt_model):
    gpt_model.use_sparse_distributions = True
    new_distributions = torch.rand(1, 10, 1000)
    new_distributions = new_distributions / new_distributions.sum(dim=-1, keepdim=True)
    gpt_model._update_distribution_context(new_distributions)
    context = gpt_model.get_distribution_context()
    assert context.is_sparse == False  # get_distribution_context returns dense tensor
    assert gpt_model.distribution_context.is_sparse == True

def test_generate_with_temperature_and_top_k(gpt_model):
    idx = torch.randint(0, 1000, (1, 10))
    generated = gpt_model.generate(idx, max_new_tokens=20, temperature=0.8, top_k=50)
    assert generated.shape == (1, 30)
    assert torch.all(generated[:, :10] == idx)

def test_model_parameters(gpt_model):
    total_params = sum(p.numel() for p in gpt_model.parameters())
    trainable_params = sum(p.numel() for p in gpt_model.parameters() if p.requires_grad)
    assert total_params > 0
    assert trainable_params > 0
    assert trainable_params <= total_params

def test_optimizer_configuration(gpt_model):
    optimizer = gpt_model.configure_optimizers(weight_decay=0.1, learning_rate=0.001, betas=(0.9, 0.999), device_type='cpu')
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 2

def test_mfu_estimation(gpt_model):
    mfu = gpt_model.estimate_mfu(fwdbwd_per_iter=1, dt=1.0)
    assert isinstance(mfu, float)
    assert mfu > 0
