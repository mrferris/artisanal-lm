"""Tests for KV cache correctness in TransformerLM."""

import torch
import pytest

from lm.model.model import TransformerLM


@pytest.fixture
def model():
    """Create a small TransformerLM for testing."""
    torch.manual_seed(42)
    m = TransformerLM(
        d_model=64,
        vocab_size=100,
        context_length=32,
        num_layers=2,
        num_heads=4,
        d_ff=128,
        rope_theta=10000,
        device="cpu",
    )
    m.eval()
    return m


def test_encode_kv_shape(model):
    """encode_kv should return a list of (K, V) tuples."""
    tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    logits, kv = model.encode_kv(tokens)
    assert logits.shape == (1, 5, 100)  # [batch, seq_len, vocab_size]
    assert len(kv) == 2  # 2 layers
    for k, v in kv:
        # Shape: [batch, num_heads, seq_len, d_k]
        assert k.shape == (1, 4, 5, 16)
        assert v.shape == (1, 4, 5, 16)


def test_kv_cache_matches_full_forward(model):
    """forward(full_seq) must match forward(suffix, kv_cache=prefix_kv) at suffix positions."""
    full_tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    prefix_tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    suffix_tokens = torch.tensor([[6, 7, 8]], dtype=torch.long)

    with torch.no_grad():
        full_logits = model(full_tokens)
        _, prefix_kv = model.encode_kv(prefix_tokens)
        cached_logits = model.forward_with_kv(suffix_tokens, prefix_kv)

    full_suffix_logits = full_logits[:, 5:, :]
    torch.testing.assert_close(cached_logits, full_suffix_logits, atol=1e-5, rtol=1e-5)


def test_kv_cache_single_token_incremental(model):
    """Incremental single-token decoding should match full forward."""
    tokens = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)

    with torch.no_grad():
        full_logits = model(tokens)

        # Step 1: encode first two tokens
        first_two = torch.tensor([[10, 20]], dtype=torch.long)
        _, kv = model.encode_kv(first_two)

        # Step 2-4: forward remaining tokens one at a time
        for tok in [30, 40, 50]:
            token = torch.tensor([[tok]], dtype=torch.long)
            logits, kv = model.forward_incremental(token, kv)

    # Last token logits should match
    torch.testing.assert_close(logits[:, -1, :], full_logits[:, -1, :], atol=1e-5, rtol=1e-5)


def test_kv_cache_batched(model):
    """KV cache should work with batched inputs."""
    prefix = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    suffix = torch.tensor([[7, 8], [9, 10]], dtype=torch.long)
    full = torch.tensor([[1, 2, 3, 7, 8], [4, 5, 6, 9, 10]], dtype=torch.long)

    with torch.no_grad():
        full_logits = model(full)
        _, prefix_kv = model.encode_kv(prefix)
        cached_logits = model.forward_with_kv(suffix, prefix_kv)

    torch.testing.assert_close(cached_logits, full_logits[:, 3:, :], atol=1e-5, rtol=1e-5)


def test_no_kv_cache_unchanged(model):
    """Default forward (no KV args) should produce identical results to before."""
    tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    with torch.no_grad():
        logits1 = model(tokens)
        logits2 = model(tokens)
    torch.testing.assert_close(logits1, logits2)
