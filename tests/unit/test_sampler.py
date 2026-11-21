"""Tests for the sampler (M5)."""

import pytest
import torch

from pico_vllm.sampler import sample

VOCAB = 10


def gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_temperature_zero_is_argmax_for_every_row() -> None:
    logits = torch.randn(4, VOCAB, generator=gen())
    assert sample(logits, torch.zeros(4)) == logits.argmax(dim=-1).tolist()


def test_returns_python_ints_in_vocab_range() -> None:
    logits = torch.randn(5, VOCAB, generator=gen())
    out = sample(logits, torch.ones(5), gen(1))
    assert len(out) == 5
    assert all(isinstance(t, int) and 0 <= t < VOCAB for t in out)


def test_tiny_temperature_behaves_like_greedy() -> None:
    logits = torch.randn(6, VOCAB, generator=gen())
    out = sample(logits, torch.full((6,), 1e-4), gen(1))
    assert out == logits.argmax(dim=-1).tolist()


def test_same_seed_is_reproducible() -> None:
    logits = torch.randn(8, VOCAB, generator=gen())
    temps = torch.full((8,), 1.0)
    assert sample(logits, temps, gen(7)) == sample(logits, temps, gen(7))


def test_mixed_batch_greedy_rows_stay_argmax() -> None:
    """Rows 0 and 2 are greedy, row 1 samples; per-row temperatures, not one for all."""
    logits = torch.randn(3, VOCAB, generator=gen())
    out = sample(logits, torch.tensor([0.0, 1.0, 0.0]), gen(1))
    greedy = logits.argmax(dim=-1).tolist()
    assert out[0] == greedy[0] and out[2] == greedy[2]


def test_each_row_uses_its_own_temperature() -> None:
    """Identical rows: a near-zero temperature stays on the argmax, a huge one spreads out."""
    row = torch.tensor([0.0, 1.0, 2.0, 3.0])
    logits = row.repeat(2, 1)
    temps = torch.tensor([1e-4, 100.0])
    g = gen(0)
    cold, hot = set(), set()
    for _ in range(200):
        a, b = sample(logits, temps, g)
        cold.add(a)
        hot.add(b)
    assert cold == {3}
    assert len(hot) > 1


def test_sampling_follows_softmax_distribution() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    g = gen(0)
    n = 4000
    counts = torch.zeros(3)
    for _ in range(n):
        counts[sample(logits, torch.ones(1), g)[0]] += 1
    expected = torch.softmax(logits[0], dim=-1)
    torch.testing.assert_close(counts / n, expected, atol=0.03, rtol=0)


@pytest.mark.parametrize("temperature", [0.5, 2.0])
def test_temperature_scales_logits(temperature: float) -> None:
    """softmax(logits / T): lower T sharpens, higher T flattens the top token's share."""
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    g = gen(0)
    n = 4000
    top = sum(sample(logits, torch.tensor([temperature]), g)[0] == 2 for _ in range(n)) / n
    expected = torch.softmax(logits[0] / temperature, dim=-1)[2].item()
    assert abs(top - expected) < 0.03
