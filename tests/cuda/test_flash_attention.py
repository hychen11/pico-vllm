"""FlashAttention 2 path matches the eager attention (M6 step 4).

Two levels:
- one ``Attention`` layer on a mixed batch (fresh prompt, prefix-cache hit, decode);
- a tiny random Qwen3 driven through the scheduler, where one scheduler feeds two
  runners (eager and FA2) with their own KV caches. Both see the same batches and
  the eager runner's greedy tokens (teacher forcing), so every step's logits are
  comparable even when bf16 noise would flip a near-tie argmax.
"""

from __future__ import annotations

import pytest
import torch

from pico_vllm.core.paged_kv_cache import PagedKVCache
from pico_vllm.core.scheduler import Scheduler
from pico_vllm.layers.attention import HAS_FLASH_ATTN, Attention
from pico_vllm.sequence import Sequence
from pico_vllm.utils.context import reset_context, set_context
from pico_vllm.worker.model_runner import ModelRunner

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not (torch.cuda.is_available() and HAS_FLASH_ATTN), reason="requires CUDA + flash_attn"
    ),
]

BS = 256  # FA2 paged KV needs block_size % 256 == 0
DTYPE = torch.bfloat16


@pytest.fixture(autouse=True)
def _block_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Sequence, "block_size", BS)


def seq(num_tokens: int, block_table: list[int], cached: int) -> Sequence:
    s = Sequence(list(range(num_tokens)))
    s.block_table, s.num_cached_tokens = block_table, cached
    s.num_scheduled_tokens = num_tokens - cached
    return s


def test_attention_layer_mixed_batch() -> None:
    """Fresh 300-token prompt; 700 tokens with 512 cached; 1 query over 400 tokens."""
    torch.manual_seed(0)
    heads, kv_heads, head_dim = 16, 8, 128
    cache = PagedKVCache(1, 8, BS, kv_heads, head_dim, dtype=DTYPE, device="cuda")
    seqs = [seq(300, [3, 5], 0), seq(700, [0, 1, 6], 512), seq(400, [2, 7], 399)]

    # KV of the already-cached tokens, as earlier steps would have written it.
    runner = ModelRunner(None, cache, BS)  # type: ignore[arg-type]
    old = [seq(s.num_cached_tokens, s.block_table, 0) for s in seqs if s.num_cached_tokens]
    _, _, ctx_old = runner.prepare_inputs(old, is_prefill=True)
    n_old = ctx_old.slot_mapping.numel()
    cache.write(0, torch.randn(n_old, kv_heads, head_dim, dtype=DTYPE, device="cuda"),
                torch.randn(n_old, kv_heads, head_dim, dtype=DTYPE, device="cuda"),
                ctx_old.slot_mapping)

    _, _, ctx = runner.prepare_inputs(seqs, is_prefill=True)
    n = ctx.slot_mapping.numel()  # 300 + 188 + 1
    q = torch.randn(n, heads, head_dim, dtype=DTYPE, device="cuda")
    k = torch.randn(n, kv_heads, head_dim, dtype=DTYPE, device="cuda")
    v = torch.randn(n, kv_heads, head_dim, dtype=DTYPE, device="cuda")
    attn = Attention(heads, kv_heads, head_dim, head_dim**-0.5, layer_idx=0)

    outputs = {}
    for use_flash in (False, True):  # same writes to the same slots: idempotent
        ctx.use_flash = use_flash
        set_context(ctx)
        try:
            outputs[use_flash] = attn(q, k, v)
        finally:
            reset_context()

    assert outputs[True].shape == (n, heads, head_dim)
    torch.testing.assert_close(outputs[True], outputs[False], atol=2e-2, rtol=2e-2)


def test_model_steps_match_eager() -> None:
    transformers = pytest.importorskip("transformers")
    from pico_vllm.models.qwen3 import Qwen3ForCausalLM
    from pico_vllm.utils.loader import init_on

    torch.manual_seed(0)
    layers, kv_heads, head_dim, vocab = 2, 2, 64, 512
    config = transformers.Qwen3Config(
        vocab_size=vocab, hidden_size=256, intermediate_size=512, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=kv_heads, head_dim=head_dim,
        max_position_embeddings=2048, tie_word_embeddings=False,
    )
    with init_on("cuda", DTYPE):
        model = Qwen3ForCausalLM(config)
    model.eval()

    num_blocks = 16
    runners = {
        use_flash: ModelRunner(
            model, PagedKVCache(layers, num_blocks, BS, kv_heads, head_dim, dtype=DTYPE,
                                device="cuda"),
            BS, use_flash=use_flash,
        )
        for use_flash in (False, True)
    }
    # Budget 512 chunks the 600-token prompt into 512 + 88.
    scheduler = Scheduler(max_num_seqs=4, max_num_batched_tokens=512,
                          num_blocks=num_blocks, block_size=BS)
    g = torch.Generator().manual_seed(1)
    long_prompt = torch.randint(0, vocab, (600,), generator=g).tolist()
    scheduler.add(Sequence(long_prompt))
    scheduler.add(Sequence(torch.randint(0, vocab, (37,), generator=g).tolist()))

    shared = Sequence(long_prompt[:512] + torch.randint(0, vocab, (90,), generator=g).tolist())
    seen_chunk = seen_prefix_hit = False
    num_decode_steps = 0
    for step in range(12):
        if step == 1:  # long_prompt's first 2 blocks are hashed now
            scheduler.add(shared)
        seqs, is_prefill = scheduler.schedule()
        for s in seqs:
            done = s.num_cached_tokens + s.num_scheduled_tokens
            seen_chunk |= is_prefill and done < s.num_tokens
            seen_prefix_hit |= is_prefill and s is shared and s.num_cached_tokens == 512
        num_decode_steps += not is_prefill

        eager = runners[False].run(seqs, is_prefill)
        flash = runners[True].run(seqs, is_prefill)
        torch.testing.assert_close(flash.float(), eager.float(), atol=5e-2, rtol=5e-2)
        scheduler.postprocess(seqs, eager.argmax(-1).tolist())  # teacher forcing

    assert seen_chunk and seen_prefix_hit and num_decode_steps >= 5
