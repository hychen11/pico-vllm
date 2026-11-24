"""CUDA Graph decode (M6 step 6) matches launching the same kernels one by one.

One scheduler feeds two runners with their own KV caches: one replays captured
graphs for decode steps, the other runs the model directly. Both see the same
batches and the plain runner's greedy tokens (teacher forcing), so every step's
logits are comparable. Sequences finish at different steps, so decode batch sizes
change and need padding.
"""

from __future__ import annotations

import pytest
import torch

from pico_vllm.core.paged_kv_cache import PagedKVCache
from pico_vllm.core.scheduler import Scheduler
from pico_vllm.kernels import kv_store
from pico_vllm.layers.attention import HAS_FLASH_ATTN
from pico_vllm.sequence import Sequence
from pico_vllm.worker.model_runner import ModelRunner

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not (torch.cuda.is_available() and HAS_FLASH_ATTN and kv_store.HAS_TRITON),
        reason="requires CUDA + flash_attn + triton",
    ),
]

BS = 256
DTYPE = torch.bfloat16


@pytest.fixture(autouse=True)
def _block_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Sequence, "block_size", BS)


def test_kv_store_skips_slot_minus_one() -> None:
    k = torch.randn(3, 2, 64, dtype=DTYPE, device="cuda")
    v = torch.randn(3, 2, 64, dtype=DTYPE, device="cuda")
    k_cache = torch.zeros(2, 4, 2, 64, dtype=DTYPE, device="cuda")
    v_cache = torch.zeros_like(k_cache)
    slots = torch.tensor([5, -1, 2], device="cuda")
    kv_store.store_kvcache(k, v, k_cache, v_cache, slots)
    flat_k, flat_v = k_cache.view(8, 2, 64), v_cache.view(8, 2, 64)
    assert torch.equal(flat_k[5], k[0]) and torch.equal(flat_k[2], k[2])
    assert torch.equal(flat_v[5], v[0]) and torch.equal(flat_v[2], v[2])
    untouched = [i for i in range(8) if i not in (2, 5)]
    assert not flat_k[untouched].any() and not flat_v[untouched].any()  # -1 wrote nothing


def test_graph_decode_matches_plain_decode(monkeypatch: pytest.MonkeyPatch) -> None:
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

    num_blocks, max_model_len = 16, 1024

    def runner() -> ModelRunner:
        cache = PagedKVCache(layers, num_blocks, BS, kv_heads, head_dim, dtype=DTYPE,
                             device="cuda")
        return ModelRunner(model, cache, BS, use_flash=True)

    plain, graphed = runner(), runner()
    graphed.capture_cudagraphs(max_num_seqs=6, max_model_len=max_model_len)
    assert graphed.graph_batch_sizes == [1, 2, 4, 6]  # max_num_seqs itself is a size

    replayed: list[int] = []
    replay = graphed._replay

    def spy(input_ids, positions, context):  # noqa: ANN001, ANN202
        replayed.append(context.block_table.shape[0])
        return replay(input_ids, positions, context)

    monkeypatch.setattr(graphed, "_replay", spy)

    scheduler = Scheduler(max_num_seqs=6, max_num_batched_tokens=2048,
                          num_blocks=num_blocks, block_size=BS)
    g = torch.Generator().manual_seed(1)
    lengths = [300, 37, 5, 260, 90, 12]  # some cross a block boundary while decoding
    seqs_all = [Sequence(torch.randint(0, vocab, (n,), generator=g).tolist()) for n in lengths]
    for s in seqs_all:
        scheduler.add(s)
    # Finish seqs one by one: decode batch sizes go 6, 5 (padded to 6), 4, 3 (to 4), ...
    finish_at = {3: seqs_all[0], 5: seqs_all[1], 7: seqs_all[2], 9: seqs_all[3], 11: seqs_all[4]}
    for step in range(14):
        if step in finish_at and not finish_at[step].is_finished:
            scheduler.finish(finish_at[step])
        if scheduler.is_finished():
            break
        seqs, is_prefill = scheduler.schedule()
        ref = plain.run(seqs, is_prefill)
        out = graphed.run(seqs, is_prefill)
        torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
        scheduler.postprocess(seqs, ref.argmax(-1).tolist())  # teacher forcing

    assert set(replayed) >= {6, 5, 4, 3, 2, 1}  # e.g. 3 replays bs=4 with one padding row
