# pico-vLLM

`pico-vLLM` is a small, test-driven LLM inference engine for learning the core ideas behind vLLM: request lifecycle, continuous batching, paged KV-cache allocation, prefix caching, chunked prefill, and a CUDA fast path. It runs dense **Qwen3** checkpoints from a local Hugging Face directory, on a single GPU or on the CPU.

The scheduler and block manager follow the design of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm). pico-vLLM adds a CPU-runnable eager path, an explicit `PagedKVCache` object, greedy sampling, and a test suite (184 tests), so every component can be checked in isolation. See [docs/design-vs-nano-vllm.md](docs/design-vs-nano-vllm.md) for the full comparison.

The milestone-by-milestone plan is in [ROADMAP.md](ROADMAP.md).

## Recommended reading

- [AIInfraGuide — AI Infra 全栈知识库](https://caomaolufei.github.io/AIInfraGuide/): a Chinese guide to the AI infrastructure stack, covering CUDA kernel optimization, distributed training, inference optimization, and performance analysis. Useful background for the paged KV cache, FlashAttention, and CUDA Graph work in this project.

## Status

| Milestone | Content | Status |
| --- | --- | --- |
| M0–M2 | project setup, `Sequence`, `SamplingParams`, `BlockManager`, `Scheduler` | done |
| M3 | native Qwen3 model and safetensors loader | done |
| M4 | `PagedKVCache` and `ModelRunner` | done |
| M5 | offline `LLM.generate` API | done |
| M6 | CUDA fast path | in progress: GPU + bf16 and the Triton KV-store kernel are done; FlashAttention, automatic KV sizing, and CUDA Graph are next |

Measured on an RTX 5090 with the fixed benchmark workload, M6 so far raises prefill from 437 to about 49,000 tok/s (112×) and decode from 62 to about 275 tok/s (4.4×) over the M5 CPU baseline. Decode is currently limited by the eager attention loop; see [docs/m6-optimizations.md](docs/m6-optimizations.md).

## Features

- Qwen3 decoder-only models: RMSNorm, RoPE, grouped-query attention, Q/K normalization, SwiGLU; fused QKV and gate/up projections.
- Continuous batching under `max_num_seqs` and `max_num_batched_tokens` budgets.
- Paged KV cache with block tables and ref-counted blocks.
- Prefix caching: full blocks are hashed (xxhash, chained) and shared between requests with the same prefix.
- Chunked prefill: a prompt longer than the step's remaining token budget is split across steps.
- Preemption: when blocks run out during decode, the newest sequence is freed and requeued.
- Sampling: greedy (`temperature=0`) and temperature sampling, mixed freely within one batch; EOS, `max_tokens`, `ignore_eos`.
- CUDA fast path: bf16 by default on GPU, Triton KV-store kernel; eager fallback on CPU.

Not in scope yet: top-k/top-p, tensor parallelism, quantization, LoRA, multimodal and MoE models, speculative decoding, HTTP serving.

## Installation

From PyPI:

```bash
pip install "pico-vllm[model]"          # + transformers/safetensors to load Qwen3 checkpoints
pip install "pico-vllm[model,cuda]"     # + Triton for the CUDA fast path (Linux)
```

For development, install [uv](https://docs.astral.sh/uv/getting-started/installation/) once, then from the repository root:

```bash
uv sync --extra dev --extra model --extra cuda
uv run pytest -q
```

`uv run` uses the project's `.venv` even if your shell shows Conda `(base)`.

FlashAttention 2 is not a declared dependency yet (it is needed from M6 step 4). It has to be built for your GPU, for example from the [flash-attention](https://github.com/Dao-AILab/flash-attention) source. Note that `uv sync` removes packages that `pyproject.toml` does not declare, so reinstall it afterwards or use `uv sync --inexact`.

## Usage

Download a checkpoint and generate:

```bash
uv run hf download Qwen/Qwen3-0.6B --local-dir ~/models/Qwen3-0.6B
uv run python example.py ~/models/Qwen3-0.6B
```

```python
from pico_vllm.llm import LLM
from pico_vllm.sampling_params import SamplingParams

llm = LLM("/path/to/Qwen3-0.6B", max_model_len=1024)   # device="auto", dtype="auto"
outputs = llm.generate(
    ["The capital of France is", "def fibonacci(n):"],
    [SamplingParams(temperature=0.0, max_tokens=32),     # greedy
     SamplingParams(temperature=0.6, max_tokens=64)],
)
print(outputs[0]["text"], outputs[0]["token_ids"])
```

`LLM` accepts the `Config` fields as keyword arguments: `max_num_seqs`, `max_num_batched_tokens`, `max_model_len`, `kvcache_block_size`, `kvcache_capacity_bytes`, `gpu_memory_utilization`, `device` (`"auto"`, `"cuda:0"`, `"cpu"`), and `dtype` (`"auto"`, `"bfloat16"`, `"float16"`, `"float32"`). With `"auto"`, the engine uses CUDA with the checkpoint dtype (or bf16) when a GPU is available, and CPU with fp32 otherwise. On CUDA, leaving `kvcache_capacity_bytes` unset sizes the KV cache after a warm-up step so that the engine uses at most `gpu_memory_utilization` (default 0.9) of the GPU, counting memory held by other processes; on CPU it defaults to 2 GiB.

Benchmark with a randomly initialized model of Qwen3-0.6B shape (no download needed):

```bash
uv run python -m pico_vllm.benchmark --device cuda:0
uv run python -m pico_vllm.benchmark --device cuda:0 --no-triton   # eager KV writes
```

## Repository layout

```text
.
├── pyproject.toml
├── README.md
├── ROADMAP.md
├── example.py                # generate with a local Qwen3 checkpoint
├── docs/                     # optimization log, experiment reports, design notes
├── src/pico_vllm/
│   ├── llm.py                # offline LLM.generate API
│   ├── config.py             # Config, device/dtype resolution
│   ├── sampling_params.py
│   ├── sequence.py
│   ├── sampler.py
│   ├── benchmark.py          # fixed throughput workload
│   ├── cli.py
│   ├── core/                 # block_manager, scheduler, paged_kv_cache
│   ├── engine/               # LLMEngine
│   ├── worker/               # ModelRunner
│   ├── models/               # Qwen3
│   ├── layers/               # attention, linear, layernorm, rotary_embedding, activation
│   ├── kernels/              # Triton KV-store kernel
│   └── utils/                # step Context, weight loader
└── tests/
    ├── unit/
    ├── integration/
    └── cuda/
```

## Documentation

| Document | Content |
| --- | --- |
| [ROADMAP.md](ROADMAP.md) | milestones M0–M11 and their tests |
| [docs/m6-optimizations.md](docs/m6-optimizations.md) | each M6 optimization: what improved, how, and by how much |
| [docs/experiments/m6-triton-kv-store.md](docs/experiments/m6-triton-kv-store.md) | Triton KV-store experiment and environment |
| [docs/design-vs-nano-vllm.md](docs/design-vs-nano-vllm.md) | M1–M5 design compared with nano-vllm |
| [docs/concepts/request-lifecycle-prefix-cache.md](docs/concepts/request-lifecycle-prefix-cache.md) | one request step by step: scheduling, prefix-cache hit, model inputs, attention (traced from real runs) |
| [docs/concepts/attention-context.md](docs/concepts/attention-context.md) | the attention fields in `Context` (`cu_seqlens_q/k`, `max_seqlen_q/k`, `block_table`): why q holds only new tokens while KV covers the whole sequence |
| [docs/concepts/kv-cache-sizing.md](docs/concepts/kv-cache-sizing.md) | how the KV cache is sized from GPU memory: the budget formula, the prefill + decode warm-up that measures the activation peak, and why the engine runs `gc.collect()` before loading weights |
| [docs/concepts/decode-attention-and-cuda-graph.md](docs/concepts/decode-attention-and-cuda-graph.md) | decode FlashAttention inputs and why decode steps can be captured as CUDA graphs |

## Qwen3 forward path

The model reads its settings from the checkpoint's `config.json` (through `transformers.AutoConfig`); the scheduler and cache code never depend on Hugging Face objects.

```text
token ids → embedding
  → repeated [RMSNorm → GQA attention with RoPE → residual
              RMSNorm → SwiGLU MLP → residual]
  → final RMSNorm → LM head → logits
```

- GQA means `num_key_value_heads` may be smaller than `num_attention_heads`.
- RoPE positions are identical in prefill and decode paths.
- Q/K normalization, projection bias, and RoPE settings come from the checkpoint rather than hard-coded assumptions. Long-context RoPE scaling is a later extension.
- Real-model tests receive a local checkpoint through `PICO_VLLM_QWEN_MODEL=/absolute/path` and skip if it is absent.

## Component boundaries

| Component | Owns | Must not own |
| --- | --- | --- |
| `SamplingParams` | validation of generation settings | scheduling |
| `Sequence` | one request's tokens, progress counters, block table | physical K/V storage |
| `Scheduler` | waiting/running selection under budgets, preemption | model execution |
| `BlockManager` | block allocation, prefix hashing, ref counts | request priority, tensors |
| `PagedKVCache` | the K/V tensors, `write` / `read` by slot | deciding who owns a block |
| `ModelRunner` | turning a batch into model inputs and a step `Context` | request lifecycle |
| `LLMEngine` | one scheduler + model step, stop rules (EOS, `max_tokens`) | HTTP protocol |

A sequence moves `WAITING → RUNNING → FINISHED`. Preemption moves it from `RUNNING` back to `WAITING` and frees its blocks.

## How the components fit together

Two independent halves meet only through `Sequence.block_table`, a plain list of block ids:

- the **bookkeeping half** (CPU, Python objects) decides *which* block ids a sequence owns;
- the **storage half** (CPU or GPU tensors) holds the actual K/V values at those ids.

`BlockManager` never touches a tensor, and `PagedKVCache` never decides who owns a block. In particular, `BlockManager` does **not** call `PagedKVCache`.

```text
                      LLMEngine (M5)  — calls both halves once per step
                     /                \
   bookkeeping (CPU)                    storage + compute (device)
   ─────────────────                    ──────────────────────────
   Scheduler (M2)                       ModelRunner (M4)
     │ owns                               │ owns            │ owns
     ▼                                    ▼                 ▼
   BlockManager (M2)                    Qwen3ForCausalLM   PagedKVCache (M4)
     · free queue, ref_count, hash        (M3)               k_cache / v_cache
     · writes block ids into ─┐           │ each layer      [L, N, B, H, D]
                              │           ▼                    ▲
                              ▼         Attention (M4) ────────┘
                     Sequence.block_table   reads Context:     write()/read()
                     e.g. [3, 8]            slot_mapping,
                              │             block_tables, ...
                              │                ▲
                              └──► ModelRunner.prepare_inputs builds Context
                                   (set_context / get_context, utils/context.py)
```

Who calls whom:

| Caller | Callee | Why |
| --- | --- | --- |
| `LLMEngine.step` | `Scheduler.schedule` | pick the seqs for this step; blocks are reserved here |
| `Scheduler` | `BlockManager.can_allocate / allocate / may_append / deallocate / hash_blocks` | manage block ids, write `seq.block_table` |
| `LLMEngine.step` | `ModelRunner.run(seqs, is_prefill)` | run the model on the scheduled batch |
| `ModelRunner.prepare_inputs` | reads `seq.block_table`, `num_cached_tokens`, ... | turn block ids into `slot_mapping` and a `Context` |
| `ModelRunner.run` | `set_context` → `model(...)` → `reset_context` | expose the step's metadata to every layer |
| `Attention.forward` | `get_context().kv_cache.write / read` | store this step's K/V and read the full context |
| `LLMEngine.step` | `Scheduler.postprocess` | append sampled tokens, hash full blocks |
| `LLMEngine.step` | `Scheduler.finish` | free the blocks of seqs that hit EOS or `max_tokens` |

One step, end to end:

```text
1. Scheduler.schedule()        BlockManager gives seq A blocks [3, 8]   (ids only)
2. ModelRunner.prepare_inputs  pos 0..4 of A → slots 12,13,14,15,32     (ids → slots)
3. Attention, every layer      kv_cache.write(slots, k, v)              (values land in k_cache)
                               kv_cache.read(block_table=[3, 8], len=5)
4. Scheduler.postprocess       append token, hash full blocks
   LLMEngine / finish          EOS or max_tokens? → deallocate          (ids return to free queue;
                                                                          k_cache is left untouched)
```

`PagedKVCache` is allocated once at startup with `num_blocks` equal to the `BlockManager`'s `num_blocks`; the two must agree, since block id `i` in one is index `i` in the other.

## Paged KV cache

For `block_size = 4`, ten tokens occupy three logical blocks:

```text
logical token positions:  0 1 2 3 | 4 5 6 7 | 8 9
logical block index:          0   |     1   |  2
block table:              [12, 7, 19]
physical block id:          12    7    19
```

`Sequence.block_table` maps logical blocks to physical blocks. Sequences with the same prompt prefix share the physical blocks of that prefix. Only **full** blocks are hashed and shared, and a full block is never written again, so writes never land in a shared block and no copy-on-write is needed.

Core invariants:

- A physical block contains exactly `block_size` token positions.
- `num_cached_tokens + num_scheduled_tokens <= len(sequence.token_ids)`.
- A block returns to the free queue only when its reference count becomes zero.
- A scheduler step does not exceed `max_num_batched_tokens`.

## Test and commit contract

For every milestone:

1. Write a failing, focused test.
2. Implement the smallest correct behavior.
3. Run the milestone tests and then all tests.
4. Make one Git commit only after all tests pass.

```bash
uv run pytest tests/unit -q
uv run pytest tests/integration -q
PICO_VLLM_QWEN_MODEL=~/models/Qwen3-0.6B uv run pytest -q    # including real-model tests
```

Use a seeded `torch.Generator` whenever a test samples randomly. Unit tests must never need CUDA, an internet connection, or downloaded weights. Real Qwen parity tests have both `integration` and `model` markers; CUDA-only tests additionally have `cuda` and skip without a GPU.

## License

[Apache-2.0](LICENSE)
