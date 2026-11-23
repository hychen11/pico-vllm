# pico-vLLM implementation roadmap

This project has two tracks:

1. **Track A — core replica:** reproduce the essential offline, single-GPU execution path of the reference nano-vLLM project for **Qwen3**.
2. **Track B — hardening:** add the tests, API surface, observability, and optional optimizations that make the implementation easier to evolve.

Track A is the immediate goal. It should be small, focused, and functional before the project expands. Each milestone is one Git commit. Before committing, run its listed test command and then `uv run pytest -q`.

## Track A — Qwen3 core replica

### M0 — Package baseline

**Goal:** Make the code importable and give every later milestone a stable test command.

**Implement**

- `src/pico_vllm/__init__.py` and a minimal `src/pico_vllm/cli.py`;
- `.gitignore` for virtual environments, Python caches, IDE files, coverage, and local model directories;
- `tests/test_package.py` and `tests/conftest.py`;
- keep Qwen loading dependencies optional: `uv sync --extra dev --extra model` is only needed from M3 onward.

**Acceptance**

```bash
uv sync --extra dev
uv run pytest -q
```

**Commit:** `chore: bootstrap nano-vllm project`

### M1 — Minimal request state and sampling

**Goal:** Represent one generation request with only the controls needed by the replica.

**Files**

```text
src/pico_vllm/sampling_params.py
src/pico_vllm/sequence.py
tests/unit/test_sampling_params.py
tests/unit/test_sequence.py
```

**Implement**

- `SamplingParams(temperature, max_tokens, ignore_eos)`; keep top-k/top-p and stop strings for Track B;
- `SequenceStatus`: `WAITING`, `RUNNING`, `FINISHED`;
- `Sequence`: prompt token ids, current total/last token, `num_cached_tokens`, `num_scheduled_tokens`, `block_table`, and status;
- append a token while preserving the original prompt length. The engine handles EOS and `max_tokens` completion in M5.

**Tests:** parameter validation; unique sequence ids; independent prompt copies and block tables; initial scheduler/cache state; token and last-token accounting after appends. Completion slicing and block helpers are added when their consumers are implemented in M2; EOS and length completion are tested with the engine in M5.

**Commit:** `feat: add minimal sequence state and sampling params`

### M2 — Block manager, prefix cache, and scheduler

**Goal:** Reproduce the distinctive vLLM behavior before connecting a real model: physical KV blocks are allocated by a scheduler, reused by matching full prompt blocks, and returned safely.

**Files**

```text
src/pico_vllm/core/block_manager.py
src/pico_vllm/core/scheduler.py
tests/unit/test_block_manager.py
tests/unit/test_scheduler.py
```

**Implement**

- fixed-size physical `Block` objects with `block_id`, `ref_count`, hash, and token ids;
- `Sequence.block_table` maps logical blocks to physical blocks;
- chained `xxhash` hash for each **full** prompt block, enabling automatic prefix reuse;
- `waiting` and `running` queues, `max_num_seqs`, `max_num_batched_tokens`, chunked prefill, decode scheduling, release, and basic preemption;
- `Scheduler.schedule()` returns a batch description and never calls model code.

**Tests:** exact/cross-block allocation; no state change after failed allocation; refcounts under shared prompts; identical prefix hits and different prefix misses; release returns blocks once; token budget is respected; waiting-to-running transition; a completed sequence is never rescheduled.

**Commit:** `feat: add paged block manager and scheduler`

### M3 — Qwen3 layers and checkpoint loader

**Goal:** Implement the native dense Qwen3 forward pass; do not optimize attention yet.

**Files**

```text
src/pico_vllm/config.py
src/pico_vllm/layers/{activation,layernorm,rotary_embedding,linear}.py
src/pico_vllm/models/qwen3.py
src/pico_vllm/utils/loader.py
tests/unit/test_rms_norm.py
tests/unit/test_rope.py
tests/unit/test_qwen3_shapes.py
```

**Implement**

- read a local Qwen3 `config.json` through Hugging Face `AutoConfig`;
- RMSNorm, RoPE, Q/K normalization where required by Qwen3 configuration, GQA tensor shapes, SwiGLU MLP, residual structure, embedding, and LM head;
- packed weight loading: Q/K/V checkpoint tensors into QKV projection, and gate/up tensors into the merged MLP projection;
- run a small eager PyTorch attention reference only. This phase is about correctness, not FlashAttention performance.

**Tests:** RMSNorm and RoPE small-tensor numerical tests; Q/K/V and gate/up weight-name mapping; GQA output shapes; a tiny local/random Qwen3 config completes a forward pass; tied embeddings share the expected weight.

**Commit:** `feat: add native qwen3 model and loader`

### M4 — Paged KV runner

**Goal:** Connect Qwen3 attention to the scheduler's block tables.

**Files**

```text
src/pico_vllm/core/paged_kv_cache.py
src/pico_vllm/worker/model_runner.py
tests/unit/test_paged_kv_cache.py
tests/integration/test_runner_fake_model.py
```

**Implement**

- allocate KV tensors indexed by `[layer, physical_block, token_offset, kv_head, head_dim]`;
- prepare prefill tensors: flattened `input_ids`, positions, cumulative sequence lengths, slot mapping, and optional block tables for prefix cache;
- prepare decode tensors: final token, absolute position, context length, and block tables;
- write K/V into the physical slots; read them through the block table for the eager reference attention path;
- estimate usable KV blocks from configured capacity first; dynamic GPU-memory sizing is deferred to M6.

**Tests:** slot mapping across physical blocks; K/V isolation across sequences; shared prefix read path; copy-on-write or explicit safe handling of a shared partial tail block; fake runner prefill/decode metadata.

**Commit:** `feat: add paged kv model runner`

### M5 — Offline Qwen3 LLM API

**Goal:** Deliver the first usable core replica.

**Files**

```text
src/pico_vllm/engine/llm_engine.py
src/pico_vllm/llm.py
src/pico_vllm/tokenizer.py
example.py
tests/integration/test_llm_engine.py
```

**Implement**

- `LLM(model_path, max_model_len, max_num_seqs, max_num_batched_tokens)`;
- `LLM.generate(prompts, sampling_params)` and token-id prompt support;
- `add_request()`, `step()`, and `is_finished()` internally;
- local `AutoTokenizer` use, EOS propagation, greedy sampling and temperature sampling;
- an example that runs a local `Qwen/Qwen3-0.6B` checkpoint.

**Tests:** two fake-model requests interleave correctly; EOS/length cleanup releases blocks; a local-model smoke test is skipped unless `PICO_VLLM_QWEN_MODEL` is set; when provisioned, greedy next-token output matches Hugging Face for a fixed prompt.

**Commit:** `feat: add offline qwen3 generation api`

### M6 — CUDA fast path (optional, after M5 works)

**Goal:** Reproduce the reference implementation's performance path without changing M5 semantics.

**Implement**

- optional `cuda` dependency group for Triton and FlashAttention;
- Triton KV-store kernel;
- FlashAttention varlen prefill and KV-cache decode paths;
- automatic KV-block count based on GPU free memory after warm-up;
- optional CUDA Graph capture for common decode batch sizes.

**Tests:** CUDA-only parity of eager and fast paths; slot mapping for the Triton kernel; a fixed benchmark workload. Mark every test `cuda` and skip on machines without CUDA.

**Commit:** `perf: add cuda paged-attention fast path`

**Track A definition of done:** a local Qwen3 model can generate text through `LLM.generate`; several requests share one paged KV pool; identical full prompt blocks are reused; and M0–M5 tests pass on CPU except explicitly marked local-model/CUDA tests.

## Track B — hardening and expansion

Start only after Track A is working.

### M7 — Test coverage and correctness hardening

Add exhaustive state-machine, allocation-failure, abort, long-prompt, sampling-boundary, model-parity, and regression tests. Add property-style randomized block-manager tests. This is the milestone that turns the compact replica into a safe codebase to modify.

**Commit:** `test: harden core engine invariants`

### M8 — Tensor parallelism

Moved ahead of the serving milestones: it changes the model layers and the runner, so it should land while M7's tests guard them and before the serving features build on top.

**Files**

```text
src/pico_vllm/distributed.py
src/pico_vllm/layers/{linear,embed_head}.py
src/pico_vllm/models/qwen3.py
src/pico_vllm/worker/model_runner.py
src/pico_vllm/engine/llm_engine.py
tests/unit/test_tp_layers.py
tests/cuda/test_tensor_parallel.py
```

**Implement**

- `Config.tensor_parallel_size` (default 1; must divide the attention-head and KV-head counts);
- one process per GPU with `torch.distributed` and NCCL; rank 0 owns the scheduler and sampler, the other ranks only run the model; per-step inputs are broadcast from rank 0;
- column-parallel QKV and gate/up projections and row-parallel `o_proj`/`down_proj` with an all-reduce; vocab-parallel embedding and LM head with a gather of logits on rank 0;
- the loader shards each checkpoint tensor by rank, including the packed Q/K/V and gate/up layouts;
- each rank keeps a KV cache for its own KV heads only; the block manager is unchanged and shared through rank 0;
- `tensor_parallel_size == 1` keeps today's single-process path with no `torch.distributed` dependency.

**Tests:** sharded linear/embedding layers reproduce the unsharded output on CPU (gloo, 2 processes); loader shard shapes; with 2 GPUs, TP=2 logits match TP=1 step by step (teacher forcing, as in M6 step 4). Skip GPU tests when fewer than 2 GPUs are visible.

**Hardware note:** the dev machine pairs an RTX 5090 with an RTX 4060 Laptop GPU (8 GB, sm_89). That is enough to check TP correctness, but not to show a speedup: the slower card and PCIe/host-memory NCCL traffic bound each step, and the sm_120-only FlashAttention wheel does not run on the 4060, so TP runs there use eager attention.

**Commit:** `feat: add tensor parallelism`

### M9 — Better serving semantics

Add top-k, top-p, stop-token sets, multiple completion candidates, streaming events, request ids, and fairer scheduling policy. Keep these additions separate from the baseline replica so their effects are measurable.

**Commit:** `feat: extend serving semantics`

### M10 — HTTP API

Add `GET /health`, `POST /v1/completions`, then optional chat completions and server-sent event streaming using FastAPI. Invalid requests must not disturb active generation.

**Commit:** `feat: expose openai-compatible api`

### M11 — Observability and scale-out

Add metrics, reproducible benchmarks, and trace/profiling hooks, and only then more advanced features such as quantization or LoRA. Tensor parallelism moved forward to M8.

**Commit:** `perf: add observability and profiling hooks`

## Size target

| Scope | Production code | Tests | Total |
| --- | ---: | ---: | ---: |
| Track A M0–M5, eager Qwen3 core | 1,500–2,500 LOC | 600–1,200 LOC | 2,100–3,700 LOC |
| M6 CUDA fast path | +300–800 LOC | +150–350 LOC | +450–1,150 LOC |
| Track B M7–M11 | +1,500–3,000 LOC | +1,500–2,500 LOC | +3,000–5,500 LOC |

The core replica therefore stays near the reference project's scale. The larger estimates apply only after deliberately adding a robust test suite, network API, observability, and additional serving features.
