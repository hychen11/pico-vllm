# M6 第 6 步实现指南：CUDA Graph（decode）

这是一份动手清单，只说每一步改哪里、为什么改、怎么验证，不给完整实现。原理（为什么 decode 能录、prefill 录不了，batch 档位怎么定）见 [concepts/decode-attention-and-cuda-graph.md](concepts/decode-attention-and-cuda-graph.md)，建议先读第 6、8、9 节。

**目标**：纯 decode 步骤不再由 CPU 逐个发起 28 层的 kernel，而是把整步录成 CUDA Graph，之后每步只拷入少量输入再回放。prefill 和超出档位的 decode 仍走现有的 eager / varlen 路径，输出语义不变。

**参考数字**（5090，`--model ~/models/Qwen3-0.6B`，benchmark 默认负载）：

| 阶段 | decode tok/s |
| --- | ---: |
| 第 4 步（varlen） | ≈ 2494 |
| 只把 decode 换成 `flash_attn_with_kvcache` | ≈ 2700 |
| 加上 CUDA Graph | ≈ 7000+ |

prefill 应保持在约 106k tok/s，不受影响。

## 总览

| 步骤 | 改动 | 文件 |
| --- | --- | --- |
| 0 | 记录基线 | — |
| 1 | KV 写入 kernel 跳过 `slot == -1` | `kernels/kv_store.py` |
| 2 | Context 增加 GPU 张量 `cache_seqlens` | `utils/context.py`、`worker/model_runner.py` |
| 3 | decode 改用 `flash_attn_with_kvcache` | `layers/attention.py` |
| 4 | 录制 graph：档位、固定缓冲区、预热、共享显存池 | `worker/model_runner.py` |
| 5 | 回放：拷入缓冲区、补齐、回退条件 | `worker/model_runner.py` + 测试 |
| 6 | 开关与接线：`enforce_eager`、engine、benchmark | `config.py`、`engine/llm_engine.py`、`benchmark.py` |
| 7 | 测量并写文档 | `docs/m6-optimizations.md` |

第 1–3 步每一步都可以单独跑测试、单独成立（不录 graph 也是对的）；第 4–5 步才真正引入 graph。

---

## 第 0 步：准备和基线

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # 确认 5090 空闲
export PICO_VLLM_QWEN_MODEL=~/models/Qwen3-0.6B
uv run pytest -q                                               # 全部通过
uv run python -m pico_vllm.benchmark --device cuda:0 --model ~/models/Qwen3-0.6B   # 跑 2 遍取平均
```

记下 prefill / decode 两个数，后面每一步都和它比。

**实测基线**（2026-11-23，main `7c69983`，5090 空闲，`uv run pytest -q` 209 passed）：

| 轮次 | prefill（tok/s） | decode（tok/s） | 总计（tok/s） |
| --- | ---: | ---: | ---: |
| 1 | 105667 | 2455 | 8426 |
| 2 | 106680 | 2420 | 8315 |
| 平均 | 106173 | 2437 | 8371 |

和总表第 4 行（106869 / 2494）的差别在 2.5% 以内，属于正常波动。

## 第 1 步：Triton kernel 跳过 `slot == -1`

**为什么**：graph 的 batch 是固定档位（比如实际 3 条，用 bs = 4 的 graph），多出来的空位行也会经过 KV 写入。空位的 `slot_mapping` 填 -1，kernel 必须什么都不写，否则会写坏别人的 KV。

**改哪里**：`kernels/kv_store.py` 的 `_store_kvcache_kernel`，在 `slot = tl.load(...)` 之后加一个提前返回。顺便更新 `store_kvcache` 的 docstring：`-1 = 跳过该行`。

**注意**：

- eager 写入（`PagedKVCache.write` 的 `index_put_` 路径）**不会**跳过 -1，Python 索引 -1 会写到整个 cache 的最后一个 slot。所以 CUDA Graph 只能和 Triton 写入一起用，第 4 步里要 `assert kv_cache.use_triton`，benchmark 里 `--no-triton` 时也不要开 graph。
- 判断写在 kernel 里（GPU 上），不能写成 Python 的 `if`，因为 graph 回放时 Python 代码不会再执行。

**验证**：加一个小测试，`slot_mapping` 里混入 -1，确认对应行没有写入、其他行正常写入，cache 的最后一个 slot 保持原值。

## 第 2 步：Context 增加 `cache_seqlens`

**为什么**：`flash_attn_with_kvcache` 用 int32 GPU 张量 `cache_seqlens` 告诉 kernel 每条序列多长。现有的 `context_lens` 是 Python list，`max_seqlen_k` 是 Python int；它们每步都变，graph 里只能放"形状固定、内容可变"的 GPU 张量。

**改哪里**：

- `utils/context.py`：`Context` 加字段 `cache_seqlens: torch.Tensor | None = None`，注释写清楚形状 `[num_seqs]`、int32、内容等于 `context_lens`。
- `worker/model_runner.py` 的 `prepare_inputs`：`cache_seqlens=as_tensor(context_lens, torch.int32)`。

**验证**：现有测试全部通过（这一步只是多建一个张量，不改行为）。

## 第 3 步：decode 改用 `flash_attn_with_kvcache`

**为什么**：`flash_attn_varlen_func` 需要 `max_seqlen_q / max_seqlen_k` 两个 Python int，录进 graph 后就被"冻结"成录制时的值。`flash_attn_with_kvcache` 只依赖 GPU 张量，可以录。

**改哪里**：`layers/attention.py` 的 `forward`，在写完 KV 之后、varlen 分支之前，加一个 `use_flash and not is_prefill` 的分支：

- `q` 是 `[num_seqs, num_heads, D]`（decode 每条 1 个 token），而接口要 `[batch, seqlen_q, heads, D]`，所以 `q.unsqueeze(1)` 进去、`squeeze(1)` 出来。
- `k_cache / v_cache` 传这一层的整个池：`kv_cache.k_cache[self.layer_idx]`。
- 传 `cache_seqlens=context.cache_seqlens`、`block_table=context.block_table`、`softmax_scale`、`causal=True`。
- 不要再传 `k`、`v` 参数（KV 已经由上面的 `kv_cache.write` 写进去了，`cache_seqlens` 已包含本步这个 token）。
- 去掉 import 行上的 `# noqa: F401`，这次真用上了。

**验证**：

- `uv run pytest tests/cuda/test_flash_attention.py`：`test_model_steps_match_eager` 已经覆盖 decode 步并和 eager 对比，应仍然通过。
- 跑一次 benchmark：decode 应从约 2494 到约 2700。提升不大是正常的，主要瓶颈还是 CPU 发起 kernel，这一步只是为录 graph 铺路。

**实测**（2026-11-23，5090）：prefill 105869 / 106642，平均 106256 tok/s（不变）；decode 2716 / 2748，平均 2732 tok/s，比基线 2437 提升 1.12×。

## 第 4 步：录制 graph（`ModelRunner.capture_cudagraphs`）

**档位**：`[1, 2, 4, 8, 16, 32, 48, …]`，到 `min(max_num_seqs, 512)` 为止；如果最大值不在列表里（例如 `max_num_seqs=20`），把它补到末尾。

**固定缓冲区**（按最大档位 `max_bs` 分配一次，录制和回放都只用它们的切片 `[:bs]`）：

| 缓冲区 | 形状 | dtype | 初始值 |
| --- | --- | --- | --- |
| `input_ids` | `[max_bs]` | long | 0 |
| `positions` | `[max_bs]` | long | 0 |
| `slot_mapping` | `[max_bs]` | long | **-1** |
| `cache_seqlens` | `[max_bs]` | int32 | 0 |
| `block_table` | `[max_bs, ceil(max_model_len / block_size)]` | int32 | **-1** |
| `outputs` | `[max_bs, hidden_size]` | 模型 dtype | 0 |

`block_table` 的宽度必须固定成 `max_model_len` 对应的块数，而不是"本步最多块数"，否则形状会变。

**录制循环**，每个档位：

1. `set_context(Context(...))`：`is_prefill=False`、`use_flash=True`、`kv_cache`，以及 `slot_mapping / block_table / cache_seqlens` 三个缓冲区的 `[:bs]` 切片。
2. **先正常跑一遍**模型（预热）：Triton 第一次调用会编译，FA2 / cuBLAS 第一次会选 kernel、分配 workspace，这些都不能发生在录制过程中。
3. `with torch.cuda.graph(graph, pool):` 里再跑一遍，把结果写进 `outputs[:bs]`。
4. `finally: reset_context()`。

**要点**：

- **从大到小录**，所有 graph **共用一个显存池**：第一个 graph 录完后取 `graph.pool()`，传给后面所有的 `torch.cuda.graph(..., pool)`。小档位的中间张量能复用大档位的显存。
- graph 只录 `self.model(...)` 到 hidden states；`compute_logits` 和采样留在 graph 外面（只取前 `n` 行算 logits，空位行不白算 lm_head）。
- 加上 `@torch.inference_mode()`。
- 两个前置断言：`self.use_flash`，以及 `self.kv_cache.use_triton`（理由见第 1 步）。

**录制期间不能出现的东西**（出现会报错，或者录出来的 graph 是错的）：

- 任何 CPU 同步：`.item()`、`.tolist()`、`.cpu()`、`print(tensor)`、`torch.cuda.synchronize()`。检查一下 decode 路径里有没有，比如 eager attention 分支里的 `cu_seqlens_q.tolist()`（录制时走 FA2 分支，所以不会碰到）。
- 依赖张量内容的 Python 分支（`if tensor > 0:`）。
- 每步新建张量再传进 kernel：graph 记住的是录制时的地址，回放时新张量不会被看到。所以 Context 里的张量必须是缓冲区切片。

## 第 5 步：回放（`run` 和 `_replay`）

**`run` 的分支**：

- `not is_prefill` 且"能用 graph"→ `_replay`，返回 `[n, hidden]`（已经是每条序列的那一行）。
- 否则走原来的 `set_context → model → reset_context`，再按 `cu_seqlens_q[1:] - 1` 取最后一行。
- 两条路最后都调用 `compute_logits`。

**"能用 graph"的条件**（`_fits_graph`）：已经录过 graph；`n <= 最大档位`；本步 `block_table` 的宽度不超过缓冲区宽度。

**`_replay`**：

1. 选档位：`bs = 第一个 >= n 的档位`。
2. 把本步数据拷进缓冲区的前 `n` 行：`input_ids`、`positions`、`slot_mapping`、`cache_seqlens`、`block_table[:n, :width]`。
3. **把空位重置**：`slot_mapping` 先 `fill_(-1)`，`cache_seqlens` 先 `zero_()`，`block_table` 先 `fill_(-1)`，然后再写前 `n` 行。不重置的话，上一步留下的旧值会让空位行写 KV（覆盖别人的 slot）或读到别的序列的块。
4. `self.graphs[bs].replay()`。
5. 返回 `outputs[:n]`。

**测试**（建议新建 `tests/cuda/test_cuda_graph.py`），沿用 `test_flash_attention.py::test_model_steps_match_eager` 的写法：

- 一个小的随机 Qwen3 + 一个 scheduler 同时驱动两个 runner：一个 FA2 不录 graph，一个 FA2 录 graph，各有自己的 KV cache。
- 每步比较两者的 logits，并用不录 graph 那个 runner 的 argmax 做 teacher forcing。
- 覆盖：条数正好等于档位、需要补齐（如 3 → 4）、条数变化（序列结束 / 新序列加入）、跨块（序列长度越过 256 的边界，`block_table` 多一列）。
- 再加一个"超出最大档位就回退 eager"的用例，例如 `max_num_seqs` 设小，只录到 bs = 2，然后放 3 条。

## 第 6 步：开关与接线

- `config.py`：加 `enforce_eager: bool = False`，含义是 True 时永远不录 graph（和 vLLM / nano-vllm 同名）。
- `engine/llm_engine.py`：在**分配 KV cache 之后**、创建 runner 之后，`if config.use_flash and not config.enforce_eager: runner.capture_cudagraphs(config.max_num_seqs, config.max_model_len)`。放在 KV 之后，graph 的显存池就从 `gpu_memory_utilization` 留下的余量里出（nano-vllm 也是这个顺序）。第 5 步的 activation profiling 不需要改。
- `benchmark.py`：加 `--no-cuda-graph`（对应 `enforce_eager=True`），并且 `--no-triton` 时也不要录 graph。
- README 的 LLM 参数说明里补上 `enforce_eager`。

**验证**：

- `uv run pytest -q` 全部通过（CPU 测试不受影响：CPU 上 `use_flash` 是 False，不会录 graph）。
- 用真实模型跑一次 `LLM.generate`，确认输出文本和 `enforce_eager=True` 时一致（greedy）。

## 第 7 步：测量和写文档

```bash
uv run python -m pico_vllm.benchmark --device cuda:0 --model ~/models/Qwen3-0.6B                  # graph
uv run python -m pico_vllm.benchmark --device cuda:0 --model ~/models/Qwen3-0.6B --no-cuda-graph  # 对照
```

各跑 2 遍取平均。可以顺便记录：

- 录制耗时（engine 启动多花了多久，档位数 × 每档）。
- graph 显存池大小（录制前后 `torch.cuda.memory_allocated` / `memory_reserved` 的差）。
- 只换 `flash_attn_with_kvcache`（第 3 步）和加上 graph（第 5 步）各贡献多少。

**实测**（2026-11-24，5090）：graph decode 7278 / 7293，平均 7286 tok/s；`--no-cuda-graph` 2710 / 2724，平均 2717 tok/s；prefill 约 106k 不变。录制 36 个档位约 0.8 s，显存池 60 MiB。详细结果已写入 [m6-optimizations.md](m6-optimizations.md#6-cuda-graphdecode)。

**和本指南的差异**：实际实现没有把档位上限设为 512，也没有 `_fits_graph` 回退：档位一定包含 `max_num_seqs`，`add_request` 拒绝超过 `max_model_len` 的请求，所以每个 decode 步都装得下，`run` 只看 `self.graphs` 是否为空。

写进 `docs/m6-optimizations.md`：总表第 6 行填数、状态改 ✅；"### 6. CUDA Graph（待做）"一节改写成实际做法和结果；`docs/design-vs-nano-vllm.md` 对应行同步更新。提交信息可以用 `perf: replay decode steps from cuda graphs`。

## 常见问题排查

| 现象 | 可能原因 |
| --- | --- |
| 录制时报 `operation not permitted when stream is capturing` | 录制路径里有 CPU 同步（`.item()` / `.tolist()` / `synchronize`），或者第一次调用 Triton / FA2 发生在录制中（忘了预热） |
| graph 输出和 eager 差很多，但第一步是对的 | Context 里用了新建的张量而不是缓冲区切片；或者回放前没把数据拷进缓冲区 |
| 补齐后其他序列的输出被污染 | 空位没重置：`slot_mapping` 不是 -1（写坏了 KV），或 `cache_seqlens` 不是 0 |
| cache 最后一个 slot 的内容莫名变化 | 用了 eager KV 写入（`--no-triton`），-1 被当成 Python 负索引 |
| 序列变长后结果出错 | `block_table` 缓冲区宽度按本步块数分配，而不是按 `max_model_len` |
| decode 没变快 | `_fits_graph` 一直返回 False（例如档位没录上），可以临时打印一下命中的档位 |
| 启动后可用 KV block 变少 / OOM | graph 录在了 KV 分配之前，或者没有共用显存池 |
