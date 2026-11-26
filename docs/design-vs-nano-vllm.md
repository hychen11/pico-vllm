# M1–M5 设计与 nano-vllm 的对比

对照版本：nano-vllm `bb823b3`（GitHub `GeeeekExplorer/nano-vllm`，2026-11-20 拉取，"chunked-prefill-refactor" 合入之后）；本项目 M5 完成时的提交 `0289b92`。

**一句话总结**：调度器、block manager、prefix cache、chunked prefill 的算法和 nano-vllm 基本一致。主要区别在于本项目把 KV cache 和计算路径做成了**能在 CPU 上运行、可测试**的形式（eager attention、`PagedKVCache` 对象、统一的输入构造），而 nano-vllm 只面向 CUDA，并且直接带有 FlashAttention、CUDA Graph、torch.compile 和张量并行。

## 相同的部分

| 模块 | 共同设计 |
| --- | --- |
| `Sequence` | `num_cached_tokens`（已算完、KV 已缓存）+ `num_scheduled_tokens`（本步要算），用这两个数表示进度 |
| `BlockManager` | xxhash 链式哈希（每块哈希包含前一块的哈希）；只对写满的块做哈希；用 `token_ids` 校验哈希冲突；`ref_count` 共享前缀块；`hash_blocks` 只处理本步新写满的块 |
| `Scheduler` | 一步要么全 prefill、要么全 decode，优先 prefill；只有本步第一条序列可以被切成 chunk，chunk 长度 = 本步剩余 token 预算；decode 时显存不足就抢占最新的序列，放回 waiting 队首 |
| `postprocess` | chunk 未结束时丢弃本步采样结果，不追加 token |
| Context | 用全局 `set_context` / `get_context` 把每步的 attention 元数据传给各层，不改每层的 `forward` 签名 |
| 线性层 | q/k/v 合并成一个 `QKVLinear`，gate/up 合并成一个 `MergedLinear`；加载时用 `packed_modules_mapping` 和 `weight_loader(param, weight, shard_id)` 把 checkpoint 的分片写进合并后的权重 |

## 不同的部分

| 方面 | 本项目（M1–M5） | nano-vllm | 原因 / 影响 |
| --- | --- | --- | --- |
| 运行设备 | CPU 和 CUDA 都能跑；M5 时只在 CPU 上跑，M6 加入 `device` / `dtype` 配置 | 只支持 CUDA | 本项目要求 CI 在 CPU 上通过 |
| KV cache 归属 | `PagedKVCache` 对象，通过 Context 传给每层，用 `layer_idx` 定位，提供 `write` / `read` | 一个大张量 `[2, L, num_blocks, block_size, H, D]`，每个 `Attention` 模块持有自己那一层的 `k_cache` / `v_cache` 视图 | 本项目的 cache 可以单独测试；nano-vllm 少一层间接调用 |
| KV cache 大小 | M5 时固定 `kvcache_capacity_bytes`（2 GiB） | 先跑一次 warmup 测出峰值显存，再用 `gpu_memory_utilization` × 总显存减去已用部分 | M6 第 5 步已补上，公式相同；warmup 的 K/V 写进一个临时 cache，测完释放，不算进 activation 峰值。显式传 `kvcache_capacity_bytes` 或在 CPU 上时仍用固定大小 |
| attention 计算 | eager：对每条序列从 cache 读出完整 K/V，逐条做 causal attention；GQA 用 `repeat_interleave` | FA2：prefill 用 `flash_attn_varlen_func`，decode 用 `flash_attn_with_kvcache` | M6 第 4 步改为 FA2 varlen，第 6 步 decode 改用 `flash_attn_with_kvcache`，现在和 nano-vllm 相同；eager 保留为 CPU 路径和正确性对照 |
| prefill 的 K/V 来源 | 永远先写 cache，再从 cache 读 | batch 内没有任何已缓存 token 时（`cu_seqlens_k[-1] == cu_seqlens_q[-1]`），直接用本步算出的 K/V；否则读 cache | 两种都正确。nano-vllm 的判断是在整个 batch 层面做的，所以 chunk 的后续段会走读 cache 的分支 |
| 输入构造 | 一个 `prepare_inputs` 同时处理 prefill 和 decode（decode 就是 q_len = 1）；`slot_mapping` 逐个 token 计算 | `prepare_prefill` 和 `prepare_decode` 分开写；`slot_mapping` 按块用 `range` 批量生成；decode 只取 `last_token` | nano-vllm 的分法是为 CUDA Graph 准备的（原因见 [concepts/decode-attention-and-cuda-graph.md](concepts/decode-attention-and-cuda-graph.md)）；本项目的统一写法更容易支持 prefill 和 decode 混在同一步 |
| 张量搬运 | `torch.tensor(..., device=...)`，Context 里的 `context_lens` / `block_tables` 是 Python list | pinned memory + `non_blocking=True`，全部转成 int32 GPU 张量，`block_tables` 用 -1 补齐 | M6 第 4 步已改成 int32 GPU 张量、`block_table` 用 -1 补齐；还没用 pinned memory + `non_blocking=True`，可作为后续优化 |
| 结束判断 | `Scheduler.postprocess` 只更新进度；EOS / `max_tokens` 由 `LLMEngine._should_stop` 判断，再调用 `scheduler.finish` | 都在 `Scheduler.postprocess` 里完成 | 本项目把调度和"是否结束"的规则分开，方便单独测试 |
| 采样 | 支持 `temperature = 0`（贪心，`argmax`），同一 batch 可以混合贪心和随机采样；随机采样用 `torch.multinomial`，可传 `generator` 保证测试可复现 | 不允许贪心（`temperature` 必须 > 1e-10）；用 Gumbel-max 技巧（`probs / Exponential(1)` 再 `argmax`），并用 `torch.compile` 编译 | 本项目的测试依赖贪心输出做对比 |
| RMSNorm / RoPE / SiLU | 普通 PyTorch 实现 | 用 `torch.compile` 融合；RMSNorm 把残差相加和归一化融合在一起（`add_rms_forward`） | 可作为 M6 之后的优化项 |
| CUDA Graph | 无 | decode 时按 batch 大小 1、2、4、8、16、32…512 录制 graph；Triton kernel 遇到 `slot == -1`（补齐的空位）直接跳过 | M6 第 6 步已实现，做法相同。区别：档位上限取 `max_num_seqs` 本身（nano-vllm 是 `min(max_num_seqs, 512)`，超过 512 条回退 eager），并且 `add_request` 拒绝超过 `max_model_len` 的请求，所以每个 decode 步都有 graph；`run` 用"是否录过 graph"判断，不在 runner 上另存 `enforce_eager` |
| 张量并行 | 无，单卡 | 支持 TP：`ColumnParallelLinear` / `RowParallelLinear` / `VocabParallelEmbedding`，NCCL all-reduce，多进程之间用共享内存传调用 | M8 已实现，做法相同（层里直接调用 `torch.distributed`，rank 0 通过共享内存 + Event 广播调用）。不同之处：端口不写死；KV block 数在各 rank 间取最小值（两张卡不一样）；process group 有 60 秒超时；TP>1 时不录 CUDA Graph。见 [m8-tensor-parallel.md](m8-tensor-parallel.md) |
| 对外接口 | `LLM.generate`；`add_request` 返回 `Sequence`；`step()` 返回本步结束的序列 | `LLM.generate` 带 tqdm 进度条，实时显示 prefill / decode 吞吐 | — |
| 测试 | 184 个测试（单元、集成、CUDA 对比） | 没有测试，只有 `example.py` 和 `bench.py` | ROADMAP 要求每个里程碑都有测试 |
| 代码量 | `src/` 1763 行，`tests/` 2371 行 | 约 1450 行（含 `example.py` 和 `bench.py`） | 本项目多出的部分主要是文档字符串、CPU 路径和配置 |

## M6 之后和 nano-vllm 趋同的部分

M6 完成后，下面几项已经和 nano-vllm 基本一致：FA2 attention、int32 的 Context 张量、按显存决定 KV block 数、decode 的 CUDA Graph，以及 Triton kernel 支持 `slot == -1`。继续保留的区别：CPU 路径和 eager 实现（作为 CPU 回退和正确性对照）、贪心采样、统一的 `prepare_inputs`，以及调度规则和结束判断分开。

## 后续计划（尚未进行）

1. 在同一台机器、同一负载下，和 nano-vllm、vLLM 对比吞吐（M6 已完成，可以开始）。
2. 根据对比结果，决定下一步优化方向，例如 torch.compile 融合、prefill 和 decode 混在同一步调度。
