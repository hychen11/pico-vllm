# M6 优化记录

M6（CUDA fast path）只做性能优化，不改变 M5 的输出语义。本文件按步骤记录每一项优化：提升了什么、怎么做到的、提升了多少。每完成一步就在表里追加一行。

## 总表

5090、bf16、同一个固定负载（见下方"测量方法"）。"本步提升"是和上一行比较，"累计提升"是和 M5 基线比较。

| # | 优化 | 状态 | prefill（tok/s） | decode（tok/s） | 本步提升 prefill / decode | 累计提升 prefill / decode |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | M5 基线：CPU，fp32，eager attention | 基线 | 437 | 62.2 | — | 1× / 1× |
| 1 | 模型和 KV cache 搬到 GPU（仍是 fp32） | ✅ | 30835 | 280.6 | 70.5× / 4.5× | 70.5× / 4.5× |
| 2 | GPU 上默认使用 bf16 | ✅ | 49021 | 274.3 | 1.59× / 1.0× | 112× / 4.4× |
| 3 | Triton KV 写入 kernel | ✅ | 48861 | 272.0 | 1.0× / 1.0× | 112× / 4.4× |
| 4 | FlashAttention 2（varlen + 分页 KV） | ✅ | 106869 | 2494 | 2.19× / 9.17× | 245× / 40.1× |
| 5 | 按 GPU 空闲显存自动决定 KV block 数 | ✅ | 不影响吞吐：KV cache 73 → 906 个 block（2.00 → 24.77 GiB，可缓存 18688 → 231936 个 token，12.4×） | | | |
| 6 | CUDA Graph（decode） | 待做 | | | | |

**瓶颈变化**：第 1–3 步 decode 一直停在约 275 tok/s，不随 GPU 算力（4060 上也是 274 tok/s）和数据类型变化。原因是 eager attention 对"每层 × 每条序列"做 Python 循环，decode 每步产生上千次 kernel 调用，GPU 大部分时间在等 CPU。第 4 步把 attention 换成每层一次 FA2 调用，decode 提升到约 2500 tok/s。decode 每步仍要由 CPU 逐个发起 28 层的全部 kernel，第 6 步（CUDA Graph）针对的就是这部分开销。

## 测量方法

```bash
uv run python -m pico_vllm.benchmark --device cuda:0 --model ~/models/Qwen3-0.6B [--dtype float32] [--no-triton] [--no-flash]
uv run python -m pico_vllm.benchmark --device cpu --dtype float32      # 第 0 行
```

- 负载：`pico_vllm.benchmark` 默认参数。32 条请求，prompt 长度 128–512 token（共 10705 个，固定随机种子），每条输出 128 个 token（`ignore_eos`）。模型：从第 4 步之后的测量开始用真实的 Qwen3-0.6B 权重（`--model`，不传时取 `$PICO_VLLM_QWEN_MODEL`）。第 0–4 行是用随机初始化、Qwen3-0.6B 形状的模型测的（不传 `--model` 时仍是这个模式）。两者形状完全相同，在第 4 步的代码上对比过：真实权重 prefill 106585 / 106519、decode 2450 / 2461 tok/s，随机权重 prefill 106869 / 106779、decode 2469 / 2440 tok/s，差别在 1% 以内，所以前面几行不用重测。
- 指标：prefill tok/s = prompt token 数 / 只做 prefill 的耗时；decode tok/s = 32 × 127 / （全程耗时 − prefill 耗时）。
- GPU 行每个配置跑 2 遍，表中是平均值；CPU 基线耗时约 2 分钟，只跑了 1 遍。
- 第 0 行用的是当前代码的 `--device cpu` 路径。CPU 上不启用 Triton，attention 也是 eager，计算路径和 M5 一致。
- 测量前确认 5090 显存空闲：`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`。
- 硬件和软件版本见 [Triton 实验报告的"实验环境"](experiments/m6-triton-kv-store.md#实验环境)。

## 各步骤说明

### 1. 模型和 KV cache 搬到 GPU

- **提升了什么**：prefill 70.5×，decode 4.5×。
- **怎么做的**：`Config` 新增 `device`（`auto` 时有 CUDA 就用 CUDA）。`utils/loader.py` 新增 `init_on`，让模型参数直接在目标设备上按目标 dtype 创建，不用先在 CPU 上建一份 fp32 再搬过去；`PagedKVCache` 也分配在同一设备上。
- **为什么 decode 只快了 4.5 倍**：decode 每步计算量很小，GPU 算得再快，也要等 CPU 一个个发起 kernel。

原始数据：prefill 30944 / 30760 / 30910，decode 274.6 / 280.1 / 281.0（3 次运行，表中取后 2 次的平均）。

### 2. GPU 上默认使用 bf16

- **提升了什么**：prefill 1.59×，decode 无变化。
- **怎么做的**：`Config.dtype` 设为 `auto` 时，CUDA 上取 checkpoint 的 dtype（没有就用 bf16），CPU 上保持 fp32。`sampler.py` 先把 logits 转成 fp32，再做 softmax 和采样，避免 bf16 精度问题。RoPE 的 `cos_sin_cache` 显式保持 fp32。
- **为什么 decode 没变**：bf16 让矩阵乘更快、显存读写减半，但 decode 卡在 CPU 发起开销上，GPU 计算变快体现不出来。

原始数据：prefill 49048 / 48995，decode 271.9 / 276.7。

### 3. Triton KV 写入 kernel

- **提升了什么**：端到端无变化。单独测 KV 写入，GPU 耗时降到原来的 1/4 左右（decode 规模：3.7 μs → 0.8 μs），kernel 数从 2 个减为 1 个。
- **怎么做的**：`kernels/kv_store.py`，每个 Triton program 拷贝一个 token 的 K 行和 V 行。cache 在 CUDA 上且能导入 triton 时自动启用。
- **为什么端到端没变**：KV 写入只占 decode 每步耗时的不到 0.5%，而且每次写入的实际耗时由约 13 μs 的 CPU 发起开销决定。等第 6 步（CUDA Graph）消除 CPU 开销后，它的收益才会显现。
- 详细实验见 [experiments/m6-triton-kv-store.md](experiments/m6-triton-kv-store.md)。

原始数据：prefill 49012 / 48710，decode 275.0 / 269.0。

**已修复的问题**：数据不在默认 GPU 上时（如 `--device cuda:1`），Triton kernel 曾因按默认 GPU 编译而报错 `no kernel image is available`。现在启动 kernel 前会切换到 cache 所在的 GPU，并有对应测试。

### 4. FlashAttention 2

- **提升了什么**：prefill 2.19×，decode 9.17×。
- **怎么做的**：每层用一次 `flash_attn_varlen_func`（带 `block_table`，直接读分页 KV cache）处理整个 batch，取代 eager 的逐条循环。prefill、前缀缓存命中、chunked prefill、decode 都走这一次调用。
  - `Config.use_flash` / `resolve_use_flash`：CUDA、fp16/bf16、能导入 `flash_attn`、`block_size % 256 == 0` 时自动启用，否则退回 eager。benchmark 用 `--no-flash` 做对比。
  - `prepare_inputs` 额外生成 `cu_seqlens_k`、`max_seqlen_q/k` 和用 -1 补齐的 int32 `block_table`，`cu_seqlens_q` 改为 int32。字段含义见 [concepts/attention-context.md](concepts/attention-context.md)。
  - FA2 2.8.4 是针对 sm_120 自己编译的，wheel 备份在 `~/wheels/`。本项目默认运行在 5090 上，自动启用时不检查 GPU 架构；在其他 GPU（如 sm_89 的 4060）上要传 `use_flash=False`，否则第一步会报 `no kernel image is available`。
- **为什么 decode 提升大得多**：eager 在 decode 时每层每条序列都要发起好几个很小的 kernel，32 条序列 × 28 层，CPU 发起开销占了绝大部分时间。FA2 每层只发起 1 个 kernel，这部分开销基本消失。prefill 本来就以矩阵乘为主，attention 的占比小，所以只快了约 2 倍。
- **正确性**：`tests/cuda/test_flash_attention.py`。单层测试用一个混合 batch（全新 prompt、前缀命中、decode）比较 FA2 和 eager 的输出；模型测试让一个调度器同时驱动 eager 和 FA2 两个 runner，覆盖 chunked prefill、前缀命中和 decode，每步用 eager 的 token 继续生成（teacher forcing），逐步比较 logits。bf16 下生成的 token 不能直接比：真实 Qwen3-0.6B 上，两条路径在 top-2 logits 几乎相等（差 0 或 0.0625）的位置会选出不同的 token，之后的序列自然不同。

原始数据：prefill 106773 / 106867 / 106871，decode 2508.0 / 2506.2 / 2481.7（3 次运行，表中取后 2 次的平均）。同一时间 `--no-flash` 为 prefill 48890、decode 272.4。

### 5. 按 GPU 空闲显存自动决定 KV block 数

这一步不提升吞吐，而是让 KV cache 用满可用显存，提高同时容纳的序列数。计算逻辑、warmup 的每一步和启动前清理的原因，详见 [concepts/kv-cache-sizing.md](concepts/kv-cache-sizing.md)。

- **怎么做的**：新增 `Config.gpu_memory_utilization`（默认 0.9），`kvcache_capacity_bytes` 默认改为 `None`。CUDA 上不传 `kvcache_capacity_bytes` 时，`worker/memory.py` 按和 nano-vllm 相同的公式计算：

  ```text
  KV 可用字节 = 总显存 × gpu_memory_utilization − 已用显存 − activation 峰值
  block 数   = KV 可用字节 // 每个 block 的字节数（Qwen3-0.6B、bf16、block_size 256 时为 28 MiB）
  ```

  - 已用显存：warmup 之后的 `torch.cuda.mem_get_info`，包括本进程（CUDA context、权重）和 GPU 上其他进程占用的显存；
  - activation 峰值：`profile_activation_bytes` 用 runner 真实地跑调度器可能产生的两种最大的步，记录这期间比起点多分配的显存峰值（`max_memory_allocated() - base`）：
    1. 最大的 prefill 步：把一步的 token 上限塞满，即 `max_num_batched_tokens // max_model_len` 条 `max_model_len` 长的 prompt（默认 4 条 × 4096）；
    2. 最大的 decode 步：`max_num_seqs` 条序列（默认 512）各算 1 个 token，每条的上下文都是 `max_model_len` 长。它们不另外分配 block，而是轮流共用第 1 步那几条 prompt 已经写好的 KV（decode 序列 i 用 prefill 序列 i % 4 的 block_table）。这几条 prompt 内容完全相同，真实运行时前缀缓存也会这样共享；每条重算自己的最后一个 token 并写回原位置，数值无所谓，只测显存；
  - warmup 的 K/V 写进一个临时 cache。这个 cache 在重置峰值计数之前分配、测完就释放，所以不会被重复计入；
  - 显式传了 `kvcache_capacity_bytes` 时仍按固定大小分配；CPU 上默认 2 GiB（`DEFAULT_CPU_KVCACHE_BYTES`）；
  - 一个 block 都放不下时直接报错，报错信息里写明已用显存和 activation 预留量；
  - CUDA 上加载权重之前先 `gc.collect()` 和 `torch.cuda.empty_cache()`。原因：同一进程里先删掉一个引擎再新建时，旧引擎的 KV cache 立即释放，但模型因为循环引用要等 gc；释放出的 KV 显存留在 PyTorch 的缓存里，新权重会被分配进去，这段显存之后无法归还，被 `mem_get_info` 算成"已用"。在 5090 上用默认参数连续创建 3 个引擎：不加这一步时 block 数依次为 906、884、884；加上之后每次都是 906。
- **测试**：`tests/unit/test_kv_sizing.py`（公式、warmup 形状、三种来源的选择、配置校验）；`tests/cuda/test_kv_sizing_cuda.py`（临时 cache 被释放、自动分配后不超过上限、调低 utilization 得到更少的 block、真实模型能生成）。GPU 已被其他进程占用 40% 以上时，后几个测试会跳过。
- **结果**：RTX 5090，真实 Qwen3-0.6B，默认参数（`max_num_batched_tokens=16384`、`max_num_seqs=512`、`max_model_len=4096`、block_size 256、bf16、FA2）：

  | 配置 | block 数 | 可缓存 token 数 | KV cache 大小 | 建好后已用显存 |
  | --- | ---: | ---: | ---: | ---: |
  | 原来固定 2 GiB | 73 | 18688 | 2.00 GiB | 4.66 GiB |
  | `gpu_memory_utilization=0.9`（默认） | 906 | 231936 | 24.77 GiB | 27.19 GiB |
  | `gpu_memory_utilization=0.5` | 447 | 114432 | 12.22 GiB | 14.64 GiB |

  - 可缓存的 token 数是原来的 12.4 倍。已用显存 27.19 GiB 加上 activation 峰值 1.02 GiB 为 28.21 GiB，正好等于 0.9 × 31.36 GiB；
  - activation 峰值 1039 MiB，主要来自 decode 步 512 行 logits 的采样：词表 151936，logits 先是 bf16，再转成 fp32 并做 softmax，每份都有几百 MB。prefill 步只有 4 行 logits，峰值约 744 MiB；
  - 吞吐不变，benchmark 仍用固定的 `--kv-cache-gib 4`，以便和前面几行对比。

  同一张 5090 上 FA2 和 eager（`use_flash=False`）的 warmup 对比，util=0.9。耗时取进程内第二次运行，第一次含 kernel 编译，FA2 约 1.3 秒：

  | attention | activation 峰值 | warmup：prefill 段 | warmup：decode 段 | block 数 |
  | --- | ---: | ---: | ---: | ---: |
  | FA2 | 1039 MiB | 0.21 s | 0.08 s | 906 |
  | eager | 1472 MiB | 0.65 s | 1.59 s | 891 |

  eager 的峰值更高，因为它为每条 prompt 生成完整的 4096 × 4096 打分矩阵。eager 的 decode 段慢 20 倍，原因和第 1–3 步 decode 吞吐只有约 275 tok/s 相同：每层 × 每条序列都要做一次 Python 循环，512 条 × 28 层就是 14336 次，每次读 4096 长的上下文并发起好几个小 kernel，时间主要花在 CPU 发起 kernel 上。

### 6. CUDA Graph（待做）

计划：对纯 decode 步骤，按几种固定 batch 大小录制 CUDA Graph。decode 分支改用 `flash_attn_with_kvcache`，因为它只依赖 GPU 张量 `cache_seqlens`，不依赖每步都会变的 Python 整数 `max_seqlen_k`。Triton kernel 需要支持 `slot == -1` 时跳过，用于 batch 补齐的空位。原理和例子见 [concepts/decode-attention-and-cuda-graph.md](concepts/decode-attention-and-cuda-graph.md)。
