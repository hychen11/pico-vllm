# M8 张量并行：结果记录

M8 把 Qwen3 的每一层切到多张卡上：attention 按 head 切，MLP 按中间维切，词表按行切，每层用 all_reduce 合并。实现步骤和设计取舍见 [m8-tp-guide.md](m8-tp-guide.md)，并行层的原理见 [concepts/tp-linear-layers.md](concepts/tp-linear-layers.md)。

**一句话结论**：TP=2 的输出和单卡一致（float32 下 logits 最大差 3.1e-5），但在这台机器上**比 5090 单卡慢**：prefill 只有单卡的 1/4，decode 差不多。原因是两张卡算力差距大，中间也没有直连。M8 的目标是验证正确性，这个速度结果符合预期。

## 环境

| | rank 0 | rank 1 |
| --- | --- | --- |
| 卡 | RTX 5090（32 GB，`cuda:0`） | RTX 4060 Laptop（8 GB，`cuda:1`） |
| PCIe 链路（`nvidia-smi`） | x4（最大 x16） | Gen4 x8 |
| attention | eager（`--no-flash`，4060 上没有 FA2 的 kernel） | eager |
| CUDA Graph | 无（TP>1 强制 `enforce_eager`） | 无 |

两卡拓扑是 **PHB**：不支持 P2P，NCCL 走 `SHM/direct`，也就是数据要经过主机内存中转。torch 2.15.0+cu132，NCCL 2.30.7，Qwen3-0.6B，bf16。

## 正确性（`tests/cuda/test_tensor_parallel.py`）

| 测试 | 内容 | 结果 |
| --- | --- | --- |
| `test_tp2_steps_match_tp1` | 子进程里跑 TP=2 engine，逐步记录 batch、logits 和 greedy token，每步之后检查 M7 的调度器不变量。主进程用 TP=1 模型重放同样的 51 步，用子进程的 token 做 teacher forcing，比较 logits。float32 | 覆盖 chunked prefill 3 次、前缀命中 3 次、抢占 2 次。logits 最大差 **3.1e-5**（容差 1e-3） |
| `test_tp2_llm_generates_and_exits_cleanly` | 同一个子进程里先后建两个 `LLM(..., tensor_parallel_size=2)`，bf16 下生成 "The capital of France is" | 两次都生成 "Paris"。`exit()` 后 worker 的退出码是 0，`/dev/shm` 里没有残留，process group 已销毁 |

**测试确实能抓到 bug**：去掉 `RowParallelLinear` 里的 all_reduce，两个测试都会失败，logits 差到 23。

**为什么用 float32 比较**：bf16 下 TP=2 和 TP=1 的 greedy 输出会在几个 token 之后分叉。这不是 bug，原因是 all_reduce 改变了加法顺序，bf16 只有 8 位尾数，误差在 28 层里累积，最后会让分数接近的两个候选 token 换位。详见 [m8-tp-guide.md 第 5 步的实测](m8-tp-guide.md#第-5-步多进程)。

## 吞吐

```bash
uv run python -m pico_vllm.benchmark --model ~/models/Qwen3-0.6B --no-flash --device cuda:0           # 5090
uv run python -m pico_vllm.benchmark --model ~/models/Qwen3-0.6B --no-flash --device cuda:1 --kv-cache-gib 2   # 4060
uv run python -m pico_vllm.benchmark --model ~/models/Qwen3-0.6B --no-flash --tp 2                    # TP=2
```

负载和 M6 相同：32 条请求，prompt 128–512 token（共 10705 个），每条输出 128 个 token。每种配置跑 2 遍，表里是平均值。三种配置都是 eager attention、不开 CUDA Graph。4060 的 KV 用 2 GiB：默认的 4 GiB 放不下，会 OOM；1 GiB 又只放得下约 9.3k 个 token，跑的过程中会触发抢占，结果不公平（实测 1 GiB 时 decode 是 224 tok/s）。

| 配置 | prefill（tok/s） | decode（tok/s） | 相对 5090 单卡 prefill / decode |
| --- | --- | --- | --- |
| 5090 单卡 | 47,942 | 262.5 | 1× / 1× |
| 4060 单卡 | 10,555 | 252.4 | 0.22× / 0.96× |
| **TP=2（5090 + 4060）** | **11,933** | **244.3** | **0.25× / 0.93×** |

## 时间花在哪里

rank 0 上用 `torch.profiler` 测了一步（32 条 prompt 共 10705 个 token，一步 prefill 完），以及 32 条序列的 decode 步。GPU 时间只统计 kernel。注意 `key_averages()` 里名为 `all_reduce` 的算子会把 NCCL kernel 的时间再算一遍，统计时要去掉。

| 配置 | prefill 步：总耗时 | GPU 计算 | NCCL | decode 步：总耗时 | GPU 计算 | NCCL |
| --- | --- | --- | --- | --- | --- | --- |
| 5090 单卡 | 333 ms | 138 ms | — | 184 ms | 22 ms | — |
| 4060 单卡 | 1137 ms | 1019 ms | — | 181 ms | 52 ms | — |
| TP=2（rank 0） | 903 ms | ≈ 79 ms | 622 ms | 186 ms | ≈ 20 ms | 3.3 ms |

每步有 57 次 all_reduce：28 层 × 2（`o_proj`、`down_proj`），再加 embedding 的 1 次。

- **prefill 慢在通信和等 4060**。每次 all_reduce 传 10705 × 1024 × 2 B ≈ 21 MiB，57 次共约 1.2 GB。按 3.9 GB/s 算，光传输就要约 300 ms。rank 0 上 NCCL kernel 的 622 ms 里，还包括等 4060 算完自己那一半的时间：4060 算完整模型要 1019 ms，算一半大约 500 ms。5090 算自己那一半只要约 79 ms，其余时间都在等。
- **decode 和卡没关系**。三种配置每步都是约 185 ms，而 GPU 计算只有 20–50 ms。瓶颈在 eager attention 对"每层 × 每条序列"做的 Python 循环，CPU 发起 kernel 太慢（M6 已经分析过）。TP 多出来的 3.3 ms 通信在这里几乎看不出来。

## 两卡之间的带宽

NCCL 测试，2 个进程，bf16：

| 消息大小 | all_reduce 耗时 | all_reduce 带宽 | send 0→1 单向带宽 |
| --- | --- | --- | --- |
| 64 KiB（decode 32 条序列一层的量） | 35 µs | 1.9 GB/s | （受延迟限制） |
| 1 MiB | 329 µs | 3.2 GB/s | 6.3 GB/s |
| 8 MiB | 2.2 ms | 3.8 GB/s | 6.7 GB/s |
| 256 MiB | 68 ms | 4.0 GB/s | 6.75 GB/s |

- 单向带宽 6.75 GB/s，接近 PCIe 4.0 x4 的理论上限（约 7.9 GB/s）。5090 那一侧的链路只有 x4，它就是瓶颈。
- all_reduce 的带宽大约是单向的一半，因为数据要经过主机内存中转，每个方向都要过两次 PCIe。
- 小消息受延迟限制：64 KiB 也要 35 µs，所以 decode 每步的 57 次 all_reduce 至少需要约 2 ms。
- 作为对比，NVLink 的带宽在几十到几百 GB/s 量级，所以实际中 TP 基本只在有 NVLink 的卡之间使用。

## KV cache 容量

默认 `gpu_memory_utilization=0.9`，block 大小 256：

| 配置 | 每张卡算出的 block 数 | 实际使用 | 可缓存 token 数 |
| --- | --- | --- | --- |
| 5090 单卡 | 871 | 871 | 222,976 |
| 4060 单卡 | 131 | 131 | 33,536 |
| TP=2 | rank 0：1812，rank 1：347 | **347**（取最小值） | 88,832 |

TP=2 时每张卡只存一半的 KV head，每个 block 占用的显存减半，所以 5090 能放的 block 数从 871 涨到 1812。但所有 rank 共用一张 block table，只能取最小值，结果由 4060 决定：容量是 4060 单卡的 2.6 倍，但只有 5090 单卡的 40%。

## 已知限制

- **worker 在等待命令时被 SIGKILL（比如 OOM killer），rank 0 会永远卡住**。`mp.Event.set()` 会等每个正在 wait 的进程确认，死掉的进程永远不会确认，NCCL 的超时也管不到。worker 抛异常则没有问题：rank 0 会在 60 秒后被 NCCL watchdog 中止。见 [m8-tp-guide.md 第 5 步的坑](m8-tp-guide.md#第-5-步多进程)。
- TP>1 时不支持 CUDA Graph，4060 上也不支持 FA2，这是方案 A 的取舍，见指南第 0 步和第 6 步。
- 写这份文档时，一个临时的 profiling 脚本有两次卡在 rank 0 的 decode 循环里：都发生在 rank 0 上用过 `torch.profiler` 之后，原因没有查。benchmark 和双卡测试从来没有出现过这种情况。

## 什么时候 TP 才会有收益

在这台机器上不会。要有收益，需要：

- **几张卡一样快**：每一步都要等最慢的那张卡；
- **卡之间带宽高**（NVLink），否则 prefill 的通信时间会超过省下来的计算时间；
- **decode 不受 CPU 限制**：要用 FA2 + CUDA Graph，TP 下还要各 rank 同步录制 graph（见指南第 6 步的可选部分）；
- 或者**模型大到一张卡放不下**：这才是 TP 最常见的用途，而 Qwen3-0.6B 用不到。
