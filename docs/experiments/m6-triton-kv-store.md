# Triton KV 写入优化实验报告

2026-11-20 · M6 第 1 步（Triton KV-store kernel）

**结论**：Triton kernel 把 KV 写入的 GPU 时间降低了 3–4 倍，但端到端吞吐没有变化。原因是 decode 的瓶颈在 CPU 发起 kernel，而不在 GPU 计算。

## 实验环境

所有数据都在同一台机器上测得。代码基于提交 `0289b92`，加上尚未提交的 M6 改动（Triton kernel、device/dtype 配置）。

| 项目 | 值 |
| --- | --- |
| GPU 0（PyTorch `cuda:0`） | RTX 5090，sm_120，31.4 GiB，经 OCuLink/M.2 接入，PCIe 4.0 x4 |
| GPU 1（PyTorch `cuda:1`） | RTX 4060 Laptop，sm_89，7.6 GiB |
| CPU / 内存 | Intel i7-13650HX / 15 GiB |
| 系统 / 内核 | Ubuntu 24.04.4 LTS / 7.0.0-43-generic |
| NVIDIA 驱动 | 595.99.02（open kernel module） |
| Python / PyTorch / CUDA | 3.14.6 / 2.15.0+cu132 / 13.2 |
| Triton | 3.9.0 |
| 数据类型 | bfloat16 |
| KV 形状 | Qwen3-0.6B：28 层，8 个 KV head，head_dim 128，block_size 256 |

机器上的预先设置：

- **5090 禁止 runtime 休眠**：`/etc/udev/rules.d/80-nvidia-5090-nopm.rules` 把 `0000:6c:00.0/.1` 的 `power/control` 设为 `on`。此前 5090 从 D3hot 唤醒失败，出现过 Xid 79（掉卡）。
- **GPU 编号**：nvidia-smi 里 5090 是 1 号，PyTorch 里 5090 是 `cuda:0`，因为 CUDA 默认把最快的卡排在第一。
- **显存**：5090 上的实验在 llama-server（占用 28 GB）关闭后进行。它运行期间，端到端实验改在 4060 上做。
- **模型**：端到端 benchmark 使用随机初始化、Qwen3-0.6B 形状的模型，因为吞吐只取决于形状。真实权重在 `~/models/Qwen3-0.6B`，用于正确性测试。

## 对比对象

两种实现做同一件事：把本步每个 token 的 K、V 行写到 `slot_mapping` 指定的缓存槽位。区别在于启动几个 GPU kernel。

| | eager（`use_triton=False`） | Triton（`kernels/kv_store.py`） |
| --- | --- | --- |
| 写法 | `k_cache[layer].view(-1, H, D)[slot_mapping] = k`，V 再写一次 | `_store_kvcache_kernel`，每个 program 拷贝一个 token 的 K 行和 V 行 |
| 每次写入的 GPU kernel 数 | 2（`index_elementwise_kernel` × 2） | 1 |
| 对 strided 输入 | 由 PyTorch 内部处理 | 按行 stride 直接读，行内不连续时才 `contiguous()` |
| 启用条件 | 始终可用，CPU 上也能跑 | cache 在 CUDA 上且能导入 triton 时自动启用 |

微基准里的 K、V 按模型的实际情况构造：它们是融合 qkv 输出的切片，行与行之间不连续。

## 实验 1：单独测 KV 写入（5090）

Triton 的 GPU 耗时只有 eager 的约 1/4。但 batch 较小时，两边 CPU 发起调用都要 13–15 μs，比 GPU 计算本身还长。

设置：单层 cache，80 个 block，slot 随机打散。测量前先用矩阵乘把 GPU 频率拉起来。CPU 耗时是连续发起 1000 次、中间不同步的平均值；GPU 耗时是 torch profiler 统计的 kernel 时长之和，取 100 次平均。

| 本步 token 数 | CPU eager（μs） | CPU Triton（μs） | GPU eager（μs） | GPU Triton（μs） | GPU 加速 |
| --- | --- | --- | --- | --- | --- |
| 1（decode 单条） | 15.1 | 13.4 | 3.7 | 0.8 | 4.6× |
| 32 | 14.6 | 13.6 | 4.1 | 0.9 | 4.6× |
| 256 | 14.8 | 13.5 | 4.3 | 1.0 | 4.3× |
| 2048 | 14.4 | 13.7 | 11.8 | 3.9 | 3.0× |
| 16384（大 prefill） | 55.1 | 12.5 | 115.1 | 79.2 | 1.5× |

在 2048 个 token 以下，一次写入的实际耗时由 CPU 发起开销决定，所以两种实现都在 12 μs 左右。第一次测量时，1 token 的 Triton 测出 35.6 μs，原因是 GPU 还没升频；预热后重测，得到上表的数值。

脚本（在项目根目录用 `uv run python <文件>` 运行）：

```python
import time
import torch
from torch.profiler import ProfilerActivity, profile
from pico_vllm.core.paged_kv_cache import PagedKVCache

dev, dt = "cuda:0", torch.bfloat16
NB, B, KH, D = 80, 256, 8, 128
fast = PagedKVCache(1, NB, B, KH, D, dtype=dt, device=dev, use_triton=True)
eager = PagedKVCache(1, NB, B, KH, D, dtype=dt, device=dev, use_triton=False)

def kv(T):  # 与模型一致：k、v 是融合 qkv 输出的切片，行间 strided
    qkv = torch.randn(T, (16 + 2 * KH) * D, device=dev, dtype=dt)
    return qkv[:, 16 * D:(16 + KH) * D].view(T, KH, D), qkv[:, (16 + KH) * D:].view(T, KH, D)

x = torch.randn(4096, 4096, device=dev)
for _ in range(50):
    x @ x  # 预热 GPU 频率
torch.cuda.synchronize()
for T in (16384, 2048, 256, 32, 1):
    k, v = kv(T)
    slots = torch.randperm(NB * B, device=dev)[:T]
    row = []
    for c in (eager, fast):  # CPU：发起调用的耗时，不同步
        for _ in range(50):
            c.write(0, k, v, slots)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(1000):
            c.write(0, k, v, slots)
        row.append((time.perf_counter() - t) * 1e3)
        torch.cuda.synchronize()
    for c in (eager, fast):  # GPU：kernel 时长之和
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(100):
                c.write(0, k, v, slots)
            torch.cuda.synchronize()
        row.append(sum(e.device_time for e in p.events() if e.device_type.name == "CUDA") / 100)
    print(T, " ".join(f"{r:.1f}" for r in row))
```

## 实验 2：端到端 benchmark

开关 Triton 对吞吐没有可测量的影响。而且 5090 的 decode 和 4060 一样快，说明 decode 的瓶颈在 CPU。

负载：`pico_vllm.benchmark` 默认参数，32 条请求，prompt 长度 128–512 token（共 10705 个），每条输出 128 个 token（`ignore_eos`），bf16。每个配置跑 2 遍，表中两遍结果用 / 分隔。

| GPU | Triton | prefill（tok/s） | decode（tok/s） |
| --- | --- | --- | --- |
| RTX 5090 | 开 | 48590 / 48784 | 270.2 / 273.5 |
| RTX 5090 | 关 | 49313 / 48888 | 276.8 / 274.8 |
| RTX 4060 | 开 | 10573 / 10465 | 274.4 / 274.4 |
| RTX 4060 | 关 | 10272 / 10579 | 277.9 / 276.6 |

1. **开关 Triton 的差异在噪声范围内**（±1–3%）。decode 每步约 117 ms，KV 写入只占 28 层 × 约 15 μs ≈ 0.4 ms，不到 0.5%。
2. **5090 的 prefill 是 4060 的 4.6 倍，decode 却一样**，都是约 274 tok/s。decode 时每条序列每步只算 1 个 token，但 eager attention 对 32 条序列 × 28 层逐条用 Python 循环处理，每步产生上千次 kernel 调用，GPU 大部分时间都在等 CPU。

## 结论与下一步

Triton kernel 本身更快，kernel 数量也更少，应当保留。但当前吞吐由 eager attention 的 CPU 开销决定，它的收益要等下面两步完成后才会显现。

1. **接入 FlashAttention 2**：用一次 `flash_attn_varlen_func`（带 `block_table`）处理整个 batch，取代"每层 × 每条序列"的 Python 循环。预计 decode 会明显提速。FA2 2.8.4 已针对 sm_120 编译，并在 5090 上验证通过。
2. **CUDA Graph**：消除 decode 每步剩余的 CPU 发起开销。到这一步，Triton 的 0.8 μs 对比 eager 的 3.7 μs 才会体现在吞吐上。

**实验中发现的 bug**：用 `--device cuda:1` 跑 Triton 路径会报错：

```
Triton Error [CUDA]: no kernel image is available for execution on the device
```

原因是 Triton 按当前默认 GPU（`cuda:0`，即 5090，sm_120）编译和启动 kernel，而数据在 `cuda:1`（4060，sm_89）上。本报告中 4060 的数据是用 `CUDA_VISIBLE_DEVICES=1` 绕过这个问题得到的。

**已修复（2026-11-21）**：`kernels/kv_store.py` 启动 kernel 时包一层 `with torch.cuda.device(k_cache.device):`，并新增测试 `tests/cuda/test_triton_store.py::test_cache_on_non_current_device`（少于 2 张 GPU 时跳过）。修复后 `--device cuda:1` 可以直接运行，不再需要 `CUDA_VISIBLE_DEVICES`。

**正确性**：设置 `PICO_VLLM_QWEN_MODEL` 后，完整测试 184 个全部通过，包括真实权重测试和 Triton 与 eager 的对比测试。

## 复现步骤

在项目根目录运行。先用 `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` 确认 5090 的显存是空闲的。

```bash
# 端到端：5090，Triton 开 / 关
uv run python -m pico_vllm.benchmark --device cuda:0
uv run python -m pico_vllm.benchmark --device cuda:0 --no-triton

# 端到端：4060（多卡 bug 已修复，可直接指定 cuda:1）
uv run python -m pico_vllm.benchmark --device cuda:1 --kv-cache-gib 2
uv run python -m pico_vllm.benchmark --device cuda:1 --kv-cache-gib 2 --no-triton

# 正确性：全部测试，包括真实权重
PICO_VLLM_QWEN_MODEL=$HOME/models/Qwen3-0.6B uv run pytest -q
```

注意：flash-attn 还没有写进 `pyproject.toml`，运行 `uv sync` 会把它卸载。卸载后可以用 `uv pip install --no-deps ~/wheels/flash_attn-2.8.4-cp314-cp314-linux_x86_64.whl` 重新安装。
