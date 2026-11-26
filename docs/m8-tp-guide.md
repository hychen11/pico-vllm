# M8 实现指南：张量并行（Tensor Parallelism）

这是一份动手清单：每一步改哪里、为什么改、怎么验证。难点给出关键代码片段，其余不给完整实现。ROADMAP 里的定义见 [ROADMAP.md](../ROADMAP.md#m8--tensor-parallelism)，nano-vllm 的对应实现在 `nanovllm/layers/linear.py`、`layers/embed_head.py`、`engine/model_runner.py`、`engine/llm_engine.py`。

**目标**：`LLM(model, tensor_parallel_size=2)` 把每层的 attention head 和 MLP 中间维切成两份，分别放在两张卡上算，每层用 all-reduce 合并结果；输出和 TP=1 一致（数值误差以内）。`tensor_parallel_size=1`（默认）的计算和现在完全相同；和 nano-vllm 一样，单卡也会建一个 world_size=1 的 process group，层里直接用 `torch.distributed`。

**这台机器上 M8 的意义是正确性，不是提速**：两张卡差别很大，中间也没有直连（见第 0 步），TP=2 会比 TP=1 慢。

**只支持单机多卡**（和 nano-vllm 一样）：用 spawn 起 worker，`init_method` 连 `127.0.0.1`，用共享内存 + `Event` 广播调用，rank i 用 `cuda:i`，这些都默认所有 rank 在同一台机器上。实际中 TP 一般也只在单机内用（每层 2 次 all_reduce，需要 NVLink 级别的带宽），跨机器用 PP 或 DP。以后要跨机器，主要改两处：
- 起进程：spawn 改成 `torchrun` / Slurm，`init_method="env://"`，`set_device(LOCAL_RANK)`；
- 广播调用：共享内存改成 `dist.broadcast_object_list`（用单独的 gloo group）或 ZMQ。

层的代码不用改，因为它只读 `dist.get_rank()` / `dist.get_world_size()`。

## 总览

| 步骤 | 改动 | 文件 |
| --- | --- | --- |
| 0 | 环境验证，做出 attention 方案决定 | `scripts/check_tp_env.py` |
| 1 | 单卡也初始化 process group（同 nano-vllm），测试里一个 fixture 兜底 | `engine/llm_engine.py`、`benchmark.py`、`tests/conftest.py` |
| 2 | 并行线性层和切分加载 | `layers/linear.py` |
| 3 | 词表并行的 embedding 和 LM head | 新建 `layers/embed_head.py` |
| 4 | 模型改用本地 head 数，KV cache 只存本 rank 的 KV head | `models/qwen3.py`、`engine/llm_engine.py` |
| 5 | 多进程：启动 worker、rank 0 广播调用、退出清理 | `worker/model_runner.py`、`engine/llm_engine.py`、`config.py` |
| 6 | 接线：配置校验、attention / graph 开关、KV block 数取各 rank 最小值、benchmark | `config.py`、`engine/llm_engine.py`、`benchmark.py` |
| 7 | 双卡测试 | `tests/cuda/test_tensor_parallel.py` |
| 8 | 测量并写文档 | `docs/m8-tensor-parallel.md` |

第 1–4 步都在单进程、TP=1 下成立，M7 的 318 个测试必须一直通过；第 2–3 步另有 CPU 上 2 进程（gloo）的切分测试。第 5 步开始才真正起多个进程。

## 架构

和 nano-vllm 相同：

```text
主进程 = rank 0（cuda:0，5090）              子进程 = rank 1（cuda:1，4060），spawn 启动
├─ Scheduler / BlockManager（只在这里）
├─ ModelRunner(rank 0)                      └─ ModelRunner(rank 1)
│    call("run", seqs, is_prefill)               loop(): 等 Event → 从共享内存读 (方法名, 参数)
│      ├─ 写共享内存，set Event  ───────────────►     → 调用同名方法
│      └─ 自己也调用 run                           run: prepare_inputs → model（各算一半 head）
│           每层 2 次 all_reduce  ◄──── NCCL ────►       → 不采样，返回 None
│           lm_head: logits gather 到 rank 0
│      └─ 只有 rank 0 采样
└─ exit(): 广播 "exit" → 关共享内存 → destroy_process_group → join 子进程（atexit 兜底）
```

- 每个 rank 都用 rank 0 传来的 `seqs` **自己运行** `prepare_inputs`。跨进程传的是 Python 对象，不是张量。
- 调度、block 分配、采样、结束判断只在 rank 0。各 rank 的 KV cache 用同一套 block_table，只是每个 rank 存自己那几个 KV head。

每层的切法（Megatron 式，每个 decoder layer 恰好 2 次 all-reduce）：

| 模块 | 切法 | 本 rank 的形状（Qwen3-0.6B，TP=2） | 之后的通信 |
| --- | --- | --- | --- |
| `embed_tokens` | 按词表切（行） | `[151936/2, 1024]` | all_reduce |
| `qkv_proj` | 按输出切（列），q/k/v 各自切 | q 8 个 head、k/v 各 4 个 head | 无 |
| attention | 每个 rank 算自己的 head | — | 无 |
| `o_proj` | 按输入切（行） | `[1024, 1024]` | **all_reduce** |
| `gate_up_proj` | 按输出切，gate / up 各自切 | `[2 × 1536, 1024]` | 无 |
| `down_proj` | 按输入切 | `[1024, 1536]` | **all_reduce** |
| `lm_head` | 按词表切 | `[151936/2, 1024]` | gather 到 rank 0 |
| RMSNorm、q_norm / k_norm、RoPE | 不切，每个 rank 一份完整的 | — | — |

q_norm / k_norm 是按 head 做的（对 `head_dim` 归一化），所以不用切，每个 rank 对自己的 head 用同一份权重。

---

## 第 0 步：环境验证

**为什么**：TP 依赖三样东西：两张卡之间 NCCL 能通；每张卡上的 kernel（FA2、Triton）能跑；CPU 上 gloo 能用（第 2、3 步的测试要用）。先用一个独立脚本确认，免得写到第 5 步才发现环境不行。

**怎么做**：`scripts/check_tp_env.py`，不依赖项目里的新代码：

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # 两张卡都空闲
uv run python scripts/check_tp_env.py
```

脚本做三件事：

1. 对每张卡：用很小的输入调用一次 `flash_attn_varlen_func` 和 `flash_attn_with_kvcache`（带 `block_table`），再调用一次项目的 Triton `store_kvcache`（包括 slot = -1 的行），检查结果。
2. gloo：用 `mp.get_context("spawn")` 起 2 个 CPU 进程，`init_process_group("gloo", "tcp://127.0.0.1:<空闲端口>")`，验证 `all_reduce` 求和、`gather` 拼接。
3. NCCL：同样起 2 个进程，rank i 用 `cuda:i`，先验证正确性，再测 all_reduce 的耗时。消息大小按 Qwen3-0.6B TP=2 每层实际发送的量：decode 32 条序列是 `[32, 1024]` bf16（64 KiB），prefill 4096 个 token 是 8 MiB。

写这类脚本要注意的几点（第 5 步也一样）：

- **必须用 spawn，不能用 fork**。fork 出来的子进程会继承父进程已经初始化的 CUDA 状态，无法使用。
- 端口用 `socket.bind(("127.0.0.1", 0))` 找一个空闲的，不要写死。
- 子进程的异常要传回主进程（脚本里用 `mp.Queue` 传 traceback），否则另一个 rank 会一直卡在 collective 上。
- 计时之前先 warm up 几次，并在计时前后 `torch.cuda.synchronize`。

**实测**（2026-11-24，torch 2.15.0+cu132，NCCL 2.30.7）：

```text
[cuda:0] NVIDIA GeForce RTX 5090 sm_120
  FA2 varlen + kvcache : OK
  Triton KV store      : OK
[cuda:1] NVIDIA GeForce RTX 4060 Laptop GPU sm_89
  FA2                  : FAIL  no kernel image is available for execution on the device
  Triton KV store      : OK
[gloo] 2 processes
  all_reduce + gather  : OK
[nccl] 2 processes
  all_reduce + gather  : OK
  all_reduce decode 32 seqs (64 KiB)   :     33.3 us  (1.97 GB/s)
  all_reduce prefill 4096 tok (8 MiB)  :   2243.7 us  (3.74 GB/s)
  -> per decode step, 56 all_reduces ~ 1.9 ms
```

**结论**：

- torch 的设备编号是 `cuda:0` = 5090、`cuda:1` = 4060（`CUDA_DEVICE_ORDER` 未设置时默认最快的卡排第一；`nvidia-smi` 是按 PCI 总线排序的，两者顺序相反）。所以 rank 0 用 5090，调度和采样都在快的那张卡上。
- NCCL 能用。`nvidia-smi topo -m` 显示两卡之间是 **PHB**，没有 P2P，数据要经过主机内存，带宽只有 2–4 GB/s。每个 decode 步光通信就要约 1.9 ms（TP=1 + CUDA Graph 时整步才 4.4 ms）；一个 4096 token 的 prefill 步通信约 126 ms（TP=1 整步约 38 ms）。
- Triton 在 4060 上没问题，TP 下继续用 Triton KV 写入。
- **FA2 在 4060 上不能用**。`cuobjdump --list-elf` 显示 `~/wheels` 里的 wheel 只编译了 sm_120（96 个 cubin）。FA2 本身支持 sm_80 以上的卡，并且 sm_80 的 cubin 可以直接在 sm_86 / sm_89 上运行（同一主版本内二进制兼容），所以只要重新编译时加上 sm_80，4060 就能用。

### 决定：TP 下的 attention 用哪个（已定：A）

| | A：TP 下用 eager attention | B：重新编译 FA2（sm_80 + sm_120） |
| --- | --- | --- |
| 代码 | TP=2 时手动传 `use_flash=False`（`--no-flash`），代码不做特殊处理 | 不需要任何特殊处理，两个 rank 都走 FA2 |
| 代价 | 无 | 重新编译 wheel：上次只编 sm_120，这次再加 sm_80，耗时大约翻倍 |
| TP=2 的 decode | eager attention 每层 × 每条序列一次 Python 循环，大约是第 1–3 步那个水平（约 250 tok/s），**主要测的是 eager 的开销，不是 TP** | 能和 TP=1 不开 graph（2717 tok/s）做有意义的对比；还能做下面的可选步骤（TP + CUDA Graph） |
| 和"只针对 5090"的原则 | 符合 | 也符合：只是换一个 wheel，不加任何 4060 专用代码 |

**M8 用 A**（2026-11-24 决定）：M8 只要验证 TP 是正确的，eager attention 是 M3–M7 一直用来做对照的路径，足够了；在这两张没有直连的卡上，即使都用 FA2，TP=2 也不会比单卡快，速度数据意义不大。不重新编译 wheel，不冒覆盖 wheel 的风险。

以后想要 B，代码不用改：`use_flash` 本来就由输入控制，重新编译 wheel 后不传 `--no-flash` 就是 B。

B 的做法：拿 flash-attention v2.8.4 源码，用 setup.py 支持的架构环境变量（2.8 版本是 `FLASH_ATTN_CUDA_ARCHS`，编译前先在 setup.py 里确认）设为 `80;120`，编译出 wheel。编完用 `cuobjdump --list-elf` 确认 sm_80 和 sm_120 都在，再跑一遍本脚本，两张卡都应该是 FA2 OK。注意：

- 先把现在的 wheel 另存一份再覆盖 `~/wheels` 里的文件；
- `flash_attn` 不在 `pyproject.toml` 里，`uv sync` 会把它删掉，装好之后用 `uv sync --inexact`。

---

## 第 1 步：单卡也初始化 process group（同 nano-vllm）

**为什么**：第 2–4 步的层和 nano-vllm 一样直接写 `dist.get_rank()` / `dist.get_world_size()` / `dist.all_reduce()`。这些函数在没有 `init_process_group` 时会报 "Default process group has not been initialized"，所以**任何建模型或建层的地方，之前都必须已经有一个 process group**。nano-vllm 在 `ModelRunner.__init__` 里无条件 init（TP=1 时 world_size=1），这一步做同样的事。

nano-vllm 不用处理、我们要处理的三件事：

1. **一个进程会建很多个 engine**（pytest、notebook）。process group 一个进程只能 init 一次，所以只在还没有时才 init，并记住是不是自己 init 的，只有自己 init 的才 destroy。
2. **测试里直接建模型和层**（`Qwen3ForCausalLM(config)`、`ModelRunner(...)`，不经过 engine）。这些之前没有人 init，用 conftest 里一个 session 级 fixture 兜底。
3. **端口不能写死 2333**：同时跑 pytest 和 benchmark 会冲突。用第 0 步的 `free_port()`。

**做什么**：

- 在 `engine/llm_engine.py`（或 `worker/model_runner.py`）加一个小函数，`from_config` 和 `benchmark.py` 在**建模型之前**都调用它：

  ```python
  def init_process_group(tp_size: int, device: torch.device) -> bool:
      """Make sure a process group of tp_size exists. Returns True if this call created it."""
      if dist.is_initialized():
          if dist.get_world_size() != tp_size:
              raise RuntimeError(f"process group has world_size {dist.get_world_size()}, "
                                 f"need {tp_size}; start TP={tp_size} in a fresh process")
          return False
      backend = "nccl" if device.type == "cuda" else "gloo"   # CPU engine 也要能跑
      dist.init_process_group(backend, init_method=f"tcp://127.0.0.1:{free_port()}",
                              world_size=tp_size, rank=0)
      return True
  ```

  返回值存到 runner 上（`self.owns_group`）。第 5 步的 `exit()` 里只有它为 True 才 `destroy_process_group()`。TP=1 时 world_size=1，层里的 `if tp_size > 1` 都不成立，不会真的通信；NCCL 在第一次集合通信时才建通信器，所以 TP=1 也几乎没有开销。

- `tests/conftest.py` 加 session 级 autouse fixture：

  ```python
  @pytest.fixture(scope="session", autouse=True)
  def _single_rank_group():
      if not dist.is_initialized():
          dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{free_port()}",
                                  world_size=1, rank=0)
      yield
      if dist.is_initialized():
          dist.destroy_process_group()
  ```

  gloo 是因为它在 CPU 上也能用；world_size=1 时不会通信，CUDA 测试也不受影响。engine 看到已经有 group，就不会再 init。

- **TP=2 的测试要在新进程里跑**：pytest 主进程里已经有一个 world_size=1 的 group，不能再建 world_size=2 的。第 7 步的测试用 `mp.get_context("spawn")` 起一个子进程，在里面建 TP=2 engine，把 logits / token 通过 Queue 传回来。第 2–3 步的 gloo 2 进程测试本来就是 spawn 的子进程，不受影响。

**验证**：`uv run pytest -q` 全部通过（这一步还没有层调用 dist，主要确认 fixture 和 engine 里的 init 不冲突）；再加两个小测试：
- 连续建两个 TP=1 engine 不报错，第二个不会再 init；
- 已有 world_size=1 的 group 时，要求 tp_size=2 会报上面那条 RuntimeError。

## 第 2 步：并行线性层

**为什么**：切分发生在线性层，加载时也要从完整的 checkpoint 张量里切出本 rank 那一块。

**做什么**（`layers/linear.py`，在现有 `_PackedLinear` 的基础上改）：

- `ColumnParallelLinear(input, output)`：本地权重 `[output / tp, input]`；加载时 `loaded.chunk(tp, dim=0)[rank]`。
- `RowParallelLinear(input, output)`：本地权重 `[output, input / tp]`；加载时按 dim 1 切；forward 后 all_reduce。有 bias 时只在 rank 0 加（Qwen3 没有 bias，但写对它不费事）。
- `QKVLinear` / `MergedLinear` 改成列并行：每个 shard（q / k / v，或 gate / up）**各自**按 tp 切，再放到本地权重里对应的位置。

最容易错的就是最后这一条。完整的 QKV 权重按行是 `[q 的 16 个 head | k 的 8 个 | v 的 8 个]`，直接把整块切成两半，rank 0 会拿到全部的 q、rank 1 拿到 k 和 v，**形状完全对得上，结果却是错的**。正确的做法是在 weight_loader 里拿到的已经是单个 shard（`q_proj.weight` 等），先切这个 shard，再写进本地的偏移位置：

```python
def weight_loader(self, param, loaded_weight, shard_id):
    offset, size = self.shard_offsets[shard_id]       # 本地（已经除以 tp）的偏移和大小
    loaded_weight = loaded_weight.chunk(self.tp_size, dim=0)[self.tp_rank]
    param.data.narrow(0, offset, size).copy_(loaded_weight)
```

`shard_offsets` 要用本地大小算（q 是 `num_heads / tp × head_dim`）。

**测试**（`tests/unit/test_tp_layers.py`）：CPU 上用 gloo 起 2 个进程（写法同第 0 步），每个 rank 用同一个种子生成完整权重和输入，走 weight_loader 加载自己那一份，检查：

- 列并行：本 rank 的输出等于完整输出的对应那几列；
- 行并行：all_reduce 之后等于完整输出；
- QKV / gate-up：本 rank 的 q、k、v 恰好是第 rank 组 head 的那几行。

CPU 测试用 fp32，结果可以精确比较（只要求和顺序相同）或用很小的容差。

## 第 3 步：词表并行的 embedding 和 LM head

**做什么**（新建 `layers/embed_head.py`）：

- `VocabParallelEmbedding`：本地存 `[vocab / tp, hidden]`，负责 `[rank × vocab/tp, (rank+1) × vocab/tp)` 这段词。forward 时把不在这一段的 token 的 id 置 0、输出也置 0，然后 all_reduce：

  ```python
  mask = (x >= start) & (x < end)
  y = F.embedding(mask * (x - start), self.weight)
  y = mask.unsqueeze(-1) * y
  if self.tp_size > 1:
      dist.all_reduce(y)
  return y
  ```

- `ParallelLMHead`：本地 logits 为 `[num_seqs, vocab / tp]`，gather 到 rank 0 后在最后一维拼接；其他 rank 返回 None。
- tie embedding：Qwen3-0.6B 的 `tie_word_embeddings=True`，`lm_head.weight` 直接指向 `embed_tokens.weight`。两者用同一种切法（都按词表切），所以共享的是切分后的那块，没有问题。
- 词表大小要能被 tp 整除（151936 / 2 可以）。不能整除时的补齐（vLLM 的做法）放到以后再做，这一步只做校验。

**测试**：同第 2 步，gloo 2 进程：embedding 结果等于完整 embedding；rank 0 拼起来的 logits 等于完整 logits。

## 第 4 步：模型和 KV cache 用本地大小

**做什么**：

- `Qwen3Attention`：`num_heads = total / tp`，`num_kv_heads = total_kv / tp`，`qkv_proj` 换成列并行，`o_proj` 换成行并行。
- `Qwen3MLP`：`gate_up_proj` 列并行，`down_proj` 行并行。
- `embed_tokens` / `lm_head` 换成第 3 步的版本。
- `Qwen3ForCausalLM.compute_logits` 在非 0 rank 上返回 None。
- `PagedKVCache` 和 `kv_block_bytes` 都用 `num_key_value_heads / tp`。

**验证**：TP=1 时以上全都退化成原来的层，M7 的全部测试（尤其是 `test_qwen3_parity` 和 HF 对比）必须照常通过。这一步是 M7 测试真正派上用场的地方。

## 第 5 步：多进程

**做什么**：

- `Config.tensor_parallel_size: int = 1`。
- engine 用 `mp.get_context("spawn")` 启动 rank 1 … tp−1，每个子进程运行一个函数：`dist.init_process_group("nccl", ..., world_size=tp, rank=i)` + `torch.cuda.set_device(i)` → 建模型、加载权重 → 建 KV cache → 进入 `loop()`。
- rank 0 的 `call(method, *args)`：tp > 1 时把 `pickle.dumps([method, *args])` 写进共享内存（前 4 个字节写长度），对每个 worker 的 `Event` 调用 `set()`，然后自己也执行这个方法。
- worker 的 `loop()`：`event.wait()` → 读出 (方法, 参数) → `event.clear()` → 执行 → 方法是 `"exit"` 时退出循环。
- 退出：rank 0 调用 `call("exit")`；`ModelRunner.exit()` 里关闭共享内存（rank 0 再 `unlink`）、`destroy_process_group()`（只在 `owns_group` 时）；engine 再 `join` 子进程。`atexit.register(self.exit)` 用于用户没有手动调用的情况。这就是 nano-vllm 里 `exit()` 和 `atexit` 那段代码的用途。

### 共享内存里存的是什么

**一句话**：只有 rank 0 写，其他 rank 只读。写的内容是一条命令，也就是"调用哪个方法、传什么参数"，不是张量。

**为什么需要**：只有 rank 0 有 Scheduler，知道这一步要算哪些 seq。其他 rank 也必须用**同样的** seqs 跑 `run`，否则每层的 all_reduce 对不上，程序会卡死。但不同进程的内存互相看不到，所以 rank 0 每一步都要把调度结果"告诉"它们。

**打个比方**：两个厨师在隔开的两个房间里，每道菜各做一半。只有主厨（rank 0）手上有菜单。两个房间之间挂着一块**黑板**（`SharedMemory`），还有一个**门铃**（`Event`）：

1. 主厨在黑板上写"run：这几条 seq，decode"（`write_shm`），然后按门铃（`event.set()`）；
2. 帮厨一直在等门铃（`loop()` 里的 `event.wait()`），听到后读黑板（`read_shm`），把门铃复位（`event.clear()`）；
3. 两人执行同一个 `run`，各算一半的 head，每层通过 NCCL all_reduce 把结果加起来。

黑板上写的只是**命令**；菜本身，也就是张量计算的结果，走的是 NCCL，不经过 shm。worker 从不往 shm 里写，也不需要回传结果：只有 rank 0 用 logits 做 sample，worker 的 `run` 返回 None。TP=3 时 rank 0 写一次、按两个门铃，两个 worker 读的是同一块黑板。

**格式**：黑板上只能写字节，所以 Python 对象要先 pickle：

```text
shm.buf: [ 4 字节：长度 n ][ n 字节：pickle.dumps((method, args)) ][ 没用到的部分 ]
```

一共只会出现两种命令：

| 时机 | 内容 |
| --- | --- |
| 每个 step | `("run", (seqs, is_prefill))`。seqs 是 `Sequence` 对象，带着 `token_ids`、`block_table`、`num_cached_tokens`、`num_scheduled_tokens` 等；worker 靠这些自己算出 input_ids、positions 和 slot_mapping |
| 退出时 | `("exit", ())` |

**只有一块黑板，rank 0 会不会把还没读完的命令覆盖掉？** 不会。`run` 里每层都有 all_reduce，rank 0 必须等所有 worker 进入 all_reduce 才能继续，所以等它执行到下一次 `write_shm` 时，所有 worker 早就读完了上一条。`exit` 里的 barrier 也起同样的作用。

**创建和销毁的顺序**：

```text
from_config:  rank 0 SharedMemory(create=True) → barrier → 其他 rank 按名字 attach
exit:         所有 rank close                  → barrier → rank 0 unlink（删掉 /dev/shm 里的文件）
```

**为什么 grep 不到 `ModelRunner.exit` 的调用**：`call` 和 `loop` 都用 `getattr(self, method)` 按名字分发（nano-vllm 也一样），调用方写的是字符串 `"exit"` / `"run"`。

**要注意的坑**：

| 坑 | 原因和处理 |
| --- | --- |
| **子进程里 `Sequence.block_size` 又变回 256** | spawn 会重新 import 模块，类属性不继承。worker 初始化时要从 config 里重新设置。测试用 block_size=4 时，如果忘了这一步，rank 1 算的 slot 会全部错位 |
| rank 1 报错后 rank 0 卡住 | rank 0 会一直等 all_reduce（NCCL 默认要等 10 分钟）。**已处理**：`init_process_group(timeout=60s)`（`process_group.TIMEOUT`）。实测：往 worker 的 `run` 里注入一个异常，worker 打印 traceback 后退出；rank 0 在 all_reduce 里等了 60 s，NCCL watchdog 判定超时，再等 15 s 做健康检查后 abort（退出码 134），`/dev/shm` 由 resource tracker 清理。没有用 Queue 传异常，简单一些 |
| worker 在 `event.wait()` 里被杀掉（SIGKILL / OOM killer），rank 0 永远卡住 | **未处理，已知限制**。`mp.Event.set()` 底层的 `Condition.notify` 会等每个正在 wait 的进程确认自己醒了，死掉的进程永远不会确认，于是 rank 0 卡在 `write_shm` 的 `event.set()` 里，根本走不到 all_reduce，NCCL 超时管不到。nano-vllm 也有同样的问题。真要处理的话，可以在 `call` 之前检查 `process.is_alive()`（仍有竞态），或者换掉 Event 的实现 |
| 第二次启动报 `FileExistsError` | 上次异常退出时共享内存没有 unlink。名字里带上 rank 0 的 pid |
| 端口冲突 | 不要像 nano-vllm 那样写死 2333，用第 0 步的 `free_port()` |
| pickle 的开销 | decode 时每步都传全部 seqs 的 `token_ids`。nano-vllm 在 `Sequence.__getstate__` 里只传 decode 需要的字段。先不做，第 8 步如果在 profile 里看到再优化 |

**验证**：起一个 TP=2 的 engine，调用一个什么都不做的方法，再 `exit()`，确认子进程已经退出、`/dev/shm` 里没有残留。

**实测（5090 + 4060，Qwen3-0.6B，`use_flash=False`，greedy，prompt 为 "The capital of France is"）**：engine 启动 4.1 s；两卡取最小后 KV block 数为 394（TP=1 只用 5090 时是 916）；`exit()` 后子进程 exitcode 为 0，`/dev/shm` 里没有残留，连续调用两次 `exit()` 也没问题。生成结果如下：

| dtype | TP=1 | TP=2 |
| --- | --- | --- |
| bf16（16 token） | Paris. The capital of **Italy is Rome**. The capital of Spain is Madrid. | Paris. The capital of **France is also** the capital of the Republic of France. |
| float32（32 token） | Paris. The capital of Italy is Rome. … The capital of Japan is Tokyo. The capital | **逐字相同** |

**bf16 下出现分叉不是 bug。** TP=2 时，每层 `o_proj` 和 `down_proj` 的输出由两张卡各算一半再 all_reduce 相加；TP=1 是在一个 matmul 里一次加完。浮点加法不满足结合律，加法顺序变了，结果就会差最后几位。bf16 只有 8 位尾数，这点误差在 28 层里不断累积，到 logits 上已经足够让两个分数接近的候选 token 换位（这里是 "Italy" 和 "France"）。greedy 一旦在某一步选了不同的 token，后面的上下文就不同了，整段输出都会走向另一条路径。float32 下误差小了几个数量级，翻不动 argmax，所以输出逐字相同。

所以 bf16 下 TP=2 不能要求逐 token 和 TP=1 相同。要验证正确性，可以用 float32 比较输出，或者按第 7 步的做法 teacher forcing 并给 logits 设容差。

## 第 6 步：接线

- `Config` 校验（`raise ValueError`）：`tensor_parallel_size >= 1`，并且能整除 `num_attention_heads` 和 `num_key_value_heads`。`intermediate_size` 和 `vocab_size` 不在 config 里查，和 nano-vllm 一样由层自己检查：列并行 / 行并行层用 `divide()`（第 2 步），`VocabParallelEmbedding` 里 `assert vocab % tp == 0`（第 3 步）。GPU 数量和 CUDA 不单独检查，卡不够时 worker 的 `set_device` 会直接报错。
- attention：`use_flash` 完全由输入控制，代码里不为 TP 做特殊处理（只针对 5090，不加其他卡的适配）。注意：默认 `None` 只按 rank 0 的 5090 判断，会变成 True，rank 1 在 4060 上第一次 prefill 就报 `no kernel image`；所以这台机器上跑 TP=2 **必须手动传 `use_flash=False` / `--no-flash`**（方案 A）。
- CUDA Graph：tp > 1 时在 `Config.__post_init__` 里强制 `enforce_eager = True`。这一条和显卡无关，是正确性：graph 的 warmup / capture 里有 NCCL all_reduce，M8 没有实现各 rank 同步录制，只有 rank 0 录会一直卡住。
- KV block 数：每个 rank 按自己卡上的空闲显存算出 `num_blocks`，然后 `all_reduce(MIN)` 取最小值。block_table 是所有 rank 共用的，任何一个 rank 都不能多分。4060 只有 8 GB，最后多半是它决定 block 数。
- `gpu_memory_utilization` 对每张卡分别生效。
- `benchmark.py`：加 `--tp`。

### 可选：TP + CUDA Graph

NCCL 的 all_reduce 可以录进 CUDA Graph，nano-vllm 在 TP 下也会录。前提是两个 rank 都用 FA2（方案 B），并且两个 rank 同时、用相同的档位顺序录制。M8 用方案 A，不做这一步；留给以后。

## 第 7 步：双卡测试（`tests/cuda/test_tensor_parallel.py`）

少于 2 张卡时跳过。

- **逐步比较 logits**：思路和 M6 第 4 步的 `test_model_steps_match_eager` 一样，但 TP=1 和 TP=2 不能在同一个进程里（第 1 步：层直接读 `dist.get_world_size()`，一个进程只有一个 group）。做法：spawn 一个子进程，在里面建 TP=2 engine，按固定的请求和配置一步步跑，把每步的 logits（转 CPU、float）和选出的 token 通过 Queue 传回来；pytest 主进程（world_size=1）用同样的配置建 TP=1 的模型和调度器，每步用子进程传回的 token 做 teacher forcing，比较 logits。调度是确定的，两边每步的 batch 相同。覆盖 chunked prefill、前缀命中、decode、抢占。bf16 下 all_reduce 的求和顺序不同，比较时要给容差，不能直接比较生成的 token（理由同 M6 第 4 步：top-2 相等时会翻转）。
- **复用 M7 的不变量**：TP=2 时每步之后在 rank 0 上调用 `check_invariants`。
- **真实模型冒烟测试**：`LLM(~/models/Qwen3-0.6B, tensor_parallel_size=2).generate(...)` 能生成合理的文本。
- **清理**：engine `exit()` 之后子进程已经退出，`/dev/shm` 里没有残留；同一个子进程里连续创建两个 TP=2 engine 也没问题（第一个 exit 时 destroy 了 group，第二个重新 init）。

## 第 8 步：测量和写文档

```bash
uv run python -m pico_vllm.benchmark --device cuda:0 --model ~/models/Qwen3-0.6B --no-flash            # TP=1 对照（eager attention，不录 graph）
uv run python -m pico_vllm.benchmark --device cuda:0 --model ~/models/Qwen3-0.6B --no-flash --tp 2
```

TP=2 必须加 `--no-flash`（不加会在 4060 上报 `no kernel image`）；TP=1 对照也要加，两边都是 eager attention 才能比。各跑 2 遍取平均。建议另外记录：

- 一步里通信占多少时间（`torch.profiler`，或者在 all_reduce 前后打 CUDA event）；
- 两张卡的 KV block 数，以及最后取的最小值；
- 4060 单卡算一层的时间：TP 下每一步都要等较慢的那张卡。

写进 `docs/m8-tensor-parallel.md`，更新 README 的状态表和 `docs/design-vs-nano-vllm.md` 的张量并行那一行。提交信息：`feat: add tensor parallelism`。

## 代码量预估

生产代码 +300–450 行（并行层和多进程 runner 占大头），测试 +250–350 行。
