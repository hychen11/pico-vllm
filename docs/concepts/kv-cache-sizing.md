# KV cache 能分到多少个 block：按 GPU 显存自动计算

**结论**：CUDA 上不传 `kvcache_capacity_bytes` 时，引擎在加载完权重之后、分配 KV cache 之前，先用假数据真实地跑一次"最大的一步"，测出这一步临时要用多少显存（activation 峰值），再把剩下的显存全部分给 KV cache：

```text
KV 可用字节 = 总显存 × gpu_memory_utilization − 已用显存 − activation 峰值
block 数   = KV 可用字节 // 每个 block 的字节数
```

代码在 `engine/llm_engine.py` 的 `num_kv_blocks` 和 `worker/memory.py`。公式和 nano-vllm 相同。

本文的数字都来自 RTX 5090、真实 Qwen3-0.6B、默认参数：`max_num_batched_tokens=16384`、`max_num_seqs=512`、`max_model_len=4096`、`block_size=256`、bf16、FA2。

## 显存被分成哪几块

```text
|<───────────────────────── 总显存 31.36 GiB ──────────────────────────>|
|<──────────────── 上限：× gpu_memory_utilization (0.9) = 28.22 GiB ─────────────>|        |
| 已用 2.42 GiB      | activation 预留 1.01 GiB | KV cache 24.77 GiB（906 个 block） |余量 | 不用 |
| CUDA context、权重、 | 一步运行中途的临时张量     | 长期存放的 K/V                      |     |      |
| 其他进程            |                          |                                    |     |      |
```

| 部分 | 是什么 | 怎么得到 |
| --- | --- | --- |
| 已用显存 | 本进程的 CUDA context 和权重，加上 GPU 上**其他进程**占的显存 | `torch.cuda.mem_get_info()` 直接问驱动 |
| activation 峰值 | 一步 forward 中途同时存活的临时张量：q/k/v、MLP 中间结果、logits、采样时的 fp32 副本等。用完就释放，但峰值时刻必须有地方放 | warmup 实测（见下文） |
| KV cache | 所有序列的 K/V，长期存放 | 上面两项扣掉后剩下的全部 |

`gpu_memory_utilization` 留出的 10% 是安全余量，也给 PyTorch 分配器的碎片留空间。

## 算一遍

```text
总显存          31.36 GiB
× 0.9         = 28.22 GiB   本引擎最多能用
− 已用            2.42 GiB   CUDA context + 权重（此时 GPU 上没有其他进程）
− activation     1.01 GiB   warmup 测得 1039 MiB
= KV 可用       24.79 GiB

每个 block = 2（K 和 V）× 28 层 × 256 token × 8 个 KV head × 128 head_dim × 2 字节（bf16）
           = 28 MiB
block 数   = 24.79 GiB // 28 MiB = 906
```

906 个 block × 256 = 231936 个 token 的 KV。原来固定 2 GiB 时只有 73 个 block（18688 个 token），是现在的 1/12。

对应的函数：

| 函数 | 做什么 |
| --- | --- |
| `kv_block_bytes`（`core/paged_kv_cache.py`） | 每个 block 的字节数，上面的 28 MiB |
| `profile_activation_bytes` | 跑 warmup，返回 activation 峰值 |
| `num_blocks_from_gpu_memory` | 调 `mem_get_info` 拿到总显存和已用显存，再调下一个函数 |
| `num_blocks_from_memory` | 纯算术：`max(总 × 比例 − 已用 − activation, 0) // 每块字节数`。不需要 GPU，单元测试可以直接代入数字 |
| `num_kv_blocks`（`engine/llm_engine.py`） | 决定用哪种方式算（见"三种情况"），一个 block 都放不下时报错 |

## activation 峰值怎么测

### 两个计数器

PyTorch 一直记录着本进程分配的显存：

| 名字 | 含义 |
| --- | --- |
| `torch.cuda.memory_allocated()` | **此刻**还存活的张量一共占多少 |
| `torch.cuda.max_memory_allocated()` | 从上次重置起，上面那个值**出现过的最高点** |
| `torch.cuda.reset_peak_memory_stats()` | 把最高点重置为当前值，从现在开始重新记 |

一步 forward 跑完后，中间结果都已经释放，`memory_allocated` 会回到原来的值，中途涨了多少从它看不出来。但 `max_memory_allocated` 记下了中途的最高点。所以做法是：跑之前重置并记下起点 `base`，跑完后用 `max_memory_allocated() − base`，得到的就是这一步临时多用的显存。

### 跑哪一步：调度器能产生的最大的两种步

调度器保证每一步不超过 `max_num_batched_tokens` 个 token、`max_num_seqs` 条序列，所以最坏情况是确定的，有两种：

| | 塞满的是什么 | 默认参数下 | 哪部分最大 |
| --- | --- | --- | --- |
| 最大的 prefill 步 | token 数：`max_num_batched_tokens // max_model_len` 条、每条 `max_model_len` 长（`warmup_shape`） | 4 条 × 4096 = 16384 个 token | 28 层的中间结果（要处理 16384 个 token） |
| 最大的 decode 步 | 序列数：`max_num_seqs` 条，每条算 1 个 token、上下文 `max_model_len` 长 | 512 条 × 1 个 token | logits 和采样（512 行 × 151936 个词，转 fp32 后做 softmax） |

两种都要跑，取峰值的最大值。只测 prefill 会漏掉 decode 的 logits，那部分实测比 prefill 还大。

prefill 用最长的序列（而不是很多条短序列），是因为 eager attention 的打分矩阵是"长度²"，长序列才是最坏情况。

### 为什么需要临时 KV cache 和 runner

- **临时 cache**：attention 层总是先把 K/V 写进 cache，再从 cache 里读出来计算，没有 cache 就没法跑 forward。但正式 cache 的大小正是我们要算的，这时还不能分配。所以先分配一个**刚好够 warmup 用**的小 cache：4 条 × 16 个 block = 64 个 block（1.75 GiB），测完就删。
- **runner**：attention 需要的 `slot_mapping`、`cu_seqlens_q/k`、`block_table` 等都由 `ModelRunner.prepare_inputs` 构造。用真正的 runner，warmup 走的就是和正式运行完全一样的代码路径，测出来的显存才可信。

### decode 步怎么构造

真跑 512 条序列的 decode，看起来要先给它们分配 block、再做 prefill。这里的做法是让它们**共用 prefill 刚写好的 KV**：

```text
prefill 写好的：  序列 0 → block 0..15    序列 1 → block 16..31
                 序列 2 → block 32..47   序列 3 → block 48..63

decode 序列 0、4、8、…   的 block_table = block 0..15
decode 序列 1、5、9、…   的 block_table = block 16..31
…（第 i 条用 prefill 序列 i % 4 的 block）

每条 decode 序列：token_ids 长 4096，num_cached_tokens = 4095，num_scheduled_tokens = 1
```

- 每条 decode 序列都以为自己的 prompt 已经全部在 cache 里，现在只算最后 1 个 token。
- 这 4 条 prompt 内容都是 `[0] * 4096`，完全相同。真实运行时 512 个相同 prompt 的请求也会通过前缀缓存共用同一批 block，这里只是手工做了同样的事。
- 每条序列重算自己的最后一个 token，并写回原来的位置，512 条会写同一批位置、互相覆盖。这一步只测显存，不关心算出的值，所以没有影响。

### 逐步看数字

| 阶段 | `memory_allocated` | `max_memory_allocated` |
| --- | ---: | ---: |
| 1. 加载完权重 | 1.66 GiB | 1.95 GiB |
| 2. 分配临时 cache（64 个 block） | 3.41 | 3.41 |
| 3. `reset_peak_memory_stats()`，记 `base = 3.41` | 3.41 | 3.41 |
| 4. prefill + 采样跑完 | 3.41 ← 中间结果已释放 | 4.13 ← 中途到过这里 |
| 5. decode + 采样跑完 | 3.55 | **4.42** |

`activation 峰值 = 4.42 − 3.41 = 1039 MiB`。prefill 段约 744 MiB，decode 段把峰值推到了 1039 MiB。

测完删掉临时 cache 和 runner，再 `empty_cache()` 把显存还给 GPU，这时 `mem_get_info` 看到的已用显存是 2.42 GiB，就是上面公式里的"已用"。

### 耗时

| attention | prefill 段 | decode 段 | 合计 | activation 峰值 | block 数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FA2（默认） | 0.21 s | 0.08 s | 0.30 s | 1039 MiB | 906 |
| eager（`use_flash=False`） | 0.65 s | 1.59 s | 2.26 s | 1472 MiB | 891 |

取进程内第二次运行的耗时；第一次含 kernel 编译，FA2 约 1.3 秒。eager 的峰值更高，因为它为每条 prompt 生成完整的 4096 × 4096 打分矩阵；decode 段慢 20 倍，因为 eager 要对"每层 × 每条序列"逐条循环（28 × 512 = 14336 次），时间主要花在 CPU 发起 kernel 上。

## 启动前先清理一次：`gc.collect()` 和 `empty_cache()`

`LLMEngine.from_config` 在 CUDA 上加载权重**之前**会先执行：

```python
gc.collect()               # 回收本进程里已经没人用、但还没释放的对象
torch.cuda.empty_cache()   # 把 PyTorch 缓存的空闲显存还给 GPU
model = build_model(config)
```

它清理的是**本进程自己留下的垃圾**，最典型的是同一个进程里之前创建过、又被 `del` 掉的 `LLM`。它不会动正在使用的对象，也释放不了其他进程的显存。

| 显存来源 | 能清掉吗 |
| --- | --- |
| 本进程里已经不用的对象（比如旧的 `LLM`） | ✅ 这就是它的作用 |
| 本进程里还在用的对象（当前权重、你手上的张量） | ❌ 不会动 |
| 其他进程（比如另一个推理服务） | ❌ 动不了，仍算在"已用"里，KV cache 相应变小 |
| CUDA context | ❌ 进程只要用了 GPU 就有 |

### 为什么需要：两层原因

**第一层，`del` 不一定马上释放对象。** Python 主要靠引用计数释放对象：没有引用时立即释放。但几个对象互相引用、形成环时，计数永远降不到 0，只能等循环垃圾回收器来清理。回收器是按 Python 对象的分配次数定期触发的，不关心 GPU 显存是否紧张。实测（4060，固定 1 GiB cache）：

```text
LLM 建好后      memory_allocated = 2.64 GiB
del llm        memory_allocated = 0.55 GiB   ← KV cache 立即释放，还有 0.55 GiB 在环里
gc.collect()   memory_allocated = 0.00 GiB
```

**第二层，释放的显存 PyTorch 不会马上还给 GPU。** 张量释放后，显存留在 PyTorch 的缓存池里供以后复用，`mem_get_info` 仍把它算作已用。`empty_cache()` 能把缓存还给 GPU，但只能归还**整段完全空闲**的显存，一段里只要还有一个张量活着，整段都还不回去。

### 两层叠加的后果

同一进程里删掉旧引擎、再建新引擎时：

```text
1. del 旧 llm       旧 KV cache（约 25 GiB 的一大段）释放，留在 PyTorch 缓存里；旧模型还在环里
2. build_model      新权重优先复用缓存，放进了旧 KV 那一段
3. empty_cache      那一段里有新权重，还不回去
4. mem_get_info     这一段剩下的空闲部分也被算成"已用" → block 数变少
```

5090 上连续创建 3 个引擎：不清理时 block 数依次为 906、884、884；清理后每次都是 906。回归测试：`tests/cuda/test_kv_sizing_cuda.py::test_second_engine_gets_the_same_blocks`。

**顺序很重要**：先 `gc.collect()`，旧对象的显存才真正释放，`empty_cache()` 才能把它还回去；两步都要在 `build_model` 之前，否则新权重已经落进旧段，就来不及了。

一个进程只建一个 `LLM` 时没有垃圾可清，这两步几乎不花时间。它们主要服务于测试、notebook 这类会反复创建引擎的场景。

## 三种情况

`num_kv_blocks` 按下面的顺序决定 block 数：

| 条件 | block 数 |
| --- | --- |
| 传了 `kvcache_capacity_bytes` | `capacity // 每块字节数`，不跑 warmup |
| 没传，且在 CUDA 上 | 本文描述的自动计算 |
| 没传，且在 CPU 上 | 按 `DEFAULT_CPU_KVCACHE_BYTES`（2 GiB）计算 |

算出来是 0 时直接报 `ValueError`，信息里写明已用显存和 activation 预留量。例如 GPU 已经被其他进程占了 93%，而 `gpu_memory_utilization=0.9`：

```text
no KV-cache block (8192 bytes) fits with gpu_memory_utilization=0.9 (... GiB already in use,
... GiB reserved for activations); raise it, or lower max_num_batched_tokens / max_num_seqs / kvcache_block_size
```

## 常见疑问

**为什么要实测 activation，不直接用公式算？** 它取决于很多细节：eager 还是 FA2（eager 生成完整的打分矩阵）、采样时 fp32 副本有几份、PyTorch 分配器的对齐等。手写公式容易漏项，直接跑一次真实的代码路径最可靠。

**`max_num_batched_tokens` 是怎么来的？** 由使用者设置，默认 16384。它是调度器每一步的 token 上限，长 prompt 会被切成多段（chunked prefill）。设大一点 prefill 更快，但 activation 更大、KV cache 更小。正因为调度器保证不超过它，warmup 才能确定最坏的一步有多大。

**和 nano-vllm 有什么不同？** 公式相同，nano-vllm 写成 `total × util − used − peak + current`（`model_runner.py` 的 `allocate_kv_cache`），其中 `peak − current` 就是 activation 峰值。nano-vllm 的 warmup 只跑 prefill，在分配 KV cache 之前进行，attention 走"不读 cache"的分支；本项目的 attention 总是先写 cache 再读，所以用临时 cache，并额外跑一次 decode，把 512 行 logits 的采样也测进去。启动前的 `gc.collect()` 也是本项目加的。
