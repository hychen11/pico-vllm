# decode 的 FlashAttention 输入，以及为什么 decode 能录 CUDA Graph

**结论**：decode 时每条序列确实要和它前面的全部 token 做 attention，但"前面有多少个 token"不体现在**张量形状**上，而是写在两个 GPU 张量的**内容**里：`cache_seqlens`（每条多长）和 `block_table`（存在哪些块）。K/V 不是每步重新传入的张量，而是整个 KV cache 池，形状和地址永远不变。所以只要 batch 大小相同，每一步的输入形状就完全相同，可以录成 CUDA Graph 反复回放。

## 1. 一个例子

Qwen3-0.6B：16 个 query head，8 个 KV head（GQA），head_dim = 128。KV cache 池有 32 个块，每块 256 个 token。

正在 decode 的 3 条序列：

| 序列 | 已有 token 数（含本步） | 存放的物理块 |
| --- | --- | --- |
| A | 5 | [7] |
| B | 300 | [3, 12] |
| C | 1000 | [20, 5, 9, 14] |

这一步，每条序列只算 **1 个新 token**，但这个 token 要和自己前面的**所有** token 做 attention：A 看 5 个，B 看 300 个，C 看 1000 个（都包括它自己，它的 K/V 已经先写进 cache）。

## 2. 传给 `flash_attn_with_kvcache` 的东西

```python
out = flash_attn_with_kvcache(
    q,                          # [3, 1, 16, 128]
    k_cache, v_cache,           # [32, 256, 8, 128]，整个 KV 池
    cache_seqlens=context_lens, # [3]，内容 = [5, 300, 1000]
    block_table=block_table,    # [3, 4]，int32
    softmax_scale=..., causal=True,
)                               # out: [3, 1, 16, 128]
```

| 参数 | 形状 | 含义 | 每步变不变 |
| --- | --- | --- | --- |
| `q` | `[batch, 1, num_heads, head_dim]` = `[3, 1, 16, 128]` | 每条序列本步那 1 个新 token 的 query。中间的 1 表示每条序列只有 1 个 query | 形状不变，内容变 |
| `k_cache` / `v_cache` | `[num_blocks, block_size, num_kv_heads, head_dim]` = `[32, 256, 8, 128]` | **整个** KV cache 池，所有序列共用。不是"这条序列的 K/V" | 形状和地址都不变；本步只往里写了 3 行新 K/V |
| `cache_seqlens` | `[batch]` = `[3]` | 每条序列要看多少个 token：`[5, 300, 1000]` | 形状不变，内容每步 +1 |
| `block_table` | `[batch, 最多块数]` = `[3, 4]` | 每条序列的物理块号，不够的位置用 -1 补齐 | 形状不变，偶尔多一个块号 |
| 输出 | `[3, 1, 16, 128]` | 每条序列本步 token 的 attention 结果 | 形状不变 |

`block_table` 补齐后的内容：

```
A: [ 7, -1, -1, -1]   ← 只用第 1 块的前 5 个位置
B: [ 3, 12, -1, -1]   ← 块 3 全部 256 个 + 块 12 的前 44 个
C: [20,  5,  9, 14]   ← 3 整块 + 块 14 的前 232 个
```

## 3. kernel 内部怎么做

GPU 上大致按"每条序列 × 每个 head"分工（长序列还会把 KV 再切成几段并行算，最后合并）。以序列 B 的第 0 个 query head 为例：

1. 取出 `q[1, 0, 0, :]`，也就是 B 这个新 token 在 head 0 上的 128 维 query。
2. GQA：16 个 query head 共用 8 个 KV head，每 2 个 query head 共用 1 个 KV head。head 0 对应 KV head 0。
3. 读 `cache_seqlens[1] = 300`，知道要看 300 个 token。
4. 按 `block_table[1] = [3, 12]` 依次去 KV 池里取：块 3 的第 0–255 号位置，块 12 的第 0–43 号位置，共 300 个 key 和 value。
5. 算 300 个注意力分数 → softmax → 对 300 个 value 加权求和 → 得到 128 维输出，写到 `out[1, 0, 0, :]`。

"和前面所有 token 交互"发生在第 3、4 步：kernel **运行时**才读出长度和块号，再决定循环多少次、去哪里取数据。B 的块 12 只写了 44 个 token，kernel 读到第 300 个就停，后面没用的位置不会读（半满的块和前缀缓存的区别见 [request-lifecycle-prefix-cache.md](request-lifecycle-prefix-cache.md) 的"常见疑问"）。

`causal=True` 在 decode 时没有实际作用：只有 1 个 query，而且它就是最后一个位置，本来就能看到前面全部 token。

## 4. 下一步会怎样

每条序列各生成 1 个 token 后，下一步的输入变成：

| | 第 1 步 | 第 2 步 |
| --- | --- | --- |
| `q` 形状 | `[3, 1, 16, 128]` | `[3, 1, 16, 128]` |
| `k_cache` 形状 / 地址 | `[32, 256, 8, 128]` / 不变 | 同左 |
| `cache_seqlens` 内容 | `[5, 300, 1000]` | `[6, 301, 1001]` |
| `block_table` 内容 | 同上 | 同上（C 到 1024 之前都不需要新块） |

**变的只有内容，形状一个都没变。** 这就是"decode 的形状一样"的意思：形状只取决于 batch 大小（3）和 `block_table` 的宽度（4），和每条序列有多长无关。

## 5. 对比 varlen：为什么 prefill 录不了

`flash_attn_varlen_func`（字段含义见 [attention-context.md](attention-context.md)）有两个问题：

- **q 的形状**是 `[本步 token 总数, ...]`。prefill 时每步的 token 总数都不同。
- **`max_seqlen_q` / `max_seqlen_k` 是 Python 整数**。录 graph 时它们会被当作常数写进 kernel 参数，以后回放还是旧值，结果就会出错。decode 时 `max_seqlen_k` 每步都 +1，所以即使是 decode，用 varlen 也录不了。

## 6. CUDA Graph 为什么只要求"形状和地址不变"

CUDA Graph 把一整步要启动的所有 kernel（28 层的矩阵乘、norm、attention 等等）**录制一次**。之后每步只需要一次"回放"，省掉 CPU 逐个发起几千个 kernel 的开销。录制时，每个 kernel 的输入地址、张量形状、整数参数都被固定下来；回放时只有**显存里的内容**可以变。

所以 decode 的做法是：

1. 启动时为每种 batch 大小分配**固定缓冲区**：`input_ids[bs]`、`positions[bs]`、`slot_mapping[bs]`、`context_lens[bs]`、`block_tables[bs, 最多块数]`、输出 `[bs, hidden]`，然后用它们各录一个 graph。
2. 每个 decode 步：把本步的数据**拷进**这些缓冲区（地址不变，只改内容），然后回放对应的 graph。实际条数小于录制的 batch 大小时怎么补齐，见第 8 节。

`block_table` 的宽度要在录制时固定下来，通常取 `max_model_len / block_size`。FA2 也是按这个**宽度**（而不是每条序列的实际长度）来决定 KV 要切几段并行，所以实际长度变化不会影响录好的 graph。

## 7. 实验验证（5090，FA2 2.8.4）

用上面的例子录一个只包含 `flash_attn_with_kvcache` 的 CUDA Graph，录制时的长度是 `[5, 300, 1000]`。之后只改缓冲区内容、不重新录制，用三组长度回放，并和 eager 参考实现（`layers/attention.py` 的 `causal_attention`）对比：

| 回放时的 `cache_seqlens` | 所有输入的形状和地址 | 和 eager 的最大误差 |
| --- | --- | --- |
| [5, 300, 1000]（录制时的值） | 和录制时相同 | 0.0064 |
| [6, 301, 1001] | 和录制时相同 | 0.0048 |
| [200, 512, 1020] | 和录制时相同 | 0.0013 |

误差在 bf16 的正常范围内。也就是说，graph 录一次就能用于任意长度，只要 batch 大小和 `block_table` 宽度不变。

## 8. batch 大小也会变，怎么办

会变，而且经常变：

- 有序列生成了 EOS 或到了 `max_tokens`，就结束退出：batch 变小。
- 新请求做完 prefill，进入 decode：batch 变大。
- 显存不够时调度器抢占序列，把它放回 waiting 队列：batch 变小。

batch 大小决定了 `q`、`cache_seqlens`、`block_table` 的第一维，所以**不同的 batch 大小需要不同的 graph**。做法是：

1. **预先录多个 graph**，每个对应一个"档位"。nano-vllm 的档位是 1、2、4、8，然后从 16 开始每 16 一档：16、32、48……一直到 `min(max_num_seqs, 512)`。
2. **每步挑一个大于等于实际条数的最小档位**，多出来的行作为空位：`cache_seqlens` 填 0，`block_table` 填 -1，`slot_mapping` 填 -1（KV 写入 kernel 跳过这些行）。空位算出来的输出直接丢掉。
3. **实际条数超过最大档位时**不用 graph，按普通方式执行（nano-vllm 里是超过 512 条时）。

例子（档位 1、2、4、8）：

| 这一步发生了什么 | 实际条数 | 用哪个 graph | 空位行数 | 和 eager 的最大误差 |
| --- | --- | --- | --- | --- |
| A、B、C 在 decode | 3 | bs = 4 | 1 | 0.0053 |
| A 结束了 | 2 | bs = 2 | 0 | 0.0011 |
| 又有 3 个新请求加入 | 5 | bs = 8 | 3 | 0.0071 |

（上表是在 5090 上实际运行的结果：4 个档位各录一个只包含 `flash_attn_with_kvcache` 的 graph，按上面的规则补齐后回放。）

档位设置是一个取舍：

- **档位越密**，补的空位越少，白算的就越少。但要录的 graph 越多，启动越慢，也占更多显存。
- 小 batch 时用 1、2、4、8 这样的密档位，因为在线服务常常只有几条请求；大 batch 时每 16 一档就够了，多算几行占比很小。
- 所有 graph 共用一个显存池（nano-vllm 里是 `graph_pool`），并且从最大的档位开始录，后面小档位的中间结果可以复用同一块显存。

## 9. 对本项目的意义

- 本项目用统一的 `prepare_inputs`：decode 只是"每条 q_len = 1"的特例，M6 第 4 步用 `flash_attn_varlen_func` 统一处理，结果是对的，但录不了 graph（第 5 节）。
- 做 CUDA Graph（M6 第 6 步）时，加一条"纯 decode 快速路径"：本步每条序列 q_len 都是 1，且条数不超过录制的最大 batch → 把数据拷进固定缓冲区，用 `flash_attn_with_kvcache` 并回放 graph；其他情况仍走统一的 varlen 路径。
- 还需要做两处准备：Triton KV 写入 kernel 要支持 `slot == -1` 时跳过；`cache_seqlens` 要有一个 int32 GPU 张量（现在的 `context_lens` 是 Python list），`block_table` 的宽度要固定成 `max_model_len / block_size`（现在按本步最多块数补齐）。
