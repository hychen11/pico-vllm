# attention 的输入：Context 里的各个字段

**结论**：每一步，`ModelRunner.prepare_inputs` 把所有序列的 token 首尾相接拼成一长条，再用 `Context` 记录 attention 需要的信息。核心只有一条规则：

- **Q 只放这一步新算的 token**：已有 KV 的 token 输出早就算过了，不用再算 query。
- **K/V 要读这条序列的全部 token**：新 token 要和它前面的每个 token 做 attention，包括缓存命中的、前几步写的、这一步刚写的。

所以对每条序列，q 的长度是 `end - start`，k 的长度是 `end`。

## attention 里 FA2 的调用需要哪些参数

`layers/attention.py` 先用 `kv_cache.write` 把本步的 K/V 写进 cache，再调用：

```python
flash_attn_varlen_func(
    q,
    kv_cache.k_cache[self.layer_idx],
    kv_cache.v_cache[self.layer_idx],
    context.cu_seqlens_q,
    context.cu_seqlens_k,
    context.max_seqlen_q,
    context.max_seqlen_k,
    softmax_scale=self.scale,
    causal=True,
    block_table=context.block_table,
)
```

以 Qwen3-0.6B 为例：16 个 query head，8 个 KV head，head_dim = 128。"例子"一列用下文的 prefill 步：A 是全新的 5 个 token，B 有 6 个 token、前 4 个命中缓存；`block_size = 4`，KV 池共 10 个块。

| 参数 | 形状 / 类型 | 从哪来 | 含义 | 例子 |
| --- | --- | --- | --- | --- |
| `q` | `[num_tokens, 16, 128]` | 这一层 q_proj + RoPE 的输出 | 本步所有序列新算的 token 的 query，首尾相接 | `[7, 16, 128]`：a0..a4、b4、b5 共 7 行 |
| `k`, `v` | `[num_blocks, block_size, 8, 128]` | `kv_cache.k_cache[layer_idx]` / `v_cache[layer_idx]` | 这一层的**整个**分页 KV 池，不是本步的 k、v | `[10, 4, 8, 128]`：本步的 7 个 K/V 已经先写进去了 |
| `cu_seqlens_q` | int32 `[num_seqs + 1]` | `Context`，`prepare_inputs` 生成 | 每条序列的 query 在 `q` 里的起止行 | `[0, 5, 7]`：A 是 `q[0:5]`，B 是 `q[5:7]` |
| `cu_seqlens_k` | int32 `[num_seqs + 1]` | 同上 | 相邻两项的差 = 每条序列要读的 KV 数 | `[0, 5, 11]`：A 读 5 个，B 读 6 个 |
| `max_seqlen_q` | Python int | 同上 | 最长的 query 数，决定启动多少线程块 | `5`（A 有 5 个，B 有 2 个） |
| `max_seqlen_k` | Python int | 同上 | 最长的 KV 数 | `6`（A 读 5 个，B 读 6 个） |
| `block_table` | int32 `[num_seqs, max_num_blocks]` | 同上 | 每条序列的物理块号，用 -1 补齐 | `[[3, 8], [5, 9]]`：B 的 b0..b3 在块 5，b4、b5 在块 9 |
| `softmax_scale` | float | `Attention.scale` | 分数缩放，`1 / sqrt(head_dim)` | `1 / sqrt(128) ≈ 0.0884` |
| `causal` | bool | 固定 `True` | 因果 mask，q 比 k 短时右下角对齐 | B 的 b4 看 b0..b4，b5 看 b0..b5 |

返回 `[num_tokens, 16, 128]`，和 `q` 形状相同。GQA 由 FA2 自己处理（query head h 读 KV head `h // 2`），不用复制 K/V。下面几节解释 `Context` 里那几个字段是怎么算出来的。

本文的数值都按 `prepare_inputs` 的代码计算。为了好看，`block_size = 4`。FA2 实际要求 `block_size` 是 256 的倍数，但各字段的算法和块大小无关。

## start 和 end

```python
start = seq.num_cached_tokens               # 前 start 个 token 的 KV 已经在 cache 里
end   = start + seq.num_scheduled_tokens    # 这一步算 [start, end) 这些 token
```

```
位置:   0   1  ...  start-1 | start  ...  end-1
        └─ 已有 KV，不再计算 ─┘ └─ 这一步新算的 ─┘
         （缓存命中 / 之前的步骤）
```

## 例子：一个 prefill 步

- A：全新的 5 个 token 的 prompt，没有缓存，`block_table = [3, 8]`。
- B：6 个 token 的 prompt，前 4 个命中前缀缓存，这一步只算 b4、b5，`block_table = [5, 9]`。chunked prefill 的后续段也是这种情况。

| 序列 | start | end | q 长度 = end − start | k 长度 = end |
| --- | --- | --- | --- | --- |
| A | 0 | 5 | 5 | 5 |
| B | 4 | 6 | 2 | 6 |

循环里逐条累加：

```python
cu_seqlens_q.append(cu_seqlens_q[-1] + (end - start))   # 加新算的 token 数
cu_seqlens_k.append(cu_seqlens_k[-1] + end)             # 加要读的 KV 总数
```

```
            cu_seqlens_q    cu_seqlens_k
初始         [0]             [0]
A: +5 / +5   [0, 5]          [0, 5]
B: +2 / +6   [0, 5, 7]       [0, 5, 11]
```

### Q：`cu_seqlens_q` 是 q 张量里的行号

`q` 是这一步真实存在的张量，A、B 首尾相连，共 7 行：

```
q 的行:  0  1  2  3  4 | 5  6
token:  a0 a1 a2 a3 a4 | b4 b5
```

用相邻两项切行：A 是 `q[0:5]`，B 是 `q[5:7]`。`max_seqlen_q = max(5, 2) = 5`。

### K/V：`cu_seqlens_k` 只表示每条读多少个

K/V 不在一个拼好的长张量里，而是分散在分页 cache 的物理块中。所以只用相邻两项的**差**：

```
A: 5 − 0  = 5  → 读 a0..a4
B: 11 − 5 = 6  → 读 b0..b5（b0..b3 是缓存命中的，b4、b5 是这一步刚写的）
```

`max_seqlen_k = max(5, 6) = 6`，也就是 `max(context_lens)`。

去哪里读由 `block_table` 决定：第 i 条序列的第 j 个 KV 在物理块 `block_table[i][j // 4]` 的第 `j % 4` 个位置。

```
block_table = [[3, 8],
               [5, 9]]     # int32，[num_seqs, max_num_blocks]
```

各条序列块数不同时，短的行用 -1 补齐成矩形。例如 A 只有 1 个块时是 `[3, -1]`。FA2 读到 `seqlen_k` 就停，不会读到 -1。

能读到这一步新 token 的 KV，是因为 attention 里先执行了 `kv_cache.write`，再调用 FA2。所以 cache 里 `[0, end)` 的 KV 已经齐了。

### causal：q 比 k 短时右下角对齐

FA2 把 q 当作 k 的**最后** `len(q)` 个位置：

```
B:  k:  b0 b1 b2 b3 b4 b5
    q:              b4 b5      b4 看 b0..b4，b5 看 b0..b5
```

这和 eager 的 `causal_attention` 一致，所以全新 prompt、前缀命中、chunked prefill、decode 都直接用 `causal=True`。

在 5090 上用混合 batch 验证过：新 prompt（300/300）、前缀命中（100/700）、decode（1/400）三种情况下，FA2 和 `causal_attention` 的最大误差分别是 0.0085、0.0012、0.0008，都在 bf16 的正常范围内。

## decode 步：每条序列只有 1 个 query

接着上面的例子，A 算 a5，B 算 b6：

| 序列 | start | end | q 长度 | k 长度 |
| --- | --- | --- | --- | --- |
| A | 5 | 6 | 1 | 6 |
| B | 6 | 7 | 1 | 7 |

```
cu_seqlens_q = [0, 1, 2]     max_seqlen_q = 1
cu_seqlens_k = [0, 6, 13]    max_seqlen_k = 7
block_table  = [[3, 8], [5, 9]]
slot_mapping = [33, 38]
```

decode 只是"每条 q 长度为 1"的特例，所以第 4 步用同一个 `flash_attn_varlen_func` 处理 prefill 和 decode。第 6 步做 CUDA Graph 时，decode 会改用 `flash_attn_with_kvcache`，原因见 [decode-attention-and-cuda-graph.md](decode-attention-and-cuda-graph.md)。

## 常见错误：`cu_seqlens_k` 也加 `end - start`

那样 `cu_seqlens_k` 会和 `cu_seqlens_q` 一样，都是 `[0, 5, 7]`：

- A 不受影响，因为 `start = 0`；
- B 只读 2 个 KV，看不到命中缓存的 b0..b3；
- decode 时每条只读 1 个 KV，也就是只看自己，生成的结果完全错误。

只有全新 prompt 的两者才相等。k 比 q 多出来的部分正好是 `start`，也就是已有 KV 的 token 数。

## 字段速查

| 字段 | 类型 | prefill 例子 | decode 例子 | 谁在用 |
| --- | --- | --- | --- | --- |
| `slot_mapping` | long `[num_tokens]` | `[12 13 14 15 32 36 37]` | `[33, 38]` | `kv_cache.write`（两条路径都用） |
| `cu_seqlens_q` | int32 `[num_seqs + 1]` | `[0, 5, 7]` | `[0, 1, 2]` | 两条路径都用（eager 调 `.tolist()`） |
| `context_lens` | Python list | `[5, 6]` | `[6, 7]` | eager |
| `block_tables` | Python list | `[[3, 8], [5, 9]]` | 同左 | eager |
| `cu_seqlens_k` | int32 `[num_seqs + 1]` | `[0, 5, 11]` | `[0, 6, 13]` | FA2 |
| `max_seqlen_q` | int | 5 | 1 | FA2 |
| `max_seqlen_k` | int | 6 | 7 | FA2 |
| `block_table` | int32 `[num_seqs, max_num_blocks]` | `[[3, 8], [5, 9]]` | 同左 | FA2 |

`max_seqlen_q` 和 `max_seqlen_k` 必须是 Python 整数，FA2 用它们决定启动多少个线程块。

## 为什么 FA2 比 eager 快

| | eager（`layers/attention.py`） | FA2 |
| --- | --- | --- |
| 多条序列 | Python 循环逐条计算 | 一次 kernel，所有序列和 head 并行 |
| 读 K/V | `kv_cache.read` 把块拷贝拼成连续张量 | 按 `block_table` 直接读 cache |
| GQA | `repeat_interleave` 复制 K/V | 映射 head 编号，不复制 |
| 分数矩阵 | 完整的 `[heads, q, k]` 写进显存 | 只在片上存一小块（在线 softmax） |
| 每层 kernel 数 | 每条序列好几个 | 1 个 |

decode 时差距最明显：32 条序列 × 28 层，eager 每层每条都要启动好几个很小的 kernel，GPU 大部分时间在等 CPU。这就是 decode 停在约 275 tok/s 的原因（见 [m6-optimizations.md](../m6-optimizations.md)）。
