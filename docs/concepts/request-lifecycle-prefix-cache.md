# 一个请求的完整流程：从进入、命中前缀缓存到 attention

**结论**：前缀缓存命中发生在**调度阶段**，只是"记账"：调度器发现新请求开头的几个块和已有的块内容相同，就把这些块号直接放进它的 `block_table`，并把 `num_cached_tokens` 设成命中的 token 数。模型这一步只计算剩下的 token。到了 attention，序列按自己的 `block_table` 读取全部 K/V，其中命中的块是**别的请求之前写进去的**，没命中的部分是**这一步刚写进去的**。

下面的数值全部来自对本项目代码的实际运行（`Scheduler`、`BlockManager`、`ModelRunner.prepare_inputs`），为了好看，`block_size = 4`，KV 池共 8 个块。

## 两个请求

| 请求 | prompt | 说明 |
| --- | --- | --- |
| A | `[11 12 13 14 15 16 17 18 19 20]`，10 个 token | 先到 |
| B | `[11 12 13 14 15 16 17 18 31 32 33]`，11 个 token | 后到，前 8 个 token 和 A 相同 |

## 第 1 步：A 进来，prefill

### ① `add_request`：进入 waiting 队列

`Sequence` 刚创建时：`block_table = []`，`num_cached_tokens = 0`，状态 `WAITING`。

### ② `Scheduler.schedule` → `BlockManager.can_allocate`：查前缀缓存

A 的 10 个 token 分成 3 个逻辑块：`[11 12 13 14]`、`[15 16 17 18]`、`[19 20]`。

`can_allocate` 从第 0 块开始，逐块计算哈希（每块的哈希包含前一块的哈希，所以相同的哈希意味着**从开头到这一块的内容都相同**），然后去 `hash_to_block_id` 里查。A 是第一个请求，表是空的，第 0 块就没命中，结果：命中 0 块。

注意：**最后一块永远不参与查找**。一是它可能没写满（没写满的块没有哈希）；二是即使写满了，也至少要留 1 个 token 给模型算，因为采样下一个 token 需要最后一个位置的 logits，而 KV cache 里只存 K/V，不存 logits。

### ③ `BlockManager.allocate`：分配块

没命中，3 个块都从空闲队列里取：`block_table = [0, 1, 2]`，`num_cached_tokens = 0`。

调度器设置 `num_scheduled_tokens = 10`：这一步要算全部 10 个 token。

### ④ `ModelRunner.prepare_inputs`：只取要算的 token

| 字段 | 值 | 含义 |
| --- | --- | --- |
| `input_ids` | `[11 … 20]` | 第 `num_cached_tokens` 到 `num_cached_tokens + num_scheduled_tokens` 个 token |
| `positions` | `[0 … 9]` | 每个 token 在序列里的位置，用于 RoPE |
| `slot_mapping` | `[0 … 9]` | 每个 token 的 K/V 写到 KV 池的哪个槽位：`block_table[pos // 4] * 4 + pos % 4` |
| `cu_seqlens_q` | `[0, 10]` | 本步 query 共 10 个 |
| `context_lens` | `[10]` | attention 要读 10 个 token 的 K/V |

### ⑤ attention（每一层）：先写，再读

1. `kv_cache.write`：按 `slot_mapping` 把 10 个 token 的 K/V 写进槽位 0–9，也就是块 0 全部、块 1 全部、块 2 的前 2 个位置。
2. `kv_cache.read`：按 `block_table = [0, 1, 2]` 读出前 10 个 token 的 K/V，做 causal attention。

### ⑥ `Scheduler.postprocess`：给写满的块算哈希

- `hash_blocks`：这一步写满了块 0 和块 1，给它们算哈希，登记到 `hash_to_block_id`。块 2 只有 2 个 token，**不算哈希**。
- `num_cached_tokens` 变成 10；追加采样出的 token 99。

此时块的状态：

```
块 0: [11 12 13 14]  ref=1  有哈希
块 1: [15 16 17 18]  ref=1  有哈希
块 2: [19 20 __ __]  ref=1  无哈希（没写满）
```

## 第 2 步：A decode

`num_scheduled_tokens = 1`，只算新 token 99：

| 字段 | 值 |
| --- | --- |
| `input_ids` | `[99]` |
| `positions` | `[10]` |
| `slot_mapping` | `[10]`（块 2 的第 2 号位置） |
| `context_lens` | `[11]` |

attention 先把 99 的 K/V 写进槽位 10，再按 `[0, 1, 2]` 读前 11 个 token。块 2 现在有 3 个 token，仍然没写满，所以仍然没有哈希。

## 第 3 步：B 进来，命中前缀缓存

### ② `can_allocate`

B 的逻辑块：`[11 12 13 14]`、`[15 16 17 18]`、`[31 32 33]`。

- 第 0 块：哈希和 A 的块 0 相同，查到块 0，再比对 token 内容确实一样 → **命中**；
- 第 1 块：哈希（包含第 0 块的哈希）和 A 的块 1 相同 → **命中**；
- 第 2 块是最后一块，不查。

结果：命中 2 块。

### ③ `allocate`

- 命中的块 0、块 1 **直接放进 B 的 block_table**，引用计数 +1（变成 2，A 和 B 共用）；
- 剩下的 1 块从空闲队列取，得到块 3；
- `block_table = [0, 1, 3]`，`num_cached_tokens = 2 × 4 = 8`。

调度器设置 `num_scheduled_tokens = 11 − 8 = 3`：**这一步只算 31、32、33**。

### ④ `prepare_inputs`

| 字段 | 值 | 含义 |
| --- | --- | --- |
| `input_ids` | `[31, 32, 33]` | 只有没命中的 3 个 token |
| `positions` | `[8, 9, 10]` | 从第 8 个位置接着往后，不是从 0 开始 |
| `slot_mapping` | `[12, 13, 14]` | 写进块 3 的第 0–2 号位置 |
| `cu_seqlens_q` | `[0, 3]` | 本步 query 只有 3 个 |
| `context_lens` | `[11]` | 但 attention 要读全部 11 个 token |

q 只有 3 个、KV 要读 11 个：FA2 用的 `cu_seqlens_k`、`block_table` 等字段就是按这个规则算的，见 [attention-context.md](attention-context.md)。

### ⑤ attention：命中的块就在这里被用上

1. `kv_cache.write`：只写 3 个 token（31、32、33）的 K/V，写到槽位 12–14。
2. `kv_cache.read`：按 `block_table = [0, 1, 3]` 读 11 个 token：

```
块 0: [11 12 13 14]   ← 第 1 步时 A 写进去的，B 没算过，直接读
块 1: [15 16 17 18]   ← 同上
块 3: [31 32 33 __]   ← 这一步 B 刚写进去的
```

3. B 的 3 个 query（位置 8、9、10）和这 11 个 K/V 做 causal attention：位置 8 看前 9 个，位置 9 看前 10 个，位置 10 看全部 11 个。

这就是前缀缓存省下的计算：token 11–18 的 K/V 不用再算一遍，B 这一步只跑了 3 个 token 的模型计算，而不是 11 个。

### ⑥ `postprocess`

B 的块 3 只有 3 个 token，没写满，不算哈希。块的状态：

```
块 0: [11 12 13 14]  ref=2  有哈希   ← A 和 B 共用
块 1: [15 16 17 18]  ref=2  有哈希   ← A 和 B 共用
块 2: [19 20 99 __]  ref=1  无哈希   ← A 自己的
块 3: [31 32 33 __]  ref=1  无哈希   ← B 自己的
```

## 常见疑问

**半满的块能被命中吗？** 不能。只有写满并且算了哈希的块才会登记，才可能被别的请求命中。块 2、块 3 要等写满 4 个 token 后，在那一步的 `postprocess` 里才会算哈希。

**那 attention 为什么能读半满的块？** 因为那是序列**自己**的块，里面的 K/V 是它自己前几步写进去的，attention 必须看到它们。kernel 按 `context_lens` 读到对应位置就停，后面还没用的位置不会读。

| | 前缀缓存命中 | attention 读 KV |
| --- | --- | --- |
| 解决什么问题 | 新请求能不能复用别的请求已经算好的块 | 一条序列读取自己前面所有 token 的 K/V |
| 能用半满的块吗 | 不能，只有写满并算了哈希的块 | 能，读到 `context_lens` 为止 |

**共用的块会被改写吗？** 不会。只有写满的块才会被共享，而写满的块不会再被写入：新 token 总是写到序列最后那个块，或者新分配的块里。所以 A 和 B 共用块 0、块 1 是安全的。

**A 结束后，块 0、块 1 会被释放吗？** `deallocate` 只把引用计数减 1。B 还在用，引用计数从 2 变成 1，块不会被释放。等 B 也结束、计数降到 0，块才回到空闲队列。块回到空闲队列后，哈希仍然保留，在被重新分配出去之前，后来的请求还能命中它。

**如果 B 的 prompt 和 A 完全一样（10 个 token）呢？** 块 0、块 1 命中；最后一块 `[19 20]` 不参与查找（而且 A 的块 2 本来也没有哈希），所以 B 会新分配一个块，重新计算 19、20 这两个 token。
