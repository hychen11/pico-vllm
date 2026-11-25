# TP 下的线性层：列并行、行并行和 `_PackedLinear`

**结论**：每个 decoder 层里，线性层是"列并行在前、行并行在后"成对出现的。列并行切输出维，不需要通信；行并行切输入维，最后用一次 all_reduce 把各卡的部分和加起来。Qwen3 的列并行层都是由几个投影合并而成的（qkv、gate_up），所以不能直接用 `ColumnParallelLinear`，而要用 `_PackedLinear`：它把**每一段各自**切成 tp 份，而不是把拼起来的整体切成 tp 份。

代码在 `src/pico_vllm/layers/linear.py`，embedding 和 LM head 在 `src/pico_vllm/layers/embed_head.py`。

## 1. 类的继承关系和用在哪里

```
nn.Linear
├─ ColumnParallelLinear          切输出维（基类，Qwen3 里不直接实例化）
│   └─ _PackedLinear             多段拼接的输出维，每段各自切（基类，不直接实例化）
│       ├─ QKVLinear             → Qwen3Attention.qkv_proj    段：q / k / v
│       └─ MergedLinear          → Qwen3MLP.gate_up_proj      段：0(gate) / 1(up)
└─ RowParallelLinear             切输入维，forward 末尾 all_reduce
                                 → Qwen3Attention.o_proj、Qwen3MLP.down_proj

nn.Embedding
└─ VocabParallelEmbedding        按词表切，查表后 all_reduce → Qwen3Model.embed_tokens
    └─ ParallelLMHead            按词表切，矩阵乘后 gather 到 rank 0 → Qwen3ForCausalLM.lm_head
```

`_PackedLinear` 前面的下划线表示它是内部基类：`models/qwen3.py` 只 import `QKVLinear` 和 `MergedLinear`，看不到 `_PackedLinear` 的名字，但构造 `QKVLinear` 时会经过它：

```
QKVLinear(hidden, head_dim, num_heads, num_kv_heads, bias)
  └─ super().__init__(hidden, {"q": 2048, "k": 1024, "v": 1024}, bias)   → _PackedLinear.__init__
       ├─ super().__init__(hidden, 4096, bias)                             → ColumnParallelLinear.__init__
       │    ├─ nn.Linear(hidden, 4096 / tp)                                  本卡权重
       │    └─ param.weight_loader = self.weight_loader                      见下面的说明
       └─ 计算 shard_offsets
```

`ColumnParallelLinear.__init__` 里的 `self.weight_loader` 按 Python 的方法查找规则取的是**子类**的版本，所以 `QKVLinear` 的参数上挂的是 `_PackedLinear.weight_loader`（带 `shard_id` 的那个）。

## 2. 一层里的通信

| 模块 | 列并行（切输出维） | 行并行（切输入维） | 通信 |
|---|---|---|---|
| Attention | `qkv_proj`（QKVLinear） | `o_proj` | 1 次 all_reduce |
| MLP | `gate_up_proj`（MergedLinear） | `down_proj` | 1 次 all_reduce |

列并行的输出是本卡那 1/tp 的 head（或 1/tp 的 MLP 中间维），正好是行并行需要的那 1/tp 输入，所以两层之间不用通信。整个模型的通信是：embedding 1 次 all_reduce + 每层 2 次 all_reduce + LM head 1 次 gather。

## 3. 为什么不能直接用 `ColumnParallelLinear`

### qkv_proj 存的是什么

把 q、k、v 三个投影合成一个矩阵，一次矩阵乘就能算出三者，比三次快。权重按行拼接。用一个小例子：4 个 q head、2 个 kv head，每个 head 画成一行：

```
qkv_proj.weight（完整）：
  q0
  q1
  q2
  q3
  k0
  k1
  v0
  v1
```

GQA 的分组：q0、q1 共用 k0/v0；q2、q3 共用 k1/v1。

### 方案 A：整体切（`ColumnParallelLinear` 的做法，错误）

把 8 行当成一个整体按顺序对半切：

```
rank 0:  q0 q1 q2 q3        ← 全是 q
rank 1:  k0 k1 v0 v1        ← 全是 k 和 v
```

但每张卡上的 attention 都认为自己拿到的是"2 个 q head + 1 个 k head + 1 个 v head"，会按 `[2, 1, 1]` 拆开：

```
rank 0 的理解：  q = q0 q1    k = q2 ❌    v = q3 ❌
rank 1 的理解：  q = k0 k1 ❌  k = v0 ❌    v = v1 ❌
```

### 方案 B：每段各自切（`_PackedLinear` 的做法，正确）

```
q 段 [q0 q1 q2 q3] 对半切 → rank0: q0 q1    rank1: q2 q3
k 段 [k0 k1]       对半切 → rank0: k0       rank1: k1
v 段 [v0 v1]       对半切 → rank0: v0       rank1: v1

rank 0:  q0 q1 | k0 | v0     ✅ q0、q1 正好对应 k0/v0
rank 1:  q2 q3 | k1 | v1     ✅
```

每张卡拿到的是一组完整的 attention head，可以独立算 attention，不需要通信。

`gate_up_proj` 是同样的道理：`SiluAndMul` 把输出的后一维对半分成 gate 和 up，再逐元素相乘，要求本卡上的 gate 和 up 是同一批中间维。整体切的话 rank 0 拿到全部 gate、rank 1 拿到全部 up，也是错的。

## 4. 代码怎么实现

### `__init__`：记录每段在本卡权重里的位置

Qwen3-0.6B、TP=2：16 个 q head、8 个 kv head、head_dim=128，所以 q 有 2048 行，k、v 各 1024 行，共 4096 行。

```python
shard_offsets = {
    "q": (0,    1024),   # 本卡 q 段：从第 0 行开始，共 1024 行（= 2048 / 2）
    "k": (1024,  512),
    "v": (1536,  512),
}
# 本卡权重共 2048 行，是 4096 行的一半
```

### `weight_loader`：多一个 `shard_id` 参数

checkpoint 里 q、k、v **本来就是三个独立的张量**（`q_proj.weight`、`k_proj.weight`、`v_proj.weight`），不存在拼好的 qkv。`utils/loader.py` 遇到 `q_proj` 时，按 `Qwen3ForCausalLM.packed_modules_mapping`（`"q_proj": ("qkv_proj", "q")`）找到要写入的参数，然后调用：

```python
param.weight_loader(param, q_proj_weight, "q")
```

`weight_loader` 做三件事：

```python
offset, size = self.shard_offsets["q"]                    # (0, 1024)
loaded = q_proj_weight.chunk(tp, dim=0)[rank]             # 完整 q 里本卡的那一份
param.data.narrow(0, offset, size).copy_(loaded)          # 写到本卡权重的对应位置
```

k、v 同理，各写各的位置。`ColumnParallelLinear.weight_loader` 只接收一个完整张量，表达不了"这是 q 段，放在哪里"。

### 对比

| | ColumnParallelLinear | _PackedLinear |
|---|---|---|
| 怎么看待输出维 | 一个整体 | 多段拼接（q/k/v，或 gate/up） |
| 怎么切 | 整体切 tp 份 | 每段各自切 tp 份 |
| 加载权重 | 一个完整张量 | 每段一个张量，用 `shard_id` 区分 |
| 适用 | 单个投影 | 合并的投影（`qkv_proj`、`gate_up_proj`） |

TP=1 时每段切 1 份等于不切，就是 M7 之前那个普通的合并 linear。

## 5. 行并行：`RowParallelLinear`

`o_proj` 的输入是本卡 attention 输出的那 1/tp 个 head，`down_proj` 的输入是本卡那 1/tp 的 MLP 中间维。所以行并行的权重按**输入维**（列）切：本卡持有 `weight[:, rank 那 1/tp]`，算出的是完整形状 `[N, hidden]` 的部分和，`all_reduce` 把各卡的部分和加起来，每张卡都得到完整结果。bias 不切，只在 rank 0 上加，这样求和后 bias 只算一次。

## 6. embedding 和 LM head

两者的权重形状都是 `[vocab, hidden]`，都按词表（行）切：rank r 持有第 `[r × V/tp, (r+1) × V/tp)` 行。Qwen3-0.6B 的 `tie_word_embeddings=True`，两者共用一个参数，切法相同，所以每张卡上共享的是同一块切片。

- **`VocabParallelEmbedding`**：每个 token id 只在一张卡上。其他卡把 id 换成 0 去查表，再把结果置 0。`all_reduce` 求和后每张卡都得到正确的那一行。
- **`ParallelLMHead`**：每张卡用自己那块权重算出 `[N, V/tp]` 的 logits，`gather` 到 rank 0 后在最后一维拼接成 `[N, V]`，其他卡返回 `None`。从数学上看它就是一个列并行层（切输出维 = 词表），但为了和 embedding 共享权重，它继承的是 `VocabParallelEmbedding`。用 `gather` 而不用 `all_gather`，是因为只有 rank 0 做采样。

LM head 有 151936 × 1024 ≈ 1.55 亿参数，约占 Qwen3-0.6B 的四分之一，所以值得切。

调用链：

```
ModelRunner.run()                         worker/model_runner.py
  └─ model.compute_logits(hidden[last])   只传每个序列最后一个 token 的 hidden
       └─ self.lm_head(hidden_states)     ParallelLMHead
            → rank 0: [num_seqs, vocab]，交给 sampler
            → 其他 rank: None
```
