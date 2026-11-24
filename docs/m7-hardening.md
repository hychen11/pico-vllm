# M7 测试覆盖与正确性加固

M7 的目标（见 [ROADMAP.md](../ROADMAP.md#m7--test-coverage-and-correctness-hardening)）：在 M8（张量并行）改动模型层和 runner 之前，把测试网补密，让之后的改动有保护。这一步加了随机化的状态机测试、压力下的引擎一致性测试、abort 功能及其测试，修了测试发现的 4 个 bug。

测试数：212 → 318（`PICO_VLLM_QWEN_MODEL` 已设置、5090 可用时全部通过）。

## 修复的 bug

每个都有对应的回归测试；把修复撤掉，相应测试会失败（已逐个验证）。

| # | 问题 | 现象 | 修复 | 回归测试 |
| --- | --- | --- | --- | --- |
| 1 | `block_size = 1` 时 decode 从不分配新 block | `can_append` / `may_append` 用 `len(seq) % block_size == 1` 判断，block_size 为 1 时永远不成立；`block_table` 比 token 数少，下一步 `slot_mapping` 越界 | 改为"`block_table` 比 `num_blocks` 少一块时分配"，与 block_size 无关 | `test_block_size_one_allocates_a_block_every_decode_step`，随机测试的 `block_size=1` 用例 |
| 2 | 需要的 KV 超过整个 pool 的请求会被接受 | prompt 本身放不下：调度时 `AssertionError`；prompt 放得下但 decode 时超出：单独运行时抢占自己，然后 `AssertionError` | `add_request` 检查 `ceil((prompt + max_tokens - 1) / block_size) <= num_blocks`，不满足就 `ValueError`。最后一个采样出的 token 不再经过模型，所以是 `- 1` | `test_request_larger_than_kv_pool_is_rejected_not_deadlocked`，`test_request_that_exactly_fills_the_kv_pool_runs` |
| 3 | 极小的 temperature 让采样报错 | `temperature = 1e-45` 时 `logits / t` 溢出成 inf，softmax 里 inf − inf = nan，`multinomial` 报错 | 先减去每行最大值再除以 t：最大值变成 0，其余变成 −inf，softmax 有定义；完全并列的最大值仍然平分概率 | `test_extreme_temperature_does_not_produce_nan`，`test_extreme_temperature_splits_exact_ties` |
| 4 | 引擎空闲时调用 `step()` 报错 | 没有请求时 `schedule()` 走到 decode 分支的 `assert scheduled_seqs` | `step()` 在 `is_finished()` 时直接返回 `[]` | `test_step_on_finished_engine_returns_nothing` |

另外 `add_request` 现在拒绝空 prompt（以前 `Sequence([])` 会在取 `token_ids[-1]` 时抛 `IndexError`）。

## 新功能：abort

- `Scheduler.abort(seq)`：不管 seq 在 waiting（包括分块 prefill 做到一半、被抢占后排队）还是 running，都把它移出队列、释放 block、标记 FINISHED。已经结束的 seq 什么也不做。
- `LLMEngine.abort_request(seq)`：对外接口，seq 是 `add_request` 的返回值。被 abort 的 seq 不会出现在 `step()` 的返回值里。
- 它已经算完的满 block 保留 hash，之后相同前缀的请求仍能命中（有测试验证命中后输出依然正确）。M10 的 HTTP 服务在客户端断开时会用到它。

## 不变量

`tests/conftest.py` 的 `check_scheduler_invariants` 在每一步之后检查，通过 `check_invariants` fixture 使用：

- **队列**：waiting / running 内部无重复、互不相交；WAITING 状态的 seq 在 waiting 里，RUNNING 在 running 里；FINISHED 的 seq 不在任何队列，也不持有 block；步与步之间 `num_scheduled_tokens == 0`。
- **block 池**：free 列表无重复，free 和 used 互不相交且合起来是全部 block；每个 block 的 `ref_cnt` 等于持有它的 block_table 个数；同一个 block_table 里没有重复 block。
- **前缀缓存**：seq 持有的每个已 hash 的 block，缓存的 token 正好等于 seq 对应位置的 token（前缀命中安全的前提）；`hash_to_block_id[h]` 指向的 block 的 hash 就是 h。
- **进度**：running 的 seq 除最新采样的 token 外都有 KV（`num_cached_tokens == num_tokens - 1`），block_table 正好覆盖这些位置；waiting 里只有队首可能持有 block（分块 prefill 进行中），其余都没有 block、进度为 0。

## 新增测试

### `tests/unit/test_scheduler_random.py`（CPU，约 0.4 s）

- `test_random_operations_keep_invariants_and_finish`：7 种配置 × 8 个种子。一个假引擎随机地加请求（60% 和之前的请求共享前缀）、abort、执行一步（假模型随机采样，10% 概率出 EOS，30% 的请求 `ignore_eos`），每次操作后检查不变量和 prompt 不被改动。最后不再加请求，所有请求必须在有限步内结束：没被 abort 的生成了 1 到 `max_tokens` 个 token，少于 `max_tokens` 时最后一个必须是 EOS；所有 block 归还。配置覆盖宽松、显存紧张（抢占）、token 预算紧张（分块）、全都紧张、`block_size` 为 1、2、8。
- `test_random_cases_exercise_the_hard_paths`：防止随机测试"空转"。所有种子合计：抢占 238 次、分块 prefill 936 次、前缀命中 1468 次、abort 365 次，测试要求每项至少 20 次。
- `test_block_manager_random_alloc_free_append`：20 个种子，直接对 `BlockManager` 随机做 allocate、decode append、hash、free，每步检查 ref_cnt、free/used 和"已 hash block 的内容等于持有者的 token"。

### `tests/integration/test_engine_hardening.py`（CPU，小的随机 Qwen3，约 2 s）

参考答案都是 `greedy_reference`：每步重算整条序列、不用 KV cache。引擎内部不管做了什么，每条 greedy 请求都必须等于它。`Recorder` 统计实际发生的抢占、分块和前缀命中次数，断言测试确实走到了要测的路径。

- **压力下的一致性**：6 条共享前缀的请求，在宽松、显存紧张、预算紧张、全都紧张 4 种配置下逐条等于参考答案，验证 batching、抢占重算、分块、前缀命中都不会让序列互相串；第二轮请求命中第一轮释放但仍保留 hash 的 block。
- **abort**：在 waiting 中、decode 中、分块 prefill 做到一半、被抢占后、和别的请求共享前缀时 abort；对已结束的请求 abort 是空操作。每种情况都检查不变量、block 归还，以及其他请求的输出不受影响。
- **请求校验**：空 prompt；超出 KV pool；正好填满 pool 能正常运行；`step()` 在空闲时返回 `[]`。
- **长 prompt**：长度正好是整数个 block（最后一个满 block 不能命中缓存）；是 token 预算的十几倍；正好到 `max_model_len`；只有 1 个 token。
- **采样边界**：`max_tokens = 1` 在 prefill 那一步就结束；第一个 token 就是 EOS；同一 batch 里一条遇 EOS 停、一条 `ignore_eos`；greedy 和 temperature 1.0 / 1e-30 / 50 混在一个 batch 里，greedy 行不受影响。

### 补到已有文件的

- `tests/unit/test_block_manager.py`：`test_can_append_with_no_free_blocks` 原来用一个 block_table 为空的 seq 测 decode，这种状态实际不会出现，现在先 prefill 再 append；加上 block_size = 1 的回归测试。
- `tests/unit/test_sampler.py`：temperature 1e-30 / 1e-45 和完全并列的情况。
- `tests/cuda/test_cuda_graph.py`：补上 M6 时缺的两个场景：decode 过程中跨 block 边界（`block_table` 多出一列），以及 decode 进行中有新序列加入。

## 没有做的

- mypy：`uv run mypy src` 在 M7 之前就有 34 个错误，M7 没有新增，也没有修。
- 随机测试用固定种子的循环，没有引入 `hypothesis`：不加依赖，失败时直接用种子复现。
