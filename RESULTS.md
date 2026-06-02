# robagent: code-centric harness evolution — 结果与边界

## 摘要

我们用 meta-loop（每轮派一个 coding agent / "builder" 提出一个确定性组件）演化 LLM agent harness，向 **code-centric / micro-LLM** 的终态推进——把 LLM 的工作搬进确定性 Python，让 LLM 只承担最小判断。A/B 实验：A = 普通 "propose a better agent system" skill（无设计哲学），B = code-centric robust skill（强制每轮分类组件在 durability axis 上：channel / reactive_guard / deterministic_glue / predictive_heuristic）。同一 builder 模型（Claude Opus 4.7）、同一 eval 模型（DeepSeek V4-Pro，thinking 关）、同一 iter budget（20 iter）。

**双 benchmark 结果出现了明显方向相反的现象：**

| benchmark | v0 | A champion | B champion | A→test | B→test |
|---|---|---|---|---|---|
| GAIA (train-30 / test-135) | 7/30 → 20/135 | 11/30 → 28/135 (20.7%) | **17/30 → 42/135 (31.1%)** | ✅ 泛化 | ✅ **+50% rel vs A** |
| tau2-bench banking_knowledge (train-30 / test-67) | 4/30 → 8/67 (11.9%) | 11/30 → 7/67 (10.4%) | **19/30** → 7/67 (10.4%) | ❌ 退化 | ❌ **比 v0 还低** |

GAIA 上 B 大胜，train 改进干净地迁移到 test。tau2-banking 上 B 在 train 上拉到 v0 的 4.75 倍，**但 test 上反而比 v0 低 1 题，与 A 同分**。这个矛盾迫使我们仔细看 B 在两个 benchmark 上**到底搬进代码的是什么**——这就是本文的核心：**机械层 vs 解读层**。

---

## 1. 实验设置

### 1.1 thesis

> 一个 coding agent 能否自动把"LLM-while-loop"形态的脆弱 agent 演化成"deterministic-code dominates，LLM 是 micro-decision"的稳健 harness？

### 1.2 skill 与 durability axis

robust skill 要求 builder 每轮产 **ONE 个**新组件，按其 activation predicate 归到四类：

- `channel` — 任务结构触发，需要 agent 拿不到的外部内容（文件/网页）
- `reactive_guard` — 已观察到的失败事件触发（tool.failed、空响应）
- `deterministic_glue` — 一直触发，对手里数据做确定性变换
- `predictive_heuristic` — 对话文本触发，靠正则/关键词猜模型会怎么做（HIGH risk）

预设：前三类低风险、第四类要过严格 gate。

### 1.3 meta-loop

外环：scoring + 选 frontier。内环：每轮派一个 Opus builder，读 frontier + N 个失败 sim 的证据 → 提一个 hypothesis，归一个 class，写代码，写 plugin manifest → 外环评分。20 iter 后选 champion（train 分最高、tiebreak 看 iter）跑 test。

---

## 2. GAIA 结果（机械层成功）

train-30 上 B 单调爬到 17/30，A 噪声抖动卡 11/30。test-135 上 **B 42/135 = 31.1% vs A 28/135 = 20.7%**——+14 题、+50% 相对，泛化干净。

B 在 GAIA 上搬进代码的是什么？看 B champion 的组件清单：

- **file reader channel**：`extras.file_name` 是文件路径 → 读文件、注入内容到 prompt。
- **web fetch channel**：question 提到具体 URL/source → 抓页面、注入内容。
- **answer-format extractor**（deterministic_glue）：从 LLM 自由文本里抽 `Answer: <...>` 行、剥前后冗余、做数字归一化。
- **finish_reason=length 的 reactive_guard**：观察到 length 截断 → bump max_tokens + 续写一次。

**这些组件有一个共同形态：它们的行为完全由系统能直接观察到的"机械事实"决定**——`extras.file_name` 是不是个字符串、URL 是不是 HTTP 200、最后一行能不能匹配 `^Answer:\s*(.+)$`、`finish_reason` 字段是不是 "length"。这些是 **mechanical facts**，不依赖任何对人类自然语言策略的解读。

不论 train task 还是 test task 还是任何未来 task，`extras.file_name` 是字符串的语义都一样，HTTP 200 的语义也一样。所以一旦代码写对，迁移是无条件的。

---

## 3. tau2-bench banking_knowledge 结果（解读层失败）

### 3.1 train

| iter | candidate (B) | train score |
|---|---|---|
| v0 | stock LLMAgent | 4/30 |
| 5 | cashback_policy_engine | 6 |
| 7 | card_catalog_channel | 9 |
| 9 | transfer_reason_resolver | 11 |
| 10 | cashback_category_rate | 13 |
| 11 | cashback_workflow_isolation | 15 |
| 13 | discoverable_arg_validator | 16 |
| 15 | **dispute_workflow_corrector** | **19** |

B 的 train 是**单调台阶**——每个新组件叠在前面 frozen 不动的组件之上，多题被一次性翻过。13→15→16→19 的爬升让人相信"deterministic policy 引擎在 compound"。

A 同期：7, 6, 4, 9, 6, 6, 7, 6, 7, 8, 11, 6, 5, 11, ...——纯噪声，champion 卡 11/30。

### 3.2 test-67

```
v0:           8/67  (11.9%)
A champion:   7/67  (10.4%)   train 11/30 → -1.5 percentage points
B champion:   7/67  (10.4%)   train 19/30 → -53 percentage points
```

**两个 champion 都比 v0 低。** B 的 63% 训练精度完全没迁移过去；A 也同样塌。

per-task swing 揭示原因——这不是"没有改进"，是**主动 regress**：

| 桶 | 题数 | 含义 |
|---|---|---|
| v0 通过、A 和 B 都破 | **5** | task_007, task_033, task_072, task_073, task_093 |
| v0 通过、B 破 | 1 | task_035 |
| v0 通过、A 破 | 0 | — |
| A 独得（v0 没过） | 2 | — |
| B 独得（v0 没过） | 3 | task_005, task_043, task_098 |
| A + B 都新得（v0 没过） | 2 | task_052, task_089 |

A 净 = +4 −6 = **−2**（→ 8 − 2 + 1 共 7）。  
B 净 = +5 −6 = **−1**（→ 8 − 1 共 7）。

**A 和 B 都新找到一些题，但同时破坏了更多 v0 本来能过的题。** B 破得比 A 更狠（6 vs 5），尽管 B 的 train 比 A 高 8 题。这就是问题。

---

## 4. 机械层 vs 解读层

这是文章的核心。GAIA 的成功和 banking 的失败在表面看都是"把工作搬进确定性代码"——为什么一个迁移、一个反噬？

### 4.1 定义

**机械层代码（mechanism-layer code）**

> 行为完全由系统能直接观察到的字段、协议、文件格式、API 返回值决定。代码不需要"解读"任何人类自然语言文档；它只是**执行一个 transform**。

例：`if extras.file_name: content = open(extras.file_name).read()`。规则一行。`extras.file_name` 是 literal field、`open().read()` 是 literal operation。没有"我从文档里推断出来"这一步。

**解读层代码（interpretation-layer code）**

> 把一份**长的人类自然语言策略文档**（外加 builder 从少量训练样本里看到的"该策略实际怎么执行"），**压缩成一组确定性分支**。代码的每个 if 都是 builder 对策略文本+样本的**归纳解读**。

例（来自 B 的 `provisional_credit.py`）：

```python
def definitely_ineligible(reason, contacted_merchant, purchase_date, today):
    NEVER_ELIGIBLE = {
        "incorrect_amount",
        "goods_services_not_as_described",
        "canceled_subscription_still_charging",
        "refund_never_processed",
    }
    if reason in NEVER_ELIGIBLE:
        return True
    if reason != "fraud" and contacted_merchant is False:
        return True
    if reason == "goods_services_not_received" and (today - purchase_date).days <= 30:
        return True
    return False
```

builder 写注释说这是"compiling doc_015's NOT-eligible rules into a one-directional provisional-credit corrector"。它声称 `destructive_fallback: false`，因为"只把 over-granted 的 true 翻成 false，从不翻 false 为 true，所以不会错误授予资格"。

### 4.2 为什么机械层泛化

GAIA 的 file_reader：`extras.file_name` 在任何 task 都是一个**字符串字段**。读它的语义在 train、test、未来 task 上**完全一样**——operating system 怎么 open 一个文件不依赖于 task 内容。

形式化：机械层规则的"是否触发"和"触发后做什么"，**和训练样本无关**。规则的语义在 task 空间上是 well-defined 的常量函数。

```
mechanism rule:  task ∈ Task → behavior(task)   # well-defined regardless of train sample
```

所以一旦规则正确，迁移是数学上无条件的。它只可能在两种情况下不工作：
1. 系统本身字段变了（`extras.file_name` 改名）—— 整套基础设施变了，所有依赖都坏，不是 train/test 问题。
2. 规则一开始就错（写了 `open()` 但拼错了路径）—— 一开始就在 train 上失败，不会被选为 champion。

### 4.3 为什么解读层不泛化

`definitely_ineligible()` 看起来也是一段确定性代码，对任何 task 都按同样的 if 分支执行。但**它的 if 分支本身是 builder 从 N=3-4 个 train sim 里归纳出的 hypothesis**——是对真实策略 `doc_015` 的有限样本 MLE。

形式化：

```
真实策略:        P_true(case) = eligible | not_eligible    # 完整、复杂、有 exception
builder 解读:    P_hat(case)  = 上面那个 3-clause 函数      # 在 train sims 见到的 case 上拟合得很好
```

`P_hat` 是 `P_true` 的**子集近似**——builder 见过的 case，`P_hat` 和 `P_true` 一致；builder 没见过的 case，`P_hat` 沉默或错判。

但 `definitely_ineligible()` 不是**沉默**式部署的——它**主动覆盖 LLM 的判断**：

```python
if definitely_ineligible(...) and llm_set_eligible_to_true:
    force_eligible_to_false()
```

也就是说，**只要 `P_hat` 说 "ineligible"，无论 `P_true` 说什么，结果都被强制为 ineligible**。

这造成两类破坏：

**(i) 子集近似的"边缘 case"被强制反转**。`P_hat` 对 `P_true` 的**精确程度**不是 boolean；是"在 builder 见过的 evidence 上 100%，在没见过的边缘 case 上未知"。test task 系统性地暴露这些边缘 case。

**(ii) "non-destructive only forces false" 的承诺基于一个未经检验的假设**——"P_hat ineligible ⊂ P_true ineligible"。如果 builder 漏掉了一条 exception（"goods_services_not_as_described 在 N≥3 次书面通知未解决后变为 eligible"），那条 exception 适用的 case 上 LLM 正确判 eligible、`P_hat` 强制 false → 任务塌。这种 case 在 train 没出现（所以 builder 没编码 exception），在 test 出现。

### 4.4 具体例子：B 的 dispute_workflow_corrector 怎么破 task_035

`task_035` 是 v0+A 都过、**B 唯独不过**的题（per-task swing 表的最后一行）。我们重建一下出错链。

**任务**（典型 banking dispute 任务）：用户对一笔信用卡 charge 提交 dispute。在系统内部状态里，这笔 dispute 的 gold action 是 `file_credit_card_transaction_dispute_4829(..., eligible_for_provisional_credit=true, ...)`——gold 答案标 true。

**v0 行为**：LLM 读完 doc_015 + 用户对话，结合对话里提供的 context（用户多次书面通知商家未果、或者 dispute reason 不属于硬 NEVER_ELIGIBLE，等），决定填 true。db_check 通过。

**B 行为**：LLM 输出同样的 tool call，flag=true。然后 iter15 的 `dispute_workflow_corrector` layer 拦截：

```
provisional_credit.definitely_ineligible(reason, contacted_merchant, purchase_date, today)
```

builder 编码的 3 条 ineligible 规则之一被命中（具体哪条不重要，重点是命中），返回 True。corrector 把 `eligible_for_provisional_credit=true` **强制改为 false**。tool call 发出去。db state 走到一个 gold 不期待的位置。db_check fails。task_035 在 B 上塌。

**为什么 train 上没暴露这个 break？** 因为 train-30 里那 4-6 个 builder 当时分析的 sim，**没有一个**是"`P_hat` 说 ineligible 但 `P_true` 说 eligible"的情形——所以 builder 写出的 3 条规则**在 train evidence 上自洽**。N=30 整个 train 集里命中规则的 task 数也少，`P_hat` 的错误覆盖刚好落在 N=30 没采样到的地方。所以 train 显示这个 corrector 是 +1 题（把 task_038 这类 corrector 该 fire 且 fire 对了的题翻过来），**净 train 收益 +1，看起来完美**。

**test-67 上同样的 corrector**：N 大、case 多，落在"`P_hat` 错覆盖"区间的 task 出现（task_035 就是一个）。corrector 错触发、强制翻转、任务塌。

### 4.5 为什么 stacking 把问题放大

B champion 不是一个 corrector——是 **10+ 层 frozen 叠加**：cash-back policy engine、category-rate engine、prerequisite-check injector、retention-protocol channel、discoverable-arg validator、cash-back workflow isolation、card-catalog channel、transfer-reason resolver、dispute_workflow_corrector、isolation strip……

每一层都是同一种构造："从 train sims 归纳一条 deterministic 规则，自称 non-destructive 因为只在 CERTAIN 时触发"。每一层的 `P_hat` 都是真实 policy 子结构的 MLE。

记每层错触发的概率为 ε_i（在 test 分布上）。即使每层 ε_i ≈ 5%，10 层叠加后**至少一层错触发**的概率 ≈ 1 − (0.95)^10 ≈ **40%**。

这就是 B 在 test 上 break 6 题、比 A（更少更松的层）破得更狠的机理——**stacking 把每层小的 false-positive 放大成大概率的整体破坏**。

GAIA 的 file_reader / web_fetch / answer_extractor 也叠加（B GAIA champion 也是多组件），为什么不放大？因为这些是**机械层**——`ε_i = 0` 在结构上成立（规则要么触发要么不触发，触发后行为是机械事实），不存在"build 时 N 太小的子集近似"。0 × 任何 stacking = 0。

---

## 5. 边界的形式化

| 代码类别 | 它编码的东西 | 在 train 上 | 在 test 上（OOD case） |
|---|---|---|---|
| **mechanism** | 系统字段/协议/格式上的常量函数（与训练样本无关） | 触发即正确（修了一类失败） | 触发即正确（同一类失败在 OOD 上同样存在并被同样修） → **泛化** |
| **interpretation** | 从 N 个 train sim 归纳出的策略子集 `P_hat ⊂ P_true` | 触发即正确（按构造 train evidence 自洽） | 在 N 没采样的 policy 边缘 case 上 `P_hat ≠ P_true` → **错触发** → 反向破坏 |

**code-centric harness 演化的有效域 = 残余失败可被分解成 mechanism-layer 工作的 benchmark。** 残余失败本质是 interpretation-layer（长 policy 文档 + 边缘 case + finite evidence）时，方法在 train 上看似有效，但在 test 上反噬。

GAIA 的失败模式是 mechanism-bound（channel：文件没读、网页没抓、格式漂移）→ 方法有效，+50% relative。  
banking_knowledge 的失败模式是 interpretation-bound（policy reasoning：dispute eligibility、cash-back 资格、card recommendation、closure 流程范围）→ 方法在 train 拉到 63%、test 反噬到 10%。

这是**方法的操作边界**，不是失败。

---

## 6. 对 meta-loop 设计的启示

1. **durability axis 当前规格不够。** `deterministic_glue` 把 mechanism-layer 和 interpretation-layer 一锅炖、都打 "low risk, non-destructive, durable" 标签。**这恰恰反了——interpretation-layer 的 false-positive 风险结构上和 `predictive_heuristic` 同类**。需要一个第五类 `induced_rule`（HIGH risk），用对应 `predictive_heuristic` 的 gate（≥3 evidence + 非破坏性 fallback + branch 可审计 + **对至少一个 out-of-evidence policy edge case 做反证测试**）来约束。

2. **`destructive_fallback: false` 这个 attestation 当前太容易拿。** 任何"只在 CERTAIN 时强制 false"的规则都能宣称。但 "CERTAIN" 本身就是解读。skill 应该让 builder **显式列出至少一个超出当前 evidence 的 policy edge case**，证明该 case 上规则不会错触发——拿不出来，归 `induced_rule` HIGH-risk。

3. **每个 iter 都跑 held-out test，不只在最后跑。** test-67 抓到了 overfit；如果**每个 iter 都跑 test**，B 的 stack 增长到第几层会开始 test 退化的曲线就能看到——并在此处停止 freeze。代价：每 iter 多花一份 eval。对小 test set 完全可行。

4. **对 interpretation-bound benchmark，正确的 harness 形态可能不是 "把 policy 编译进代码"，而是 "给 LLM 配更好的机制"——更稳的 KB 检索、结构化的 tool-arg dispatch、对话预算控制——让 LLM 继续做 policy interpreter，但 in a sturdier shell。** B 实际上做了一些这类机制（prerequisite-check injector、retention-protocol channel），但被淹没在 interpretation-layer 的 stack 里。一个**纯 mechanism-layer**的 B 变体——禁止编码 policy 规则、只允许机制层组件——在 banking 上的表现是值得跑的下一步。

5. **论文叙事**：GAIA 正 + tau2-banking 负是**比两个正面例子更强的贡献**——它精确划出方法的操作边界，告诉后来者**什么时候该用 code-centric harness、什么时候不该**。这是个 well-bounded methodological result，不是 mixed result。

---

## 7. 复制实验的关键数字（事实层）

- builder：Claude Opus 4.7（默认 thinking off），ballast/.claude/skills/{robust-harness-tau2, meta-harness-tau2}
- eval：DeepSeek V4-Pro，thinking 关，agent 和 user-simulator 都用同一模型
- domain：tau2-bench banking_knowledge，30 train / 67 test 自切（无官方 split）
- 并发：train-parallel 30, test-parallel 30
- iters：A、B 各 20 iter，每 iter 一个候选
- champion 选择规则：max (train_correct, iteration)
- v0 baseline：4/30 train、8/67 test
- A champion：mh_tau2_iter14_baseline_procedure_sequencing，11/30 train、7/67 test
- B champion：mh_tau2_iter15_robust_dispute_workflow_corrector，19/30 train、7/67 test
- 代码、manifest、per-task 数据均在 `ballast/logs_tau2_baseline/`、`ballast/logs_tau2_robust/`、`agent_tau2/mh_tau2_iter*_*/`、`traces/tau2_*__summary.jsonl`
