# robagent — base agent + 全量基准 trace 收集

最小起点：一次 LLM 调用 + event logging + 官方 scorer，目标是收集 GAIA 和 tau2bench 的真实 trace，后续 harness 由失败驱动生长。

## 目录结构

```
agent/        base agent（无 while-loop）+ event log + Together AI 客户端
bench/        每个 benchmark 的 loader + scorer（官方 vendored / adapter）
harness/      空——等失败 trace 出现后再增长
traces/       每次 run 的事件日志（JSONL）+ summary
evals.lock    scorer 文件的 sha256，CI 用
```

## 准备

1. 依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 环境变量（`.env`）：
   ```
   TOGETHER_AI_API=<your_key>
   MODEL_NAME=deepseek-ai/DeepSeek-V4-Pro
   HF_TOKEN=<huggingface_token>      # GAIA 是 gated dataset，需要 HF 授权
   TAU2_DATA_ROOT=/path/to/tau2-bench/data/tau2   # 可选，否则自动找
   ```

3. tau2-bench 数据（任选其一）：
   - `pip install tau2-bench`（如果 PyPI 有）
   - `git clone https://github.com/sierra-research/tau2-bench tau2-bench-src`

## 跑

```bash
# 烟雾测试
python run_benchmark.py gaia --limit 1
python run_benchmark.py tau2bench --limit 1

# 全集
python run_benchmark.py gaia
python run_benchmark.py tau2bench
python run_benchmark.py all
```

## 产物

- `traces/runs/<bench>__<task>__<run>.jsonl` — 每个 task 一份事件流
- `traces/<bench>__summary.jsonl` — 整体得分汇总

## 不变式

1. `bench/*/scorer.py` 是 read-only，CI 校验 `evals.lock` 里的 sha256
2. agent 代码不允许 import scorer 模块（只能由 `run_benchmark.py` 调用）
3. 所有外部副作用（LLM 调用、tool 调用）必须先 emit event
