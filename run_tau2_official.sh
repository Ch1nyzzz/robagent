#!/usr/bin/env bash
# Run all 5 tau2 domains via official simulator with Together AI DeepSeek.
# Sequential per-domain, each with internal --max-concurrency 8.
set -euo pipefail
cd "$(dirname "$0")"

export TOGETHER_API_KEY=$(grep "^TOGETHER_AI_API=" .env | cut -d= -f2)
export TOGETHER_AI_API_KEY=$TOGETHER_API_KEY

AGENT_LLM=${AGENT_LLM:-together_ai/deepseek-ai/DeepSeek-V4-Pro}
USER_LLM=${USER_LLM:-together_ai/deepseek-ai/DeepSeek-V4-Pro}
PARALLEL=${PARALLEL:-8}
OUT_DIR=tau2-runs/full

mkdir -p "$OUT_DIR"

run_one() {
  local domain="$1"
  local n="$2"
  shift 2
  echo "=== [$(date +%H:%M:%S)] domain=$domain n=$n ===" | tee -a "$OUT_DIR/run.log"
  tau2 run \
    --domain "$domain" \
    --agent llm_agent \
    --agent-llm "$AGENT_LLM" \
    --user user_simulator \
    --user-llm "$USER_LLM" \
    --num-trials 1 \
    --num-tasks "$n" \
    --max-concurrency "$PARALLEL" \
    --save-to "$OUT_DIR/${domain}.json" \
    "$@" 2>&1 | tee -a "$OUT_DIR/run.log" | tail -1
  echo "=== [$(date +%H:%M:%S)] done $domain ===" | tee -a "$OUT_DIR/run.log"
}

run_one mock 10
run_one airline 50
run_one retail 114
run_one banking_knowledge 97 --retrieval-config bm25
run_one telecom 70

echo "ALL DONE $(date +%H:%M:%S)"
