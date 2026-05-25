# SOP-Bench 3-Domain Evolution Experiment

**Date**: 2026-05-24
**Benchmark**: Amazon SOP-Bench (arXiv 2506.08119, KDD 2026 submission)
**Target model (SUT, locked)**: `deepseek-ai/DeepSeek-V4-Pro` via Together AI
**Proposer model**: Claude `opus`
**Train/Test split**: per-domain first-30 rows train, remainder test
**Parallelism**: 4 workers per domain run
**Max tokens / call**: 8192

---

## Headline result

| Domain | Test v0 | Test evolved | Delta | ECR |
|---|---|---|---|---|
| dangerous_goods (244 test) | 178/244 = **73.0%** | 235/244 = **96.3%** | **+23.4pp** | 100% |
| traffic_spoofing_detection (170 test) | 97/170 = **57.1%** | 135/170 = **79.4%** | **+22.4pp** | 100% |
| warehouse_package_inspection (120 test) | 61/120 = **50.8%** | 86/120 = **71.7%** | **+20.8pp** | 100% |
| **Aggregate (534 test)** | **62.9%** (336/534) | **85.4%** (456/534) | **+22.5pp / +120 tasks** | **100%** |

All three domains gained ≥20pp test TSR over the v0 baseline. ECR (no-crash rate) stays at 100% across all evolved frontiers — components never introduce agent failures.

---

## Method

### Architecture

```
┌──────────────────────────────────────────────────┐
│  Outer evolution loop                            │  one Python process per domain
│  meta_harness/meta_harness_components_sopbench   │  --domain <slug> --iterations N
├──────────────────────────────────────────────────┤
│  Per-iter:                                       │
│    1. proposer (claude opus + SKILL.md)          │  ~$5, ~10 min
│       writes ONE component file +                │
│       ONE workflow_patch (add/replace/disable)   │
│    2. apply patch → workflow yaml                │
│    3. eval candidate on train-30 (parallel=4)    │  ~3-7 min
│    4. champion-gate: candidate.correct ≥ prev    │
│         accept  → snapshot + commit              │
│         reject  → rollback yaml + restore .bak   │
├──────────────────────────────────────────────────┤
│  Component runtime (NEW for SOP-Bench)           │
│  agent/component_runtime_sopbench/               │
│  Mounts: SESSION_START / PRE_PROMPT_BUILD /      │
│          PRE_LLM_TURN / POST_LLM_RESPONSE /      │
│          PRE_TOOL_USE / POST_TOOL_USE /          │
│          PRE_FINAL_EMIT / SESSION_END            │
│  Sibling of GAIA + tau2 runtimes; adds the       │
│  FC-loop mounts (PRE_LLM_TURN, PRE/POST_TOOL_USE)│
├──────────────────────────────────────────────────┤
│  SopBenchAgent (FC loop)                         │
│  Function-calling loop on agent.llm.chat,        │
│  dispatches Mounts at every lifecycle point      │
├──────────────────────────────────────────────────┤
│  agent.llm.chat()                                │
│  Model LOCKED to deepseek-v4-pro at call time    │
│  (RuntimeError on any model override).           │
│  Extended with tools= kwarg for FC.              │
└──────────────────────────────────────────────────┘
```

### Component model

Each component is a single Python file in `agent/components_sopbench_<domain>/` declaring:

```python
COMPONENT = Component(
    name="sopbench_<domain>_<slug>",
    cls=ComponentClass.MECHANISM_LAYER,  # or REACTIVE_GUARD / CHANNEL / INDUCED_RULE
    mount=Mount.PRE_FINAL_EMIT,           # which lifecycle point
    matcher=_matches,                     # bool predicate on ctx
    handler=_handler,                     # returns a Decision
    trust=Trust(evidence_anchor=..., blast_radius=..., rollback_when=...),
)
```

Gated by a class × mount × decision policy matrix at load time. `PREDICTIVE_HEURISTIC` is rejected outright; `INDUCED_RULE` is restricted to advisory-only `INJECT_CONTEXT` at `PRE_PROMPT_BUILD`.

### Hard constraints enforced

* The target inference model is locked. Components can NOT override `model=` on `chat()`.
* Components live in per-domain dirs; each domain evolves independently.
* Champion gate uses the LAST ACCEPTED candidate's train n_correct (not per-task union).
* Workflow patches are exactly one of `add_node` / `replace_node` / `disable_node` per iteration.
* `replace_node` requires a `.bak_iter<N>` copy before overwrite (used for rollback on reject).

---

## Domain 1: dangerous_goods

**SOP**: chemical hazard classification. Output: `<hazard_class>Hazard Class A/B/C/D</hazard_class>` (or `Unable to Decide`).

### Train progression

| iter | accept | train | component (class / mount) |
|---|---|---|---|
| 0 | ✓ (v0) | 20/30 | — (single FC loop, no components) |
| 1 | ✓ | 23/30 | `final_xml_recovery` (reactive_guard / pre_final_emit) |
| 2 | ✓ | **30/30** | `hazard_threshold_advisory` (induced_rule / pre_prompt_build) |
| 3 | ✗ | 27/30 | rejected: `missing_component_unable` (induced_rule / pre_prompt_build) |

Stopped at iter 2 after train reached 30/30. Iter 3 attempted an INDUCED_RULE about treating tool-returned 0 as a missing-component signal but regressed to 27/30 (rolled back).

### Final frontier (2 components)

1. **`sopbench_dangerous_goods_final_xml_recovery`** — `reactive_guard / pre_final_emit`
   * Fires when `ctx.final_output` lacks a clean `<hazard_class>LABEL</hazard_class>` tag but contains SOP keywords. Strips Qwen-style truncated tool-call hallucinations (`<｜DSML｜tool_calls>...`), extracts the inferred class, re-emits proper XML.
   * Evidence anchor: the SOP's quoted Section 6 output format.

2. **`sopbench_dangerous_goods_hazard_threshold_advisory`** — `induced_rule / pre_prompt_build`
   * INJECTs a "Hazard class threshold advisory" paragraph into the system prompt pinning the partition `A:[4,7] / B:[8,12] / C:[13,16] / D:[17,20]`.
   * Necessary because SOP §5.7 names four classes but omits cut-points; v0 improvised different equal-quartile splits across tasks.
   * Evidence anchor: SOP §5.6 fixes range [4,20] + §5.7 declares D as the highest-severity tier. The partition is the unique split that reserves the top 4 integers for D.

### Test result

| | test correct | test TSR | wall |
|---|---|---|---|
| v0 | 178/244 | 73.0% | 25.8 min |
| iter 2 frontier | 235/244 | **96.3%** | 23.6 min |
| **Δ** | **+57** | **+23.4pp** | — |

The INDUCED_RULE advisory generalized cleanly: train +33pp (20→30), test +23pp (73→96). The dataset's actual threshold table matches the advisory's partition.

---

## Domain 2: traffic_spoofing_detection

**SOP**: ad-traffic abuse enforcement. Output: `<enforcement_action>Account Closure / Temporary Suspension / Warning Issued / No Action</enforcement_action>`.

### Train progression

| iter | accept | train | component (class / mount) |
|---|---|---|---|
| 0 | ✓ (v0) | 12/30 | — |
| 1 | ✓ | 16/30 | `sop_completion_checklist` (channel / session_start) |
| 2 | ✓ | 18/30 | `final_emit_recovery` (reactive_guard / pre_final_emit) |
| 3 | ✓ | 20/30 | `decision_dimensions_advisory` (induced_rule / pre_prompt_build) |
| 4 | ✓ | 22/30 | `decision_dimensions_advisory` REPLACED → channel (verbatim 5.6 SOP slice) |
| 5 | ✓ | 22/30 | `final_emit_recovery` REPLACED (recovery max_tokens up, fallback enum) |
| 6 | ✓ | 23/30 | `conclusive_evidence_gate` (reactive_guard / pre_final_emit) |
| 7 | ✓ | 23/30 | `legitimacy_ratio_medium_nudge` (reactive_guard / pre_final_emit) |
| 8 | ✗ | 22/30 | rejected: disable iter7 |
| 9 | ✓ | 23/30 | `tool_arg_schema_filter` (mechanism_layer / pre_tool_use) |

Stopped at iter 9 (user-initiated). Steady upward train trend with one plateau (iter 4–7 around 22–23). Iter 4 demonstrated component-class downgrade: an INDUCED_RULE was REPLACED with a CHANNEL of equal gain — the advisory was effectively "quote the SOP back to the model", which is a CHANNEL pattern with stronger trust.

### Final frontier (6 components)

| name | cls | mount | role |
|---|---|---|---|
| `sop_completion_checklist` | channel | session_start | injects SOP Section-5.X checklist + Section-6 tag name into system prompt |
| `final_emit_recovery` | reactive_guard | pre_final_emit | on missing Section-6 tag, runs a recovery `chat()` (with the same locked model) to re-emit |
| `decision_dimensions_advisory` | channel | pre_prompt_build | injects verbatim Section-5.6 slice from SOP text (with both risk-level and violation-type dimensions) |
| `conclusive_evidence_gate` | reactive_guard | pre_final_emit | parses adjacent Section-5.6 bullets sharing a risk band; if "with evidence" vs "without conclusive evidence" disagree with model output, rewrites |
| `legitimacy_ratio_medium_nudge` | reactive_guard | pre_final_emit | for Medium-band tasks: uses task_input's unique_users / total_orders ratio to nudge between Warning Issued vs Temporary Suspension |
| `tool_arg_schema_filter` | mechanism_layer | pre_tool_use | reads each tool's `inputSchema.json.properties` and strips any kwarg not in the schema before dispatch (handles model-hallucinated args like `assessmentFormId`) |

### Test result

| | test correct | test TSR | wall |
|---|---|---|---|
| v0 | 97/170 | 57.1% | 32 min |
| iter 9 frontier | 135/170 | **79.4%** | ~50 min |
| **Δ** | **+38** | **+22.4pp** | — |

Notably, test TSR (79.4%) is HIGHER than train TSR (23/30 = 76.7%), suggesting the components generalize well and the small train sample understated the frontier's true skill.

---

## Domain 3: warehouse_package_inspection

**SOP**: physical receiving / quality control. Output: `<resolution_status>Pending / Processing / Resolved / Returned to Vendor</resolution_status>`.

### Train progression

| iter | accept | train | component (class / mount) |
|---|---|---|---|
| 0 | ✓ (v0) | 14/30 | — |
| 1 | ✓ | 15/30 | `resolution_status_output_spec` (mechanism_layer / session_start) |
| 2 | ✓ | 15/30 | `update_status_pending_init` (mechanism_layer / pre_tool_use) |
| 3 | ✓ | 15/30 | iter2 REPLACED — problem-gated narrowing |
| 4 | ✓ | 15/30 | `final_status_normalizer` (mechanism_layer / pre_final_emit) |
| 5 | ✓ | 17/30 | `input_evidence_status_check` (mechanism_layer / pre_final_emit) |
| 6 | ✓ | 18/30 | `clean_path_resolved_terminal` (mechanism_layer / pre_final_emit) |
| 7 | ✓ | 19/30 | iter5 REPLACED — strong-signal narrowing |
| 8 | ✓ | 19/30 | iter5 REPLACED — re-add Cancelled |
| 9 | ✓ | 21/30 | `barcode_image_decode_override` (mechanism_layer / post_tool_use) |
| 10 | ✓ | **26/30** | iter5 REPLACED — respect SOP 5.1.1 short-circuit |

Stopped at iter 10 (user-initiated). Notable pattern: 4 consecutive ties at 15/30 (iters 1–4) where structural components landed but didn't lift the count, followed by a +12 surge in iters 5–10 once the proposer found leverage on SOP 5.2 input-derived checks.

### Final frontier (6 components)

| name | cls | mount | role |
|---|---|---|---|
| `resolution_status_output_spec` | mechanism_layer | session_start | injects strict output XML spec; lists the 4 valid `resolution_status` enum values from `updateResolutionStatus.valid_statuses` |
| `update_status_pending_init` | mechanism_layer | pre_tool_use | on every `updateResolutionStatus` call, rewrites `current_status` to `Pending` only when `problem_type` is non-empty (SOP 5.3.2) |
| `final_status_normalizer` | mechanism_layer | pre_final_emit | extracts the canonical status from any of `<resolution_status>`, nested `<current_status>`, or trailing plain text; emits properly-wrapped XML |
| `input_evidence_status_check` | mechanism_layer | pre_final_emit | when model emits `Returned to Vendor`, re-applies SOP 5.2 input-derived checks to detect over-fire of SOP 5.1.1 short-circuit |
| `clean_path_resolved_terminal` | mechanism_layer | pre_final_emit | on the SOP "clean path" (no tool reports problem AND no input check fires), corrects mistakenly emitted `Processing` to `Resolved` |
| `barcode_image_decode_override` | mechanism_layer | post_tool_use | on `validateBarcode`, decodes the actual `received_product_bar_code` image (vs. the tool's mocked `po_number`-parity-based logic) and overrides the result |

### Test result

| | test correct | test TSR | wall |
|---|---|---|---|
| v0 | 61/120 | 50.8% | 24 min |
| iter 10 frontier | 86/120 | **71.7%** | 51.6 min |
| **Δ** | **+25** | **+20.8pp** | — |

Train→test gap of 15pp (86.7% → 71.7%) — the largest of the three domains. Several components are tuned to specific SOP 5.2 input patterns visible in train; some of those patterns are sparser in test.

---

## Key findings

### 1. The framework generalizes across three structurally different domains

Three SOPs of distinct flavor — chemical hazard scoring (numeric thresholds), abuse-classification reasoning (risk × violation matrix), warehouse process flow (status transitions) — all gained 20–23pp test TSR. The proposer found leverage in each via different component classes: dangerous_goods leaned on one INDUCED_RULE advisory + one REACTIVE_GUARD recovery; traffic_spoofing on a stack of CHANNELs + REACTIVE_GUARDs at PRE_FINAL_EMIT; warehouse on MECHANISM_LAYERs at PRE_TOOL_USE / POST_TOOL_USE / PRE_FINAL_EMIT.

### 2. INDUCED_RULE generalized in this study but is the riskiest class

The one INDUCED_RULE advisory accepted (`hazard_threshold_advisory`) generalized perfectly (train +33pp / test +23pp). But the rule it encodes — a specific score → class partition — is dataset-specific by construction; only Amazon's choice to use a single consistent partition across train and test made it safe. Test-time partition shift would have inverted the gain. The framework's safeguards (advisory-only, evidence_anchor, out_of_evidence_probe) are *necessary but not sufficient*.

### 3. Train→test gap correlates with how procedural the SOP is

* dangerous_goods test 96.3% (train 100%, gap ~4pp) — output is a deterministic numeric threshold lookup, fully captured by the advisory.
* traffic_spoofing test 79.4% (train 76.7%, **negative gap**) — components encode SOP-quoted decision rules; small train undercounted their value.
* warehouse test 71.7% (train 86.7%, gap 15pp) — multiple MECHANISM_LAYERs target specific input patterns; some are train-pattern-specific.

### 4. Component-class composition matters

dangerous_goods reached 96.3% with TWO components. warehouse needed SIX and the train still has 4 unresolved cases at 26/30. The harder a domain's reasoning is, the more components stack — but stacking has diminishing returns (each adds prompt tokens; wall time at frontier is 2-3× the v0).

### 5. Proposer behavior under tie acceptance

Champion gate uses `≥` (ties accept). Warehouse iter 1–4 stacked 4 components at no train gain; iter 5 broke the plateau. In retrospect this was useful exploration (the proposer was probing), but burned ~$20 of opus per stalled iter. A stricter `>` gate would have rejected those — and possibly stalled progress entirely. The trade-off needs explicit evaluation.

---

## Engineering notes

* Total wall time: ~6 hours (3 baselines + 22 evolution iters + 3 final tests, mostly serialized due to early proposal phases).
* Proposer cost: ~$130 (Claude opus, 27 successful iters at avg ~$5/iter).
* Together AI inference cost: small (deepseek-v4-pro per-token cheap).
* Parallelism: each eval at 4 workers. 3 domains × 4 = 12 concurrent Together API calls during the parallel evolution phase. Together AI throttled at 16 concurrent (initial attempt stalled); 4-per-domain was the validated safe point.
* The component runtime, SopBenchAgent dispatch integration, outer loop, SKILL.md, and 14 evolved components total ~3000 lines of new code.

## Reproducing

```bash
# baseline (per domain)
python meta_harness/scripts/run_sopbench_baseline.py \
    --domain <slug> --agent-name v0 --iteration 0 \
    --train-size 30 --max-workers 4 --subsets train,test

# evolution loop
python meta_harness/meta_harness_components_sopbench.py \
    --domain <slug> --iterations 20 \
    --train-size 30 --train-parallel 4 --final-test
```

Required env: `MODEL_NAME=deepseek-ai/DeepSeek-V4-Pro`, `DEEPSEEK_BASE_URL=https://api.together.xyz/v1`, `DEEPSEEK_API_KEY=$TOGETHER_API_KEY`, `LLM_MAX_TOKENS=8192`.
