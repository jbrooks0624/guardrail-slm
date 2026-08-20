# Guardrail SLM

QLoRA-tuned Qwen2.5-1.5B prompt-injection classifier, measured against frontier
zero/few-shot, an off-the-shelf guardrail, and a ModernBERT full fine-tune.

The question this repo answers is **when is fine-tuning an LLM the right call
for a narrow classification task?** The answer is whatever the measurements
say. A 149M encoder can match or beat a 1.5B decoder here: the output is one
token from a five-way vocabulary, and generative capacity buys nothing on that
task. If ModernBERT ships, that is the headline. The QLoRA run is required
either way; training wall-clock and GPU cost for both models are part of the
answer, not a footnote.

## Setup

Python 3.12 (pinned). Copy `.env.example` to `.env`.

```bash
cp .env.example .env
uv sync --dev
uv run python -m guardrail_slm --help
uv run python -m guardrail_slm show-config
```

GPU training and vLLM serving run on RunPod (`scripts/runpod_setup.sh`). Dataset
build, labeling, frontier baselines, and plots run locally.

Secrets: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`.
`allenai/wildjailbreak` is gated; accept the terms once and set `HF_TOKEN`.

## Measurement contract

Locked in [`config/eval.yaml`](config/eval.yaml) and
[`config/taxonomy.yaml`](config/taxonomy.yaml) **before any results exist**.
Change these only by committing a new contract, never to fit a number.
`show-config` dumps the same values the code loads.

### Prevalence

Precision is a function of base rate. A figure measured on a balanced test set
does not transfer to traffic that is almost all benign.

| Split | Attack prevalence | Benign prevalence |
| --- | --- | --- |
| Train | 50 percent | 50 percent |
| Primary test | **30 percent attack / 70 percent benign** | |
| Declared deployment | 5 percent and **1 percent** | |

Every precision figure is quoted at the prevalence it was measured or projected
at. The target line looks like: *0.94 precision at our 30 percent test
prevalence, 0.71 at a realistic 1 percent deployment prevalence.*

TPR and FPR do not depend on prevalence. Projection from the same confusion
matrix is:

```
precision(π) = TPR × π / (TPR × π + FPR × (1 − π))
```

### Selection statistic and ship rule

One statistic, used identically by the QLoRA sweep, the ModernBERT
hyperparameter choice, and the ship rule:

**projected precision at 1 percent prevalence, subject to attack-class recall
at or above 0.90.**

The recall floor cannot be relaxed to rescue a config. If nothing clears it,
that is the reported result. Macro-F1 is logged as a secondary diagnostic and
does not decide anything.

**Ship rule:** the model that ships is the one meeting the latency budget with
the best value of that statistic. If that is ModernBERT, ModernBERT ships.

### Latency budget

The guardrail must add **under 100ms p95** to the request it protects. Phase 5
reports pass or fail against that number. A guardrail's SLA is defined by what
it adds, not by isolated throughput.

### Prior shift

Training near 50/50 leaves an implicit prior that attacks are common.
Deploying at 1 percent then makes the model overconfident on the attack class.
Temperature scaling rescales sharpness, not location, so ECE can look healthy
while both operating points sit in the wrong place.

The plan is: train near-balanced, correct the prior at inference. Per-class
logit adjustment maps the training prior onto the declared deployment prior:

```
adjusted_logit[y] = logit[y] + log(p_deploy[y] / p_train[y])
```

Order is fixed: temperature is fitted on validation at the training prior
(sharpness), then the prior adjustment (location). ECE is reported before
correction, after correction, and on a validation set resampled to the
deployment prevalence. The corrected precision must also reproduce the
analytic `precision(π)` projection; disagreement means something upstream is
wrong.

Loss reweighting during training is the rejected alternative: it entangles the
prior with optimization. The logit adjustment keeps the shipped artifact's
prior explicit and adjustable by config, with no retraining.

### Taxonomy

Labels are the single capitals `A`–`E`. Human-readable names live in
[`config/taxonomy.yaml`](config/taxonomy.yaml), never in the model. Binary
attack score is `P(attack) = 1 − P(A)`.

| Token | Name | Meaning |
| --- | --- | --- |
| A | benign | No prompt-injection or jailbreak intent |
| B | roleplay | Persona or roleplay jailbreak |
| C | instruction_override | Bare instruction override |
| D | encoding_obfuscation | Encoding or obfuscation of a payload |
| E | context_injection | Indirect injection via retrieved or pasted context |

The phenomenon is multi-label (a roleplay jailbreak, base64-encoded, inside a
retrieved document is B, D, and E at once). The model is single-label. The
documented precedence rule is delivery channel beats encoding beats persona
beats bare override:

```
E > D > B > C > A
```

The schema stores both `label` (the precedence winner, what the model trains
on) and `categories_all` (the full set). Per-category recall is partly a
statement about this convention and about LLM-assigned labels on sources that
only ever carried binary labels. A 200-example hand audit with reported
agreement covers the labeler; the precedence rule covers consistency.

### Test slices

Every model is scored on all three slices. Headline numbers come from
**natural** only.

| Slice | What is in it | Why it is separate |
| --- | --- | --- |
| `natural` | Human-authored attacks (deepset, jackhhao), Dolly easy benign, held-out wildjailbreak `adversarial_benign` hard benign. Fixed at 30 percent attack. | Headline. Frontier models have no generator-in-distribution advantage. |
| `gpt_generated` | Held-out wildjailbreak `adversarial_harmful` | Hardest attacks live here. Frontier baselines have an unearned advantage because wildjailbreak is GPT-generated. |
| `ood` | `xTRam1/safe-guard-prompt-injection` | Untouched by any training run. No declared license, so it is never used for training. |

Own-synthetic data never enters any test slice (train and validation only,
tagged `synthetic: true`). False positive rate is also split by easy versus
hard benign; hard-negative FPR is what drives deployed precision.

### Licensing

The committed split manifest is hashes rather than text. Dolly is CC-BY-SA-3.0
share-alike; redistributing derived text would carry attribution and share-alike
obligations. Hashes sidestep that on purpose.

## Comparisons

All models write identical JSONL through one predict path; one metrics module
computes numbers.

- Frontier zero-shot and few-shot (OpenAI, Anthropic). Few-shot uses five
  fixed exemplars, one per class.
- Off-the-shelf: `protectai/deberta-v3-base-prompt-injection-v2`.
- Encoder: full fine-tune of `answerdotai/ModernBERT-base`, selected on the
  same statistic as the QLoRA sweep.
- QLoRA: `Qwen/Qwen2.5-1.5B-Instruct`, single-token constrained generation,
  completion-only loss. A sequence-classification LoRA head is rejected
  because vLLM cannot load a `score` layer
  ([vllm#11012](https://github.com/vllm-project/vllm/issues/11012)).

Two named operating points, each quoted at test and deployment prevalence: a
**high-recall** safety posture and a **high-precision** UX posture.

## Results

Empty until eval JSONL exists. A later `report` command will fill this block
from `results/*.jsonl` so no hand-typed number can drift from its artifact.

<!-- results:start -->
No results yet.
<!-- results:end -->

## Layout

```
guardrail_slm/   data, baselines, train, eval, serve
config/          taxonomy, eval, train, sweep, cost
data/            raw / interim / processed (gitignored; split manifest committed)
results/         JSONL, tables, plots
scripts/         runpod_setup.sh
```

## Tests

```bash
uv run pytest
```
