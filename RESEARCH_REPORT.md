# Jamii Afya Qwen3-0.6B Research Report

I inspected the repository, training scripts, dataset builders, benchmark harnesses, export path, reports, Git history, and relevant recent research. No files were modified, no commits were created, and the working tree remains clean.

The central conclusion is:

> Jamii’s largest current weakness is not that 596M parameters are intrinsically incapable. It is that the training process allocates most learning pressure to competition-style answer ranking, while generation, general knowledge, instruction following, reasoning, and conversational behavior receive comparatively little clean supervision.

The highest-probability route is a clean, broad, full-precision training run from the original \`Qwen3-0.6B-Base\`, using high-quality same-family teacher distillation, explicit replay, a much smaller MCQA component, and on-policy distillation only after the student has acquired basic generation ability.

Exotic methods such as OPSA, model merging, GRPO, or quantization-aware training should be controlled late-stage experiments, not the main strategy.

## 1. Current model diagnosis

### 1.1 Repository and reproducibility caveat

The repository is a useful record of the project’s evolution, but it is not a single perfectly reproducible current run. Several scripts, generated-data paths, reports, and historical checkpoints describe different iterations. Some builders create large derived datasets at runtime, while the committed data and the reports refer to earlier versions. Results therefore need to be attached to exact commit, data manifest, tokenizer/template, checkpoint, and export settings.

That matters here because the reported ARC Easy score and the later Q4_0 recovery are not evidence that the same model has broad capability. They are evidence that one part of the training/evaluation target was optimized effectively.

### 1.2 The original training distribution is overwhelmingly MCQA

The original mixed dataset builder is dominated by multiple-choice material. The effective distribution is not “general instruction plus some MCQA”; it is closer to “MCQA/listwise ranking plus a smaller amount of clinical and general text.” The data includes ARC, OpenBookQA, MedMCQA and related examples, with substantial repetition/upsampling of the competition-relevant component.

This explains the characteristic behavior:

- Strong likelihood ranking of answer choices.
- High ARC Easy performance.
- Weakness on ordinary prompts that do not present four explicit choices.
- A tendency to produce answer-like continuations instead of natural responses.
- Clinical and benchmark associations that are overrepresented relative to everyday language.

The issue is not merely label imbalance. MCQA examples change the learned use of the model. The student is repeatedly rewarded for choosing among already supplied candidates, not for constructing an answer, asking a useful clarification, maintaining a conversation, or deciding when evidence is insufficient.

### 1.3 The custom loss is competition-aligned but generation-hostile when overused

In \`scripts/train_lora.py\`, the MCQA path evaluates each answer choice by appending it to a formatted prompt and computing the conditional log-probability of the answer tokens. It normalizes by answer length and applies a listwise loss so that the gold option ranks above the distractors. The implementation also combines the ranking term with language-model loss on the training sequence.

This is a sensible response to an \`acc_norm\`-style evaluator. It is better aligned with the competition than training only the first answer token. It also explains why the current model can become unusually good at ARC-style questions without becoming generally good at language generation.

The problem is proportion and gradient allocation:

1. The listwise loss receives direct pressure on every candidate comparison.
2. Candidates are often short, templated, and semantically close.
3. The format teaches a stable benchmark protocol.
4. The same examples may be repeated or upweighted.
5. Ordinary SFT examples are not necessarily long or diverse enough to counterbalance that pressure.

The listwise term should remain in the final recipe because it protects competition accuracy, but it should be a minority objective and should be evaluated against generation regressions after every stage.

### 1.4 The normalization is close to, but not guaranteed to be identical with, evaluation

The training code normalizes candidate scores by token length. The competition harness uses its own prompt and tokenization path. Small differences in whitespace, answer prefixes, special tokens, newline placement, truncation, or whether the first token is included can change ranking. The training loss is therefore competition-inspired rather than provably identical to the judge.

The canonical scorer should be reused for offline validation wherever possible. A model can improve the local listwise loss while failing to improve the exact evaluation implementation.

### 1.5 Clinical distribution remains too dominant

\`scripts/build_accuracy_sft.py\`, \`scripts/build_instruction_sft.py\`, \`scripts/build_general_sft.py\`, and \`scripts/build_healthcare_corpus.py\` represent sensible attempts to broaden the model. However, the resulting mixture still carries a strong clinical identity:

- Clinical instruction examples are numerous and often repeated.
- Healthcare causal text adds domain vocabulary and factual associations but does not directly train dialogue behavior.
- Medical MCQA is both a benchmark target and a large source of model updates.
- General examples are present, but they do not appear to dominate the effective token count.

Continued pretraining on health text may help medical recall, but it can also displace ordinary conversational knowledge and make the model more likely to answer every prompt in a clinical register. It is especially risky when the corpus has low diversity, repeated passages, or no balanced general replay.

### 1.6 Repetition and upsampling likely waste training budget

The builders use repetition and/or sampling weights to ensure that important datasets appear in the final corpus. This is understandable under time pressure, but repetition of the same benchmark-style prompts has diminishing returns and can cause memorization of answer formats. It also increases the probability that rare but highly artificial phrasing becomes a dominant style.

The next runs should log:

- raw examples and unique examples per source;
- total examples and total non-padding tokens per source;
- number of epochs-equivalent exposures;
- exact duplicate and near-duplicate rates;
- average prompt, answer, and total sequence lengths;
- source contribution after packing and truncation.

Percentages should be reported by tokens, not only by rows.

### 1.7 The general builders are directionally correct but not sufficient

The general/instruction builders add greetings, ordinary questions, explanations, conversations, factual QA, and instruction-following material. They address a real failure mode seen in earlier versions, where a simple social prompt could trigger a clinical answer.

However, adding a builder is not equivalent to creating a balanced capability curriculum. The decisive questions are:

- How many tokens survive into the actual training manifest?
- How many examples are synthetic or templated?
- How many are ordinary user turns rather than benchmark formats?
- Are answers concise, natural, and varied?
- Are follow-up turns represented?
- Are vague and underspecified prompts represented?
- Are “I don’t know” and clarification behaviors rewarded?
- Are general examples repeated less than MCQA examples?

The code should be treated as a data-generation pipeline whose output must be profiled, not as proof that the model received enough general training.

### 1.8 The healthcare causal-LM corpus is not interchangeable with instruction data

Healthcare text continuation can teach terminology, style, and facts. It does not reliably teach the model to answer a user, follow constraints, summarize, distinguish a question from a passage, or state uncertainty. If mixed indiscriminately with SFT, it can consume updates while producing little improvement on conversational evaluation.

It should be tested as an isolated ablation, a short low-weight warm-up, and a replayed auxiliary component. It should not be assumed to be beneficial simply because the target application is health-related.

### 1.9 Synthetic bilingual and hard-Kiswahili data are useful but high risk

\`scripts/generate_synthetic_data.py\` and \`scripts/generate_hard_swahili.py\` can cheaply fill gaps, but templated synthetic data often has:

- repetitive discourse patterns;
- unnatural translations;
- teacher hallucinations;
- excessive identity statements;
- answer lengths unlike real usage;
- leakage of the expected answer;
- a narrow set of topic and syntax patterns.

Synthetic data should be filtered by language identification, deduplication, length/style statistics, teacher agreement, and human spot checks. Kiswahili should be a meaningful minority stream, not the main curriculum. High-quality English/general data remains the best way to raise the overall capability ceiling.

### 1.10 Identity and refusal examples can overfit

Identity, safety, and refusal examples are necessary, but repeated examples can make the model mention its identity or limitations when the user did not ask. The model needs calibrated uncertainty, not a memorized refusal script. Use varied contexts and include positive examples where the model answers normally, asks a concise clarification, or gives a conditional answer.

### 1.11 The late model versions show that quantization was not the main capability bottleneck

The reports describe a progression in which Q4_0 achieved the strongest speed/memory profile and later training changes recovered or improved benchmark accuracy. Higher precision did not show a statistically meaningful gain on the current evaluation set relative to the throughput cost.

That strongly suggests the main loss is training/data/objective quality, not the fact that inference uses Q4_0. The conclusion should still be rechecked after obtaining a substantially better FP/BF16 checkpoint: quantization sensitivity can change when weights become more specialized or sharper.

### 1.12 Export and calibration are part of the experiment

The export path merges the adapter into the base model, converts to GGUF, and quantizes for llama.cpp-style inference. The deployment path also includes calibration and scalar CPU benchmarking. The final model is still the same 596M-parameter architecture, but the exact quantization type, conversion library version, tokenizer metadata, and prompt template can affect both quality and benchmark behavior.

Every serious comparison needs at least:

- unmerged or merged BF16/FP16 evaluation;
- merged GGUF at the candidate deployment quantization;
- exact competition-style likelihood ranking;
- generation battery;
- tokens/sec, peak RSS, first-token latency, file size, and thermal observation.

### 1.13 Current evaluation is too narrow

ARC Easy is an important competition proxy, but it is currently acting as a strong optimization signal and a weak guardrail. OpenBookQA, MedMCQA, and a few generation examples do not measure ordinary general language competence. There is no adequate frozen battery for vague requests, multi-turn behavior, instruction constraints, calibrated uncertainty, or novel Kiswahili.

The reported \(\sim79.5\%\) ARC Easy, \(\sim51.5\%\) OpenBookQA, and \(\sim36\%\) MedMCQA should therefore be interpreted as a capability profile, not a single model quality number.

### 1.14 Strengths worth preserving

Jamii already has several valuable properties:

- The base architecture is efficient and fixed at approximately 596M parameters.
- Q4_0 runs around 20 tokens/sec on the scalar competition-style profiler.
- Peak RSS is around 527 MB and thermal throttling was not observed.
- The MCQA scorer and listwise objective are well aligned with the competition.
- Medical data and Kiswahili work provide useful domain differentiation.
- The project has already explored data builders, synthetic augmentation, export, and benchmarking.

The goal is not to discard the competition skill. It is to prevent that skill from consuming the model’s whole behavioral repertoire.

## 2. Capability bottlenecks

| Rank | Bottleneck | Type | Evidence | Likely remedy |
|---|---|---|---|---|
| 1 | Effective training distribution is MCQA/clinical-heavy | Fixable | High ARC with weak open generation | Token-balanced data and broad SFT/distillation |
| 2 | Listwise ranking is overrepresented | Fixable | Strong choice ranking, poor arbitrary prompting | Reduce weight; retain as a late auxiliary objective |
| 3 | General data quality and token accounting are unclear | Fixable | Builders exist but effective mixture is not audited | Manifest-level profiling, deduplication, quality filtering |
| 4 | Lack of broad same-family teacher supervision | Fixable | Student receives mostly hard labels or generated text | Off-policy and selective on-policy distillation |
| 5 | No robust replay against forgetting | Fixable | Later specialization and corpus continuation can regress dialogue | Token-level replay in every stage |
| 6 | Weak instruction and conversation coverage | Fixable | Reported social and vague-prompt failures | High-quality multi-turn instruction battery |
| 7 | Limited reasoning supervision | Partly fixable | Small model struggles with broad reasoning | Filtered rationales, concise-answer distillation, exact rewards |
| 8 | 0.6B capacity and knowledge ceiling | Architectural/capacity | Cannot retain all broad world knowledge | Distill priorities; improve data efficiency and calibration |
| 9 | Q4_0 quantization error | Probably secondary | Strong speed and acceptable accuracy already | Sweep only after best FP checkpoint |
| 10 | Tokenizer/context/inference constraints | Architectural/system | Long reasoning and multilingual output are costly | Train concise robust answers; preserve template fidelity |

The most important distinction is between a fixable post-training allocation problem and a true capacity limit. The current evidence does not justify concluding that the model has reached its 0.6B ceiling.

## 3. Research findings

### 3.1 Strong-to-weak distillation is the highest-priority research direction

Qwen3’s technical report describes using strong-to-weak distillation, including off-policy and on-policy components, for smaller models. The same-family teacher/student relationship is especially attractive here because tokenizer, architecture family, chat conventions, and latent representations are compatible.

Relevant sources:

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Qwen3 official model card and release material](https://huggingface.co/Qwen/Qwen3-0.6B)
- [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)
- [Generalized Knowledge Distillation](https://arxiv.org/abs/2306.13649)

Why it may help:

- Transfers more information than a one-hot answer.
- Provides natural answer style, structure, and prioritization.
- Can improve factual QA and instruction following without changing inference.
- Same-family teachers reduce tokenizer and formatting mismatch.

Why it may fail:

- The capacity gap from 14B/32B to 0.6B is severe.
- Teacher probabilities can be overconfident or encode behavior the student cannot represent.
- Distilling teacher mistakes or verbosity can waste capacity.
- Pure teacher-forced/off-policy data does not solve the student’s own error distribution.
- A large teacher-generated corpus can still be narrow if prompts are narrow.

Recommendation: begin with filtered response-level and token-level off-policy distillation; add on-policy distillation only after the broad student checkpoint is stable.

### 3.2 Off-policy sequence and token distillation

Off-policy KD trains on prompts and trajectories generated by the teacher or drawn from a curated dataset. The simplest useful version is SFT on high-quality teacher answers. A stronger version stores teacher logits or top-k token probabilities and adds a soft-target loss on response tokens.

At small scale, response SFT is likely to provide most of the gain per unit engineering effort. Full-vocabulary logits are expensive to store and may overfit teacher uncertainty. Top-k logits plus probability mass for “other” is a practical compromise.

A useful token loss is:

\[
L_\mathrm{KD} = T^2 \operatorname{KL}(p_T^T \,\|\, p_S^T)
\]

combined with hard-label NLL:

\[
L = \alpha L_\mathrm{hard} + (1-\alpha)L_\mathrm{KD}.
\]

Use masking so the KD term applies to response tokens and not accidentally to prompt tokens. Keep a hard NLL term because the teacher distribution is not always the correct target and because it stabilizes learning on rare tokens.

### 3.3 On-policy distillation is promising after warm-up

On-policy distillation samples responses from the student, evaluates those same prefixes with the teacher, and trains the student toward the teacher distribution on the states it actually visits. This addresses exposure bias: the student is trained where it makes its own mistakes rather than only on ideal teacher trajectories.

Relevant sources:

- [MiniLLM](https://arxiv.org/abs/2306.08543)
- [GKD: Generalized Knowledge Distillation](https://arxiv.org/abs/2306.13649)
- [On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649)

Practical risks:

- Student sampling is expensive because teacher inference and scoring are needed for every generated prefix.
- Early student generations may be so poor that the teacher has little useful signal on them.
- Naively minimizing forward KL can make the student imitate low-probability teacher tails.
- Long generations amplify style and verbosity errors.

Start with short responses, temperature-controlled student sampling, teacher scoring on the generated trajectory, and a mixture of teacher-generated/retrieved prompts. Weight teacher confidence and stop-gradient through the sampled tokens.

### 3.4 DistiLLM and alternative KL objectives

Forward KL encourages the student to cover teacher-supported modes, while reverse KL is more mode-seeking and can be less tolerant of a capacity-limited student. DistiLLM argues for distillation objectives and sampling strategies that reduce the mismatch between teacher data and student capacity.

Relevant source:

- [DistiLLM: Towards Streamlined Distillation for Large Language Models](https://arxiv.org/abs/2402.03898)

This is worth a small ablation, but it is not a reason to delay the baseline. The first comparison should be:

- response SFT;
- hard NLL plus forward KL;
- an adaptive/skewed KL variant;
- on-policy forward KL.

Do not infer that a more exotic divergence is better from a single aggregate score. Check general generation, calibration, repetition, and MCQA ranking separately.

### 3.5 TAID addresses capacity-gap mismatch

Temporally Adaptive Interpolated Distillation changes the target over training, beginning closer to the student and moving toward the teacher. The motivation is that a small student may be harmed by an immediate demand to match a much larger teacher.

Relevant source:

- [TAID: Temporally Adaptive Interpolated Distillation](https://arxiv.org/abs/2406.09486)

TAID is plausible for 32B-to-0.6B transfer, but implementation adds a schedule and potentially a teacher/student interpolation target. It should be tested only after a direct-KD baseline exists. The key ablation is a matched-data run with:

- hard SFT;
- direct teacher KD;
- TAID with the same teacher calls and token budget.

If TAID does not improve broad capability or stability at matched compute, drop it.

### 3.6 Teacher-soft MCQA ranking is competition-specific and worthwhile

The current MCQA objective uses the correct answer identity and student scores. A strong teacher can provide a distribution over answer choices:

\[
q_i = \operatorname{softmax}(s_i^T/\tau),
\qquad
p_i = \operatorname{softmax}(s_i^S/\tau).
\]

Then use:

\[
L_\mathrm{MCQA}
= \lambda_\mathrm{gold}L_\mathrm{listwise}
 + \lambda_\mathrm{soft}\operatorname{KL}(q\|p)
 + \lambda_\mathrm{lm}L_\mathrm{response}.
\]

This can teach distractor similarity, calibration, and relative plausibility. It is especially relevant to an \`acc_norm\`-style evaluator.

The danger is that teacher choice probabilities are not calibrated probabilities of correctness. They are scores induced by a prompt and answer wording. Use them as a secondary ranking signal, calibrate temperature on a held-out set, and preserve the hard gold loss.

MCQA should be late-stage and low-weight. A good starting allocation is 10–15% of training tokens, with the soft teacher term applied only to this stream.

### 3.7 Hard-example mining and data selection

Student-hardness mining is likely to provide better returns than blindly increasing corpus size. Run the current student over candidate examples and retain a balanced mixture of:

- student wrong / teacher confident;
- student uncertain / teacher correct;
- teacher/student disagreement;
- high loss but valid target;
- representative easy examples for calibration and replay.

Relevant source:

- [LESS: Selecting Influential Data for Targeted Instruction Tuning](https://arxiv.org/abs/2402.04333)

Influence methods such as LESS are valuable as a selection idea, but direct error mining is simpler and more closely tied to Jamii’s actual failure modes. Avoid selecting only high-loss examples: noisy, impossible, ambiguous, or mislabeled examples also have high loss.

### 3.8 Reasoning and rationale distillation

Teacher rationales can teach decomposition, intermediate checks, and answer verification. This is most likely to help GSM8K-style arithmetic, science, medical scenarios, and logical tasks when rationales are correct and concise.

Relevant sources:

- [Distilling Step-by-Step](https://arxiv.org/abs/2305.02301)
- [STaR: Self-Taught Reasoner](https://arxiv.org/abs/2203.14465)
- [Quiet-STaR](https://arxiv.org/abs/2403.19103)

For a 0.6B deployment model, do not train exclusively on long visible chains of thought. They consume context, can make answers verbose, and may teach brittle teacher-specific prose. Use short verified rationales or hidden/intermediate training where operationally feasible, then include a later concise-answer stage. For exact-answer domains, verify arithmetic or symbolic answers before accepting a rationale.

### 3.9 Full fine-tuning versus QLoRA

The base has only about 596M parameters, and inference cost is fixed by the final architecture, not by whether training used LoRA. Full fine-tuning may therefore raise the attainable capability ceiling, especially for broad distributional changes that a rank-32 adapter cannot express.

Risks:

- Catastrophic forgetting is easier with unrestricted updates.
- A bad mixture can permanently move the base away from general language behavior.
- Optimizer and activation memory are larger.
- Full FT may amplify noisy synthetic labels.

The correct comparison is not one arbitrary full-FT run. Compare matched tokens and validation checkpoints:

- current QLoRA r32;
- LoRA r128 or rsLoRA;
- DoRA;
- full FT with conservative learning rate and replay.

The full-FT run should start from the original base, use a small learning rate, nonzero but modest weight decay, gradient clipping, and frequent general replay. The adapter baselines are useful because they separate “method limitation” from “data/recipe limitation.”

### 3.10 Alternative PEFT methods

Relevant sources:

- [LoRA](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [DoRA](https://arxiv.org/abs/2402.09353)
- [PiSSA](https://arxiv.org/abs/2404.02948)
- [LoftQ](https://arxiv.org/abs/2310.08659)
- [rsLoRA documentation/implementation](https://huggingface.co/docs/peft)

Assessment:

- LoRA r128/rsLoRA: cheap, easy, and the most important adapter ablation. More rank may help broad shifts.
- DoRA: plausible improvement over LoRA for weight-direction/magnitude adaptation; low inference cost after merge.
- PiSSA: potentially better initialization and quantization behavior; needs careful comparison because initialization changes the early optimization path.
- EVA: useful data-driven initialization, but more preprocessing and not a guaranteed small-model gain.
- LoftQ/QPiSSA/QA-LoRA: most relevant when training directly against quantized weights or targeting quantization damage; defer until FP/BF16 quality is optimized.

None of these changes final parameter count or runtime once merged. The expected gains are smaller and less certain than fixing the data mixture and adding teacher supervision.

### 3.11 Staging and replay

Staging is preferable to one opaque mixed run because it makes interference visible. A sensible order is:

1. broad capability and clean general instruction;
2. teacher-distilled instruction, factual QA, reasoning, and conversation;
3. medical and Kiswahili specialization;
4. low-weight competition MCQA;
5. concise-answer/replay recovery.

Replay should be token-based and present in every later stage. A 10–20% replay fraction is a reasonable starting point, but the right value depends on the actual token counts. Measure forgetting after every checkpoint rather than assuming a fixed percentage works.

### 3.12 Specialist checkpoints and merging

Specialists are worth a bounded investigation, not as the main plan. Train general, medical, MCQA, and Kiswahili descendants from the same base, then test delta interpolation and task-vector merging.

Relevant sources:

- [Model soups](https://arxiv.org/abs/2203.05482)
- [TIES-Merging](https://arxiv.org/abs/2306.01708)
- [DARE](https://arxiv.org/abs/2311.03099)
- [RegMean](https://arxiv.org/abs/2103.07847)

The strongest first tests are:

- direct interpolation of deltas with a small grid;
- TIES on the general + medical + MCQA deltas;
- a validation-selected “soup” of checkpoints from one stable run.

Risks include conflicting updates, incompatible training trajectories, and an apparent aggregate gain caused by one benchmark while generation regresses. Merging only helps when specialists contain complementary, compatible improvements. It cannot create capacity or recover information that no specialist learned.

### 3.13 GRPO and reinforcement learning

GRPO can be useful when rewards are reliable and verifiable: math, exact structured answers, code, or constrained formats. It is a weak first answer to Jamii’s broad language problem because conversational helpfulness, factuality, and medical safety are difficult to score with a reliable scalar reward.

Use RL only for a narrow late-stage branch:

- verified GSM8K-style arithmetic;
- MCQA correctness/ranking if the reward exactly matches evaluation;
- format/constraint tasks.

Compare it against teacher distillation on the same prompt set. Do not let RL overwrite broad replay.

Relevant source:

- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)

### 3.14 OPSA

On-Policy Self-Adaptation is interesting because it proposes suppressing poor or high-entropy student tokens without requiring a strong teacher for every update. The idea is related to self-training and confidence-aware on-policy optimization.

It is too new and too uncertain to anchor the two-week program. If tested, make it a small, isolated branch after baseline on-policy KD. It must beat a matched self-distillation or entropy-filtered SFT baseline on the frozen battery and must not increase refusal, repetition, or calibration failures.

## 4. Frozen evaluation battery

Before major training, create an immutable manifest and record the exact prompt/template/scoring code. Every checkpoint is evaluated on the same suite.

### General knowledge and QA

- ARC Easy and ARC Challenge.
- OpenBookQA.
- MMLU subsets spanning humanities, STEM, social science, and professional knowledge.
- BoolQ.

### Commonsense

- HellaSwag.
- PIQA.
- WinoGrande.
- CommonsenseQA if available under the project’s licensing constraints.

### Reasoning

- GSM8K with exact numeric matching.
- Selected BBH tasks covering logical reasoning, causal judgment, tracking, and word problems.
- A small private set that prevents benchmark-specific overfitting.

### Instruction following

- IFEval or an equivalent licensed split.
- Constraint following: JSON, bullets, word limits, inclusion/exclusion, transformations, and multi-condition requests.
- Adversarially vague or underspecified prompts.

### Conversation

Maintain a private prompt battery with:

- greetings and social turns;
- vague questions;
- incomplete questions;
- follow-up questions requiring context;
- factual queries;
- comparisons;
- simple explanations;
- summaries and rephrasing;
- ordinary social conversation;
- out-of-domain requests;
- ambiguity and clarification;
- “I don’t know” and uncertainty cases.

Score with a blinded rubric for helpfulness, factuality, relevance, naturalness, and calibration. Use pairwise comparisons against the current checkpoint where possible.

### Medical

- MedMCQA.
- PubMedQA.
- MedQA or permitted splits.
- Private clinical scenarios.
- Hallucination and uncertainty checks.

### Kiswahili

- Ordinary conversation.
- General factual questions.
- Medical questions.
- Novel generation.
- Translation and cross-lingual consistency.

### Competition and systems

- Exact competition-style likelihood ranking.
- ARC Easy.
- Any known hidden-judge proxy.
- Tokens/sec.
- First-token latency.
- Peak RSS.
- GGUF file size.
- Thermal behavior and sustained throughput.

Use a composite dashboard, but never collapse all categories into one number during diagnosis. A weighted score can hide catastrophic conversation regression behind a small MCQA gain.

## 5. Prioritized experiment matrix

| Experiment | Expected upside | Risk | Compute | Difficulty | Inference impact | Metrics to move | Kill criterion |
|---|---:|---:|---:|---:|---|---|---|
| Rebuild token-profiled balanced data; no method change | High | Low | Low | Low | None | General, conversation, instruction, retained ARC | No broad gain after matched tokens |
| Full FT broad SFT + replay | High | Medium | Medium/high | Medium | None | Broad battery and generation | Worse than QLoRA at matched data and no recovery |
| LoRA r128/rsLoRA matched run | Medium | Low | Low/medium | Low | None after merge | Broad battery | No gain over r32 |
| Same-family teacher response SFT | High | Medium | Medium | Low/medium | None | Knowledge, instruction, conversation | No gain over filtered human/curated SFT |
| Hard NLL + top-k teacher KD | High | Medium | Medium/high | Medium | None | Broad capability, calibration | KD hurts generation or only improves teacher-like prompts |
| Student on-policy KD | High | High | High | High | None | Student-hard prompts, robustness | No gain over off-policy KD at matched teacher budget |
| TAID schedule | Medium | Medium | High | High | None | Stability and broad scores | No matched-compute improvement |
| Teacher-soft MCQA | Medium | Low/medium | Low/medium | Medium | None | acc_norm, calibration, distractor margin | ARC gain with broad regression |
| Hard-example mining | Medium/high | Medium | Medium | Medium | None | Weak slices and data efficiency | Selected data no better than random |
| Rationale distillation | Medium | Medium | Medium | Medium | None | GSM8K/BBH/medical reasoning | Verbosity or hallucination increases |
| Specialist deltas + TIES/interpolation | Medium | High | High | Medium/high | None | Pareto frontier across domains | No merged checkpoint beats a parent |
| Narrow GRPO | Low/medium | High | High | High | None | Verifiable reasoning/MCQA | No gain against KD or reward overfits |
| OPSA/self-adaptation | Unknown | High | Medium | High | None | Robustness/calibration | No clear gain or confidence collapse |
| QAT/LoftQ/QPiSSA | Low/medium late | Medium | Medium/high | High | None | Q4 versus FP gap | Does not improve Q4 at matched quality |
| Q4_K/M or mixed quantization sweep | Low/medium late | Low | Low | Low | Possibly speed/RAM | Quality-speed Pareto | No quality gain worth speed/RAM |

## 6. Two-week experimental plan

### Days 1–2: freeze measurement and audit the data

Freeze the current best checkpoint, base revision, tokenizer, prompt format, conversion tools, and benchmark splits. Build the evaluation manifest before training.

Profile every builder output by source and tokens. Deduplicate exact and near-duplicate examples. Identify benchmark contamination, synthetic templates, truncation, answer-length distribution, and repeated rows. Run the current student on a large candidate pool and record correctness, confidence, entropy, margins, teacher disagreement where available, and generation quality.

Parallel branches:

- Evaluation harness and private conversation battery.
- Dataset statistics and filtering.
- Teacher availability, inference throughput, and logit extraction prototype.
- Export/quantization reproducibility.

### Days 3–4: cheap decisive baselines

Run matched-token experiments from the original base:

1. current QLoRA recipe;
2. balanced broad SFT with no MCQA oversampling;
3. full FT broad SFT with replay;
4. LoRA r128/rsLoRA broad SFT.

Use short runs and early checkpoints. The purpose is to determine whether the main gain comes from distribution and training freedom before spending heavily on teacher logits.

### Days 5–7: teacher response distillation and data selection

Generate or curate high-quality teacher responses for the balanced prompt pool. Filter for correctness, length, language, formatting, and repetition. Run:

- teacher-response SFT;
- teacher-response SFT plus 10–20% general replay;
- hard-example-selected teacher SFT;
- a rationale branch on verified reasoning subsets.

Keep a fixed prompt holdout and compare against the best broad-SFT baseline.

### Days 8–10: token KD and on-policy branch

Run the matched token-level KD experiment with top-k teacher probabilities. Test one temperature and one small temperature ablation rather than a broad hyperparameter sweep. Begin on-policy distillation from the best student after warm-up:

- short student samples;
- teacher evaluation on visited prefixes;
- confidence-weighted forward KL;
- hard NLL and replay;
- stop-gradient through sampled tokens.

If direct KD is not clearly useful, do not assume on-policy KD will rescue it; diagnose teacher/data mismatch first.

### Days 11–12: competition and specialization

Add the late low-weight MCQA stage to the best general checkpoint. Compare hard listwise, teacher-soft listwise, and no-MCQA variants. Evaluate whether a short medical/Kiswahili stage with replay improves target slices without broad regression.

Train only the most promising specialist branches. Test delta interpolation, TIES, and checkpoint soups using held-out validation.

### Days 13–14: final selection and quantization

Select the best FP/BF16 checkpoint by a predeclared composite with hard floors for conversation, instruction following, medical safety, and competition accuracy. Export and quantize only the finalists.

Sweep Q4_0, the best nearby Q4_K variant, and at most one higher-precision candidate. Measure exact scalar throughput, peak RSS, first-token latency, file size, sustained thermal performance, and all frozen capability slices. Keep Q4_0 if the quality gain from another format is not material relative to the score function.

## 7. Recommended distillation design

### Teacher

Use the strongest practical same-family Qwen3 teacher available, preferably Qwen3-14B or Qwen3-32B in a compatible instruct/reasoning form. If a reasoning teacher emits very long traces, create separate concise-answer and reasoning subsets. Do not mix teacher variants without recording the source and filtering behavior.

### Prompt mixture

Use a broad prompt pool:

- 25% ordinary factual QA and explanations;
- 20% conversation and vague/incomplete prompts;
- 15% instruction/constraint following;
- 15% reasoning and science;
- 10% medical;
- 5% MCQA;
- 5% Kiswahili;
- 5% uncertainty, correction, and safety.

These are starting token targets, not sacred row percentages. MCQA can be raised to 10–15% in the competition stage.

### Off-policy stage

Begin with teacher-response SFT because it is cheap, robust, and easy to inspect. Use teacher answers that are short enough for the deployment context, fact-checked where possible, and filtered for refusal/template artifacts. Mix curated human-quality data with teacher data so the student does not inherit all teacher idiosyncrasies.

Then add token KD on the response span. Store either:

- top-k token IDs and log-probabilities, plus residual probability mass; or
- quantized logits with a documented dequantization scheme.

Top-k \(k=32\) or \(k=64\) is a reasonable storage starting point. Evaluate storage error against a full-logit reference on a small sample.

### On-policy stage

For each prompt, sample a short student response with temperature around 0.7–1.0 and a modest top-p. Score the exact generated prefix under the teacher. Apply KD only where the teacher is sufficiently confident or where teacher and student disagree meaningfully. Include a hard NLL term against a filtered teacher answer or verified target.

Do not backpropagate through sampled token selection. Mask prompt tokens. Limit generated length initially to roughly 128–256 tokens, then test longer responses only if the task requires them.

### Loss

A practical starting objective is:

\[
L =
0.50L_\mathrm{hard}
0.30L_\mathrm{KD}
0.10L_\mathrm{replay}
0.10L_\mathrm{MCQA}
\]

where the MCQA term is zero outside the MCQA stream and the coefficients are adjusted by stream rather than blindly applied to every example. More practically, use separate batches/streams with loss masks and monitor each component. Do not let the numerical coefficients obscure token-count imbalance.

For MCQA:

\[
L_\mathrm{MCQA}
= 0.6L_\mathrm{gold\ listwise}
  + 0.2L_\mathrm{teacher\ choice\ KL}
  + 0.2L_\mathrm{answer\ NLL}.
\]

These are initial values for an ablation, not guaranteed optima.

### Temperature and capacity gap

Start with KD temperature \(T=2\) or \(T=3\), retain the \(T^2\) scaling, and compare one nearby value. Weight or discard examples with highly diffuse teacher distributions. Use TAID only after direct KD is measured.

The student should not be forced to reproduce teacher verbosity, rare stylistic choices, or uncertain tails. Distill the teacher’s answer decisions and useful token alternatives, not every token with equal weight.

### Dataset size and batching

Prefer tens or hundreds of thousands of high-information examples over millions of duplicated examples. The actual target should be chosen by token budget and teacher cost. Use packed sequences only if prompt/response boundaries and loss masks remain exact. Keep a held-out prompt family split so near-duplicate leakage does not inflate results.

### Checkpoints and stopping

Evaluate after every meaningful fraction of an epoch or fixed token count. Stop a branch when:

- broad generation improves but MCQA collapses beyond the declared floor;
- loss falls while private conversation quality regresses;
- the model becomes more verbose, repetitive, or overconfident;
- the teacher-disagreement slice stops improving;
- Q4 export erases the FP gain.

The checkpoint is selected from the frozen multi-category battery, not final training loss.

## 8. Data curriculum

Recommended starting token shares for the broad phase:

| Stream | Share |
|---|---:|
| General factual QA and explanations | 20% |
| Ordinary conversation and vague prompts | 15% |
| Instruction and constraint following | 15% |
| General reasoning and science | 15% |
| Medical knowledge and reasoning | 12% |
| MCQA/listwise competition data | 10% |
| Commonsense and social reasoning | 5% |
| Kiswahili and bilingual transfer | 5% |
| Uncertainty, correction, and safety | 3% |

The exact proportions should be adjusted by token counts and evaluation results. The broad phase should not use the competition dataset as the dominant source. A late competition stage can temporarily raise MCQA to 15–20%, but it should include 10–20% broad replay and should be followed by a concise general recovery pass.

For the data itself:

- Prefer unique, natural, answerable examples.
- Keep difficult and easy examples; easy examples calibrate style and confidence.
- Include short answers and moderately detailed answers.
- Include multi-turn contexts and follow-ups.
- Include prompts where asking a clarification is the best response.
- Include explicit uncertainty and correction examples.
- Filter teacher outputs for factuality and unwanted identity/refusal behavior.
- Keep medical safety data, but do not let it define all conversational style.
- Treat synthetic data as augmentation, not as the default truth source.
- Reserve private holdouts by source, topic, and prompt template.

## 9. PEFT versus full fine-tuning recommendation

Run both, but make full fine-tuning the primary capability experiment.

The current QLoRA r32 result is a valuable baseline. A LoRA r128/rsLoRA run is a cheap check for adapter rank saturation. DoRA is the only additional PEFT method I would prioritize before full FT because it is relatively simple and has a plausible optimization benefit after merging.

Full FT from the original base should use conservative updates and replay. It is the only branch that directly tests whether the adapter subspace is the bottleneck. Training memory is a concern, but it is a research-time concern; it does not affect final inference memory.

PiSSA/EVA/LoftQ/QA-LoRA are second-wave methods. Run them only if:

- the broad data recipe is already demonstrably better;
- full FT is unavailable or unstable;
- or Q4 quantization becomes the limiting quality loss.

## 10. Model merging recommendation

Do not make merging the main bet. First establish one excellent general checkpoint and one excellent competition checkpoint from a controlled recipe. Then test:

1. direct delta interpolation over a small coefficient grid;
2. checkpoint soup across nearby checkpoints with no obvious regression;
3. TIES merging for general + medical + competition deltas.

Use held-out category scores to select merge coefficients. A merge is successful only if it improves the Pareto frontier: broad capability must not fall materially for a small ARC gain.

DARE and more complicated task arithmetic should be optional follow-ups. Specialist training costs enough that it should happen only after the data audit identifies genuine, separable domain improvements.

## 11. Quantization plan

Quantization comes after the strongest FP/BF16 checkpoint, not before.

For each finalist:

- evaluate BF16/FP16 merged weights;
- export with the same tokenizer and chat metadata;
- quantize Q4_0 using the current reproducible path;
- test one nearby Q4_K format and one higher-precision format;
- compare exact benchmark and generation results;
- measure scalar throughput, RSS, first-token latency, file size, and thermal behavior.

If quality drops after Q4, test calibration/data-aware options such as importance-matrix quantization or QAT/LoftQ only on the final finalist. Do not spend the first week optimizing quantization around a weak checkpoint.

The expected result is that Q4_0 remains attractive because the scoring function heavily rewards throughput and memory. The final choice should be based on total score and capability floors, not benchmark accuracy in isolation.

## 12. Top five bets

1. **Token-balanced broad data rebuild with exact profiling and replay.** The current evidence points to data allocation and objective interference as the biggest fixable problems.

2. **Conservative full-parameter fine-tuning from the original Qwen3-0.6B-Base.** At this model size, the adapter rank may be constraining broad capability, while full FT has no deployment cost.

3. **Same-family teacher-response distillation followed by selective top-k token KD.** This is the most credible way to compress broad knowledge, style, and instruction behavior into the fixed student.

4. **Student-hard example mining plus a short, carefully gated on-policy KD stage.** Use the student’s actual failures to spend teacher/data budget where it matters.

5. **Late competition-aware fine-tuning with teacher-soft MCQA ranking and general replay.** Preserve the ARC/competition advantage without allowing listwise optimization to define the whole model.

The methods I would not bet the two weeks on are a large GRPO run for broad capability, OPSA as the primary recipe, an elaborate specialist-merge program before the data audit, or QAT before the best full-precision model exists. They may be useful controlled branches, but the likely path to a dramatically better 596M model is disciplined data allocation, full-model optimization, high-quality same-family distillation, and evaluation that makes regressions impossible to hide.
