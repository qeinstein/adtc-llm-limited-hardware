# Native Sparse Architecture Speedup Hypotheses

## Objective

Target:

\[
>15 \text{ tok/s}
\]

while preserving the current model's accuracy and behavior.

The governing rule for every experiment is:

> **Speed gains do not count if they materially reduce accuracy.**

Any approximate architectural modification must beat the existing model on speed while remaining within the established quality/accuracy tolerance. If it cannot, kill it.

We should think less like "make each operation 5% faster" and more like:

- Why does this operation exist?
- Why is it executed separately?
- Why is the same input read repeatedly?
- Why materialize an intermediate tensor if the next operation can consume the previous result directly?
- Why execute eight similar computations independently?
- Why execute every layer/expert if the same function can be represented more cheaply?

The current model is roughly:

\[
x
\rightarrow
\mathrm{RMSNorm}
\rightarrow
\mathrm{Mixer}
\rightarrow
+x
\rightarrow
\mathrm{RMSNorm}
\rightarrow
\mathrm{MoE}
\rightarrow
+x
\]

across roughly 40 blocks, with a hybrid of Gated DeltaNet and full-attention blocks.

The MoE uses roughly 256 routed experts with native top-8 routing plus a shared expert.

---

# 1. Joint Top-8 Expert Kernel

## Problem

The same hidden vector \(x\) is sent independently through eight routed experts.

Conceptually:

\[
E_1(x),E_2(x),...,E_8(x)
\]

Each expert repeatedly:

- reads/prepares the same \(x\),
- decodes weights,
- executes gate projection,
- executes up projection,
- executes down projection,
- writes temporary output.

The runtime treats them as eight unrelated GEMVs even though they form one logical operation.

## Idea

Make the primitive:

\[
F(x,\{e_1,...,e_8\})
\]

instead of eight calls to:

\[
F(x,e_i).
\]

Prepare \(x\) once, traverse its coordinates once where possible, execute all selected experts as a single fused kernel, and accumulate directly into one output.

### Accuracy

Exact architectural transformation.

No approximation should be required.

### Potential

High.

This should be one of the first three architectural experiments.

---

# 2. Shared Expert Projection Bases

This is the largest structural hypothesis.

Each expert currently contains independent matrices such as:

\[
W^g_e,\quad W^u_e,\quad W^d_e.
\]

Yet hundreds of experts were trained inside the same layer and may share substantial low-dimensional structure.

Suppose:

\[
W^g_e \approx A^g_e B
\]

and

\[
W^u_e \approx A^u_e B
\]

where \(B\) is shared across all experts.

Then instead of computing:

\[
W_e x
\]

eight times in the original 2048-dimensional space, compute:

\[
r=Bx
\]

once and let each expert operate on \(r\):

\[
A_e r.
\]

For rank \(r=256\):

Current gate/up work is approximately:

\[
8\times2\times512\times2048
\approx16.8M
\]

weight interactions.

Shared-basis version:

\[
256\times2048
+
8\times2\times512\times256
\approx2.62M.
\]

Potential reduction:

\[
\sim6.4\times
\]

for this portion.

The same idea applies to the down projection:

\[
W^d_e \approx C D_e.
\]

Then:

\[
\sum_e\alpha_e W^d_ez_e
=
C\left(\sum_e\alpha_eD_ez_e\right).
\]

The expensive common projection \(C\) happens once rather than eight times.

## Additional benefit

This could dramatically reduce total expert parameter storage, possibly reducing or eliminating the streaming problem itself.

### Accuracy

Approximate unless the matrices happen to admit an exact factorization.

Must use teacher distillation / reconstruction training and be rejected if the quality gate degrades.

### Potential

Extremely high.

This should be one of the first three architectural experiments.

---

# 3. Fused Gate + Up Projection

Each expert computes:

\[
g=W_gx
\]

and separately:

\[
u=W_ux.
\]

Both consume exactly the same \(x\).

Instead define:

\[
W_{gu}
=
\begin{bmatrix}
W_g\\
W_u
\end{bmatrix}.
\]

Then:

\[
\begin{bmatrix}
g\\
u
\end{bmatrix}
=
W_{gu}x.
\]

For top-8 experts this can become one larger selected-expert projection rather than sixteen independent gate/up GEMVs.

Benefits:

- prepare \(x\) once,
- fewer dispatches,
- fewer reads of \(x\),
- potentially fewer quantization/decode passes,
- better SIMD utilization.

### Accuracy

Exact.

### Potential

Medium-high alone; larger when combined with the joint Top-8 kernel.

---

# 4. Direct Weighted Down-Projection Accumulation

Current conceptual execution:

\[
y_e=W^d_ez_e
\]

for each selected expert, followed by:

\[
y=\sum_e\alpha_ey_e.
\]

This creates eight separate 2048-dimensional outputs and then reads them again to combine them.

Instead calculate:

\[
y_i=
\sum_e
\alpha_e
\sum_j
W^d_e[i,j]z_e[j].
\]

Use one output accumulator from the beginning.

So:

\[
8\text{ expert outputs}
\rightarrow
1\text{ direct accumulated output}.
\]

Benefits:

- fewer writes,
- fewer reads,
- no temporary expert-output tensors,
- no separate weighted-sum pass.

### Accuracy

Exact, modulo normal floating-point accumulation-order differences.

### Potential

Medium and likely stacks naturally with #1 and #3.

---

# 5. Fold RMSNorm Parameters Into Following Projections

RMSNorm computes:

\[
y=
\frac{g\odot x}
{\sqrt{\frac1d\sum_jx_j^2+\epsilon}}.
\]

Define:

\[
c(x)=
\frac1{\sqrt{\frac1d\sum_jx_j^2+\epsilon}}.
\]

Then:

\[
Wy
=
c(x)W\operatorname{diag}(g)x.
\]

Since \(g\) is constant, precompute:

\[
W'=W\operatorname{diag}(g).
\]

Runtime becomes:

\[
W'x
\]

followed by the scalar factor \(c(x)\).

This means the full normalized vector does not necessarily need to be materialized.

### Accuracy

Mathematically exact before quantization.

Quantized representations must be checked carefully because folding and requantization can introduce error.

### Potential

Moderate individually, but this occurs many times per token.

---

# 6. Fuse Residual Addition With Next RMSNorm

Current pattern:

\[
y=x+f(x)
\]

then later:

\[
\operatorname{RMSNorm}(y).
\]

Usually this means:

1. produce \(y\),
2. write \(y\),
3. read \(y\) again,
4. compute \(\sum y_i^2\),
5. normalize.

Instead, while calculating:

\[
y_i=x_i+f_i
\]

also accumulate:

\[
s=\sum_i y_i^2.
\]

Then the RMS normalization scalar is already known when the residual pass finishes.

Combined with hypothesis #5, the normalized tensor may never need to exist as a standalone buffer.

### Accuracy

Exact.

### Potential

Moderate per block, but repeated throughout the entire network.

---

# 7. DeltaNet Single-Pass State Update + Output

A large fraction of Qwen3.5's mixer blocks are recurrent Gated DeltaNet-style blocks.

Generic execution may conceptually do:

\[
S_t=f(S_{t-1},k_t,v_t,...)
\]

and then separately:

\[
y_t=q_t^\top S_t.
\]

If state update and state query traverse the same recurrent state separately, that is potentially redundant.

For a structure like:

\[
S'=aS+\Delta S,
\]

the output is:

\[
q^\top S'
=
a(q^\top S)+q^\top\Delta S.
\]

It may therefore be possible to compute:

- state read,
- update,
- output contribution,

in one fused traversal.

Target:

\[
\boxed{\text{one state read + one state write}}
\]

per layer/token.

### Accuracy

Should be algebraically exact if implemented correctly.

### Potential

High because DeltaNet-style blocks make up much of the backbone.

This should be one of the first architecture-runtime experiments.

---

# 8. Dynamic Expert Count Instead of Fixed Top-8

The model always evaluates eight routed experts.

But router distributions may often look like:

\[
0.45,\ 0.28,\ 0.13,\ 0.07,\ 0.03,\ldots
\]

where the final experts contribute little.

Use adaptive \(K\):

\[
K(x)=
\min
\left\{
k:
\sum_{i=1}^{k}p_i\ge\tau
\right\}.
\]

Then:

- easy token: \(K=2\),
- moderate token: \(K=4\),
- ambiguous token: \(K=8\).

Do not simply hard-code top-4.

Train/distill the adaptive model to match the original top-8 teacher.

### Accuracy

Not intrinsically exact.

Must survive the full accuracy gate.

If quality drops beyond tolerance, reject it.

### Potential

Very high because expert computation dominates.

---

# 9. Replace Eight Experts With One Router-Conditioned Expert

The current function is:

\[
y=
\sum_e\alpha_eE_e(x).
\]

Instead train a conditional expert:

\[
\hat y=E(x,r)
\]

where \(r\) represents router information.

Teacher target:

\[
y^*=
\sum_e\alpha_eE_e(x).
\]

The student learns to approximate the combined transformation directly.

Then:

\[
8\text{ expert evaluations}
\rightarrow
1\text{ conditional transformation}.
\]

The router still carries specialization information, but we stop executing eight full expert networks.

### Accuracy

Approximate and high-risk.

Only keep it if distillation preserves the model's capabilities.

### Potential

Extremely high if successful.

---

# 10. Reduce Network Depth With Block Distillation

Instead of making all ~40 layers slightly faster, ask whether all ~40 are required.

For example, if the repeated structure resembles:

\[
G,G,G,A
\]

where \(G\) is DeltaNet and \(A\) is attention, train fewer stronger blocks to reproduce several teacher blocks.

Example:

\[
G_1\circ G_2\circ G_3
\rightarrow
\tilde G_1\circ\tilde G_2.
\]

If:

\[
40\rightarrow30
\]

layers while preserving quality, theoretical backbone compute falls roughly:

\[
25\%.
\]

More aggressive compression such as:

\[
40\rightarrow24
\]

would be much larger.

### Accuracy

Approximate.

Must be distilled against the original model and rejected on quality loss.

### Potential

Very high because it removes whole computations.

---

# 11. Reduce Full-Attention Frequency

The hybrid model already demonstrates that every layer does not need conventional attention.

If the current pattern is approximately:

\[
G,G,G,A
\]

we can investigate whether a deployment model can use something like:

\[
G,G,G,G,G,G,G,A.
\]

The original attention frequency was chosen for the pretrained general-purpose model, not specifically for short-context CPU deployment.

Potentially distill:

\[
2\text{ attention blocks}
\rightarrow1.
\]

### Accuracy

Approximate.

Must preserve benchmark and long/short-context behavior relevant to the competition.

### Potential

Medium-high depending on actual attention share.

---

# 12. Avoid Full Vocabulary Scoring Every Token

Final decoding computes:

\[
W_{\text{vocab}}h
\]

for roughly 248k vocabulary entries even though only a tiny number are competitive.

Use a two-stage output:

\[
h
\rightarrow
\text{coarse candidate selector}
\rightarrow
\text{small candidate set}
\rightarrow
\text{exact logits}.
\]

For example:

\[
248,000
\rightarrow
2,000
\]

candidate tokens.

Possible mechanisms:

- hierarchical vocabulary,
- approximate maximum inner-product search,
- learned coarse clusters,
- exact rescoring of shortlisted candidates.

Candidate recall must be extremely high.

### Accuracy

Potentially approximate.

Must preserve chosen-token agreement or downstream accuracy.

### Potential

Moderate because the LM head is meaningful but not the entire decode cost.

---

# 13. Incremental Projection Across Consecutive Tokens

Normally every token recomputes:

\[
Wx_t.
\]

But:

\[
Wx_{t+1}
=
Wx_t+W(x_{t+1}-x_t).
\]

Define:

\[
\Delta x_t=x_{t+1}-x_t.
\]

If hidden-state deltas are highly compressible, low-rank, sparse, or concentrated in a reusable basis, then many projections might be updated cheaply rather than recomputed fully.

Possible scheme:

\[
\text{exact anchor}
\rightarrow
\text{several cheap delta updates}
\rightarrow
\text{exact anchor}.
\]

This first requires measuring the actual structure of:

\[
x_{t+1}-x_t.
\]

### Accuracy

Potentially exact if the delta computation itself is exact, otherwise approximate.

### Potential

Unknown but enormous if hidden deltas have exploitable structure.

This is a moonshot measurement first, not an implementation-first idea.

---

# Separate Lever: Multi-Token Prediction (MTP)

MTP is already being tested and should remain separate from the architecture experiments above.

Normal decoding:

\[
1\text{ expensive target step}
\rightarrow
1\text{ token}.
\]

MTP attempts:

\[
1\text{ verification cycle}
\rightarrow
k\text{ accepted tokens}.
\]

If raw decoding reaches:

\[
8\text{ tok/s}
\]

and MTP produces an effective multiplier of:

\[
1.9\times,
\]

then:

\[
8\times1.9
=
15.2\text{ effective tok/s}.
\]

MTP can potentially stack with almost every architecture optimization in this document.

---

# Experimental Order

Wait for the current MTP and staged-Q2_K results first.

Then run architecture investigations in parallel batches of three.

## Batch 1 — Highest expected value

### A. Shared Expert Projection Bases
Potentially attacks expert compute, expert storage, and streaming simultaneously.

### B. Joint Top-8 Expert Kernel
Architecture-preserving and directly attacks repeated work in the current MoE.

### C. DeltaNet Single-Pass State Kernel
Architecture-preserving and attacks a large fraction of the backbone.

These three are attractive because they attack three different surfaces and can potentially stack.

## Batch 2

- Dynamic expert count
- Gate/up + down-accumulation fusion
- Depth/block distillation

## Batch 3

- Router-conditioned single expert
- RMSNorm/residual/projection fusion
- Reduced attention frequency

## Batch 4

- Vocabulary shortlist/head redesign
- Incremental hidden-state projections
- any remaining combinations suggested by telemetry

---

# Accuracy Rule

Every candidate must be compared against the current accuracy-preserving frontier.

For exact transformations:

\[
\text{output differences should be numerical only}.
\]

For learned/approximate transformations:

\[
\text{speed gain}
\land
\text{quality within tolerance}
\]

is required.

If:

\[
\Delta\text{quality}<-\text{allowed tolerance},
\]

kill the idea regardless of speed.

The eventual design may combine several independent wins:

\[
\text{fused exact runtime}
+
\text{compressed/shared MoE}
+
\text{reduced redundant blocks}
+
\text{MTP}.
\]

The objective is not necessarily to discover one magical \(2.3\times\) optimization.

Several multiplicative changes can compound:

\[
1.30\times1.25\times1.20\times1.30
\approx2.54\times.
\]

That would turn:

\[
6.5\text{ tok/s}
\]

into roughly:

\[
16.5\text{ tok/s}
\]

if the gains are sufficiently independent and, critically, if accuracy remains intact.

The model's capability is the invariant.

Everything else — expert structure, number of layers, routing strategy, representation, runtime, execution order — is negotiable.
