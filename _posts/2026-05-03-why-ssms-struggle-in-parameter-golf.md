---
title: "Why SSMs Struggle in Parameter Golf: A Structural Analysis at 25M Parameters"
date: 2026-05-03 00:00:00 -0400
categories: [ML Systems, Research]
tags: [SSM, Mamba, Triton, quantization, LLM, parameter-golf]
---

## TL;DR

Over ~3 weeks of experimentation on an SSM-based submission to OpenAI's Parameter Golf, I converged on a legal Mamba-3 hybrid at **post-quant+TTT 1.1456 bpb**, the best SSM submission in the 16MB track. Despite this, a persistent gap to the transformer SOTA remained. The key contribution of this writeup is not the submission itself but **two structural handicaps** that empirically cap SSMs in this regime. Neither is an architectural bug I could patch, and neither is visible when SSM architectures are studied at larger scales:

1. **LZMA compression penalty**: SSM `in_proj` weights compress ~3× worse than attention's QKV/O under LZMA, inflating compressed model size by ~3.26× beyond raw-parameter-count intuition.
2. **SP4096 → SP8192 non-transfer**: architectural choices validated at a smaller vocab can *flip sign* at a larger one. Embedding-table parameter cost at 25M dominates allocation choices in ways that erase extrapolations from 100M+ literature.

Both are measured. I also flag a third structural concern, Muon optimizer incompatibility with SSM heterogeneous weight rows, as an open hypothesis without direct experimental support (see Section 6).

The total gap to the final transformer winner (1.0611 bpb) is ~85 mBPB. Of that, ~5.5 mBPB is a known engineering debt (the AR GPTQ fallback, Section 3.3), and a small further recovery may be available from untested techniques. Crediting both generously, a best-case fully-patched result lands around ~1.09 bpb, still ~30 mBPB from SOTA. That residual appears paradigm-level: it does not close with SSM-internal technique iteration and would require a fundamentally different architecture or optimizer.

---

## 1. Background

**Parameter Golf** ([openai.com/index/parameter-golf](https://openai.com/index/parameter-golf), [github.com/openai/parameter-golf](https://github.com/openai/parameter-golf)) is a constrained-LM-training challenge run by OpenAI from March to April 2026. Rules: **10 min training on 8×H100, 16 MB decimal artifact (code + compressed model), evaluated on FineWeb validation bits-per-byte.** The competition is now complete. The final leaderboard was topped by transformer variants stacking Muon optimization, INT6 GPTQ, CaseOps tokenizer transforms, parallel residuals, depth recurrence, legal score-first TTT, and LQER compression; the final winner scored **1.0611 bpb** (codemath3000, PR #1855). No SSM submissions appeared on the leaderboard. My earlier submission (PR #1644, 1.1473 bpb) was selected by the organizers as a **notable non-record submission** and merged to the main branch, the only SSM architecture to receive that distinction. A follow-up submission (PR #1890, 1.1456 bpb 3-seed mean) is pending review at competition close.

I chose Mamba-3 hybrid as my submission architecture for two reasons:
1. **Strategic niche**: no competitive SSM submissions existed in the leaderboard. A well-optimized SSM baseline is a meaningful first-of-its-kind contribution even if not #1.
2. **Learning goals**: Mamba-3 SISO uses Triton kernels, making it a great target for someone learning Triton and ML systems engineering.

The architecture I converged on is a **7-layer hybrid: 5 Mamba-3 SSM + 2 attention blocks** ([Mamba-3 SISO](https://arxiv.org/abs/2603.15569)) at d_model=512, d_state=64, expand=2, with SP8192 BPE tokenizer, [Muon optimizer](https://kellerjordan.github.io/posts/muon/) (Jordan, 2024) with [MuonEq-R](https://arxiv.org/abs/2603.28254) row normalization, INT6 weights + INT8 embeddings, per-row mixed-precision quantization for SSM dynamics rows, and 2-epoch score-first chunk TTT. Final submission: **post-quant+TTT 1.1456 bpb (3-seed mean), 15.96 MB legal.**

## 2. What I tested

The architecture that made it to submission was the result of roughly three weeks of experiments across ~20 distinct directions, most of which failed. What follows is a representative sample of both categories; the full experiment log lives in the repository.

**What worked.** The final stack converged on SP8192 tokenization (re-trained from scratch; the community-hosted tokenizer had a byte-counting bug), Muon with MuonEq-R row normalization, and a 7-layer hybrid where d_model=512, d_state=64, headdim=64, and mlp_mult=3, which turned out to be the throughput sweet spot for the Mamba-3 Triton kernels at this shape. For quantization, the most impactful single change was switching embeddings to INT8 with the embed Hessian collected via a `final_norm` output hook during training; this alone closed 90% of the BF16-to-post-quant gap, with matrix GPTQ contributing only the last 6 mBPB. A per-row mixed-precision override for the SSM dynamics rows recovered another 0.8 mBPB at negligible size cost (see Section 3.4). The single largest post-training lever was multi-epoch chunk TTT: increasing from 1 to 2 adaptation epochs recovered 1.4–5.1 mBPB depending on the checkpoint, converting what had been a net quantization regression into a net improvement. Switching to stateful-overlap evaluation (overlap=1024) freed the eval budget for TTT by cutting eval time from ~500s (sliding window) to ~32s.

**What didn't work.** Most failures clustered into three groups.

*Architectural shape.* Narrow-deep configurations (d=384, 12-16 layers) performed worse despite similar parameter counts: the Mamba-3 kernels have a large fixed per-call overhead (~2-3 ms) that doesn't scale with dim, so doubling layer count nearly doubles kernel dispatch cost with diminishing quality return. Wider-shallower (d=640/768, 5-6 layers) didn't clearly win either because MLP cost scales with dim², erasing the SSM throughput advantage. The empirical sweet spot was d=512 at 7-8 layers.

*Recurrence and attention ratio.* Depth recurrence at expand=2 collapsed quality by +67.7 mBPB. At expand=1.5 with mixed-precision dynamics protection it was less catastrophic but still gave +13.9 mBPB worse BF16 at SP8192, despite looking like a clean win at SP4096. Reducing the attention count from 2 to 1 block (7:1 SSM:attn ratio) showed the same pattern: −9.8 mBPB at SP4096 but +7.5 mBPB at SP8192. The 7:1 ratio was motivated by production hybrid deployments: [Jamba](https://arxiv.org/abs/2403.19887) (Lieber et al., 2024) uses 7:1, [Nemotron-H](https://arxiv.org/abs/2504.03624) (NVIDIA, 2025) uses 12:1, and [Falcon-H1](https://huggingface.co/blog/tiiuae/falcon-h1) (TII, 2025) reports that "more attention hurts," but none of those results held at our scale and vocabulary. The section on SP vocab non-transfer explains why.

*Compression and parameter reduction.* Aggressive quantization (MLP INT5: +8 mBPB, base INT5: +17.5 mBPB) confirmed that SSM weight matrices have less quantization headroom than advertised in larger-scale literature. Low-rank factorization of `in_proj` at rank=128 with random init regressed by +26 mBPB and never recovered: Mamba-3 initializes `dd_A` rows as log decay rates for multi-scale temporal tracking, and a random factored product U@V carries none of that structure, forcing the model to rediscover meaningful recurrence dynamics from noise. (SVD initialization of the upstream dense weight would preserve the top-r singular subspace; this remains untested.) Other failed directions: window TTT (gradient signal too weak per window), residual-stream SLOT (no consistent gradient direction on general LM), truncated backpropagation through time with persistent SSM state across training segments (the model learns document-stream-specific recurrent state that is unrecoverable at eval on unseen text), sequence length tweaks, and windowed attention during training.

What was most surprising was not any individual failure but the *pattern* across them. The sections that follow document that pattern.

## 3. Triton kernel engineering: fusion limits, SMEM pressure, and a quantizer driver bug

Three engineering efforts targeted the kernel stack. Two produced negative results with instructive failure modes; one produced a positive result that generalizes beyond this competition.

### 3.1 SMEM-per-stage is the binding constraint, not register pressure

I profiled the Mamba-3 SISO backward pass with a chrome trace at the production shape (chunk_size=64, headdim=64, bsz=32, seq=4096):

| Kernel | Time |
|---|---|
| `mamba3_siso_bwd_kernel_dqkv` | 1190 µs |
| `mamba3_siso_bwd_kernel_rotary_bias_angles` | 588 µs |
| `mamba3_siso_bwd_kernel_dzdo` | 455 µs |
| `mamba3_siso_bwd_kernel_ddt_dtrap_dinput_states` | 18 µs |

All kernels show `regs/thread=255`, the ptxas ceiling. This looks like register pressure is the binding constraint. It is not. Those register spills land in L1 cache (cheap, ~4 cycle latency on H100). The actual binding constraint is **SMEM-per-pipeline-stage**:

- Autotuned winner for `dqkv`: `num_warps=4, num_stages=2, SMEM=107,280 B per stage`
- H100 shared memory budget per SM: 228 KB
- 107 KB × 2 stages = 214 KB, leaving only 14 KB of headroom before the pipeline must degrade to `num_stages=1`

To verify this was already the Pareto front, I extended the autotune grid from 9 configs (3 stages × 3 warps) to 36 configs, adding `maxnreg ∈ {None, 128, 192, 255}` as an outer dimension. The compiler picked the identical winner across all 36 candidates. Constraining `maxnreg` to force register spills into different spill targets did not improve throughput. The SMEM budget is the wall.

### 3.2 dzdo → dqkv backward fusion: correct but slower

`dzdo` computes `dO_scaled` and `dZ` from the output gate, elementwise operations whose outputs are immediately consumed by `dqkv`. The fusion goal: absorb `dzdo` as a prologue inside `dqkv`, eliminating 455 µs × 5 SSM layers = 2.3 ms per step.

**Correctness:** Compared against the unfused reference via relative L2 norm on all 9 backward gradients — `rel_l2 = 0` on every output. Numerically exact.

**Performance:** +1.56 ms wallclock (9.59 → 11.15 ms per Mamba-3 layer, **−16%**). The fusion made things strictly worse.

**Likely root cause:** Absorbing `dzdo`'s z-projection tile adds approximately 8 KB of shared memory per stage. At `num_stages=2` that would push 107 KB → ~115 KB × 2 = 230 KB, just over the 228 KB H100 budget, at which point the autotuner must retreat to `num_stages=1`, sacrificing software pipelining. The latency cost of losing pipelining would far exceed the 455 µs saved by eliminating the kernel launch. I confirmed the SMEM budget and the unfused kernel's autotune winner but did not directly profile the fused kernel's chosen config to verify that `num_stages=1` was indeed selected.

The fused variant is preserved in the repo as an env-gated option but disabled by default. Any future revisit must avoid adding shared memory (for example by keeping the absorbed computation register-resident, or by first reducing the base dqkv tile size to recover headroom), otherwise the same pipeline degradation repeats.

### 3.3 Train-data GPTQ blocked by torch.compile incompatibility: +5.5 mBPB cost

[GPTQ](https://arxiv.org/abs/2210.17323) (Frantar et al., 2022) calibrates quantization by minimizing `||W - Q||_H` where `H` is the layer's activation Hessian. Ideally `H` is collected from actual training data, the same distribution the model learned from. In practice, running a post-training forward pass for Hessian collection in the same process as `torch.compile` reproducibly caused the collection to segfault silently or return garbage. I did not fully diagnose the root cause. Two attempted workarounds: removing the custom Triton allocator hook caused `torch.compile` to spend 12+ minutes recompiling, blowing the eval budget; subprocess isolation was estimated at ~1 day of implementation and not pursued.

The fallback was auto-regressive (AR) GPTQ: generate 32 calibration sequences by sampling from the model itself (~240s), then collect Hessians on those. **Cost: +5.5 mBPB** compared to train-data Hessian collection, measured on comparable SP4096 runs. This was the highest-confidence remaining quality gain in the stack, but I ran out of time to fix it.

### 3.4 Mixed-precision SSM dynamics protection: −0.8 mBPB at negligible size cost

Mamba-3's `in_proj` projects `d_model=512` to a concatenated output of 2232 rows split across semantically distinct groups: `z`, `xv`, `B`, `C`, `dd_dt`, `dd_A`, `trap`, `angles`. At INT6, all rows are quantized uniformly. The `dd_A` and `dd_dt` rows (32 of 2232) feed directly into the recurrence matrix Ā via the SSD kernel's state update:

```
h_t = Ā · h_{t-1} + B_t · x_t
```

Quantization error in Ā *multiplies* through every subsequent state transition. Over a sequence of length T, a per-element error ε in Ā compounds to approximately `ε · T` in the final hidden state, whereas quantization error in any other weight (B, C, MLP) contributes additively and independently at each step. At seq_len=4096, this amplification is substantial.

Promoting just the `dd_A` and `dd_dt` rows to INT8 (32 rows × 512 columns × 5 SSM blocks = 81,920 extra bytes before compression, negligible after LZMA) recovered **−0.8 mBPB** with no meaningful size penalty. The implementation threads per-row bit widths through both the GPTQ Hessian path and the percentile-search fallback, with a helper that derives the row offsets from the model config (d_model, expand, d_state, ngroups, headdim) so the mask stays correct across architecture changes.

The principle generalizes: in any model with a multiplicative recurrence (SSM, RNN, linear attention with state), the weights that modulate the state transition accumulate quantization error differently from feed-forward weights and warrant separate treatment in mixed-precision schemes.

## 4. Finding 1: SSM layers compress ~3× worse than attention under LZMA

Measured from three production runs on 8×H100:

| Hybrid ratio | Raw INT6 payload | LZMA compressed | Reduction |
|---|---|---|---|
| 5 SSM + 2 attn (PR #1644 shape) | 25.32 MB | 15.08 MB | **40.4%** |
| 6 SSM + 1 attn | 26.21 MB | 17.98 MB | **31.4%** |
| 7 SSM + 1 attn | 27.38 MB | 18.24 MB | **33.4%** |

Swapping one attention block for one SSM block adds ~1M raw parameters but **~2.9 MB of compressed artifact**, a 3.26× amplification beyond nominal param count.

**Candidate mechanism (untested).** Mamba-3's `in_proj` weight matrix concatenates rows with structurally distinct roles and initialization schemes:
- `z` (gating), `xv` (values): standard projections
- `B`, `C`: state selection matrices
- `dd_dt`: initialized so `softplus(dd_dt + dt_bias) ≈ 1`
- `dd_A`: log decay rates, initialized for multi-scale temporal tracking
- `trap`, `angles`: additional structured parameters

Each group has a different natural scale and distribution. The hypothesis is that this within-tensor heterogeneity produces a high-entropy byte stream after INT6 quantization, with different row groups mapping to different integer sub-ranges with no consistent Markov structure, leaving LZMA's dictionary-based compression little to exploit. Attention's Q/K/V/O matrices, by contrast, are structurally homogeneous: same init scheme, same role, same natural scale across all rows, yielding a more compressible byte stream. I did not directly measure the per-row-group entropy or byte distributions to confirm this account.

**This is not a quantization bug we could patch.** Mixed-precision protection of the most sensitive rows (dd_A, dd_dt at INT8) helps quality but does not flatten the heterogeneity. The problem is the heterogeneity itself (B, C, z, xv all having different distributions) that hurts compression, not the individual row quantization error.

**Implication**: at 16 MB, an SSM architecture effectively has ~3× less "compressed parameter budget" per raw parameter than a transformer. For a fixed budget, transformers fit ~13% more parameters. Over 25M-scale designs this compounds to several mBPB of inherent disadvantage.

## 5. Finding 2: SP4096 → SP8192 non-transfer

Two architectural choices that I empirically validated at SP4096 *flipped sign* when re-run at SP8192:

### 1-attention ratio
- SP4096 8L: 1-attn at position 4 gave **−9.8 mBPB BF16** vs 3-attn baseline (2×H100 Tier-2 sweep)
- SP8192 7L: 1-attn at position 3 gave **+7.5 mBPB BF16** vs 2-attn baseline (8×H100, direct comparison to PR #1644)

### Depth recurrence at expand=1.5
- SP4096: `NUM_LOOPS=2 LOOP_START=3 LOOP_END=4` gave **−21 mBPB BF16** (prior reported result)
- SP8192 7L: same loop structure with delayed activation and mixed-precision dynamics gave **+13.9 mBPB BF16** vs dense baseline (step 4877 final)

### Why it happens at 25M

At d_model=512, the SP4096 embedding table is 2.10M params (~8% of a 25M budget). SP8192 is 4.19M (~17%). Moving from SP4096 to SP8192 consumes an additional ~8% of the parameter budget for the embedding alone.

At 100M+ parameters, embeddings are a small fraction of total budget; scaling-law literature for SSM vs transformer hybrid ratios, for depth recurrence benefits, for layer-count tradeoffs is mostly derived at this regime. Those scaling laws probably don't transfer to 25M because parameter allocation is qualitatively different.

One candidate explanation: depth recurrence only helps when the looped layers have sufficient representational capacity for the compounded forward pass to add information. When embedding tax forces thinner layers, looping a weaker layer may compound noise per iteration rather than refine representation. The sign flip from −21 mBPB to +13.9 mBPB is consistent with this account, but I did not isolate the embedding-tax contribution from other co-varying factors (different layer counts, different architectural configs between the two experiments).

The practical implication: SP4096 (or any smaller-vocab) sweeps are not reliable predictors of SP8192 behavior at 25M scale. Every architectural finding needs re-validation at the final tokenizer before committing engineering effort.

## 6. What the structural findings imply together

Taken together, the two findings describe a structural advantage transformers have at the Parameter Golf boundary conditions: 25M parameters, 16 MB compressed artifact, 10-minute training. The advantage does not come from specific techniques that SSMs could adopt; it comes from structural properties of transformer weight matrices that happen to align with both of this competition's hard constraints:

1. Homogeneous weight matrices → good LZMA compression (more params fit in 16 MB)
2. Smaller embedding table relative to layer capacity → architectural sweeps at small vocab transfer to larger vocab more cleanly

A third potential axis, optimizer incompatibility, deserves mention as an open hypothesis. [Muon](https://kellerjordan.github.io/posts/muon/)'s update rule operates on the full `in_proj` matrix, treating rows with structurally distinct roles and initialization schemes (`dd_A`, `dd_dt`, `B`, `C`, `z`, `xv`) as a single homogeneous operator. Whether this is harmful compared to a row-group-aware variant is an untested question. I did not run a controlled experiment (e.g., Muon vs AdamW on equivalent SSM checkpoints) that would establish causality.

The transformer's advantage at this scale compounds across both measured axes. An SSM architecture pays a tax on each: ~3× LZMA penalty constrains the compressed parameter budget, and sweep non-transfer means every architectural choice has to be validated from scratch at the target tokenizer before committing engineering effort.

**The final transformer winner scored 1.0611 bpb** (codemath3000, PR #1855). The submission in this writeup is at 1.1456 bpb, an ~85 mBPB gap. Two caveats before calling that gap structural: the AR GPTQ fallback (Section 3.3) costs a known +5.5 mBPB, and SVD-initialized low-rank `in_proj` is untested and could recover a few more. Crediting both generously, a best-case fully-patched result would be roughly ~1.09 bpb, still ~30 mBPB from the final winner. That residual I do consider structural: it does not close with any amount of SSM-internal technique iteration I can see, and would require a paradigm-level change: a different linear-attention variant ([Gated DeltaNet](https://arxiv.org/abs/2412.06464), [RWKV-7](https://arxiv.org/abs/2503.14456), [HGRN-2](https://arxiv.org/abs/2404.07904)), a different quantization scheme that handles heterogeneous weights ([Quamba2](https://arxiv.org/abs/2503.22879)), or an SSM-aware optimizer (the Muon hypothesis above points to one concrete research direction there).

## 7. Submission details

Results correspond to [PR #1890](https://github.com/openai/parameter-golf/pull/1890), submitted 2026-04-28, pending review at competition close.

- Architecture: SP8192 7L hybrid (5 Mamba-3 SSM + 2 attention at positions 2, 5), d_model=512, d_state=64, expand=2, headdim=64, mlp_mult=3
- Quantization: INT6 weights, INT8 embeddings (with embed Hessian via `final_norm` output hook), INT8 mixed-precision for dd_A + dd_dt rows of `in_proj` (32 of 2232 rows per SSM block)
- Evaluation: stateful-overlap (overlap=1024) post-quant eval + 2-epoch score-first chunk TTT
- Commit: `d646ffa` (scale-floor bug fix + production cleanup)

**3-seed results (SEED ∈ {1337, 42, 2025}):**

| Seed | BF16 | Post-quant+TTT | Compressed (B) | Total submission (B) |
|---|---|---|---|---|
| 1337 | 1.1389 | 1.1441 | 15,813,408 | 15,930,191 |
| 42 | 1.1462 | 1.1460 | 15,844,420 | 15,961,203 |
| 2025 | 1.1495 | 1.1468 | 15,858,300 | 15,975,083 |
| **Mean** | **1.1449** | **1.1456** | **15,838,709** | **15,955,492** |
| **Std** | 0.0045 | 0.0011 | 18,852 | 18,852 |

All three seeds individually beat [PR #1644](https://github.com/openai/parameter-golf/pull/1644)'s 1.1473 (itself merged to the main branch and selected as a notable non-record submission) and fit within the 16 MB decimal limit.

- **Δvs PR #1644 (1.1473)**: **−1.7 mBPB** (3-seed mean)
- **Seed variance on submission metric**: 1.1 mBPB std (tight enough that single-seed reports are trustworthy within ±2 mBPB)
- **Net quant+TTT gap is consistently negative**: BF16 mean 1.1449 → post-quant+TTT mean 1.1456 is only +0.7 mBPB higher than BF16. PR #1644's single-epoch TTT had a +8.3 mBPB gap. Multi-epoch TTT recovers nearly all quant damage.

**Training step counts** converged to 5186-5222 across seeds (noise from OS scheduling on 8×H100). Step time stable at 114.3-115.7 ms.

## 8. Ongoing work

The kernel engineering results in Section 3 are based on chrome-trace profiling and autotune grid expansion, but stop short of a systematic roofline analysis of the Mamba-3 SISO forward and backward passes. The planned follow-up is a formal benchmarking pass using Nsight Compute to obtain arithmetic intensity, memory bandwidth utilization, and occupancy measurements at the production shape. The goal is both to sharpen the SMEM constraint claim with measured register file pressure and to identify whether any remaining headroom exists in the kernel that profiling alone missed. If the Nsight pass surfaces actionable improvements, the natural next artifact is a PR to [state-spaces/mamba](https://github.com/state-spaces/mamba) with the findings.

## 9. Acknowledgements

- **OpenAI Parameter Golf organizers** — for creating an unusually rich testbed that exposes small-scale scaling laws that production model development papers over.
- **@tridao and @albertgu** — for the Mamba-3 SSD architecture and pure-Triton kernels ([state-spaces/mamba](https://github.com/state-spaces/mamba)).
- **@clarkkev** ([PR #1394](https://github.com/openai/parameter-golf/pull/1394)) — SP8192 vocabulary and GPTQ on embeddings.
- **@bigbig** ([PR #1217](https://github.com/openai/parameter-golf/pull/1217)) — MuonEq-R row normalization.
- **@abaybektursun** ([PR #399](https://github.com/openai/parameter-golf/pull/399), [PR #549](https://github.com/openai/parameter-golf/pull/549), [PR #756](https://github.com/openai/parameter-golf/pull/756)) — Muon optimizer integration and score-first TTT.
- **@raahilshah, @dexhunter** ([PR #535](https://github.com/openai/parameter-golf/pull/535), [PR #1060](https://github.com/openai/parameter-golf/pull/1060)) — full Hessian GPTQ.
- **@integrate-your-mind, @mikeapedia** ([PR #289](https://github.com/openai/parameter-golf/pull/289), [PR #1089](https://github.com/openai/parameter-golf/pull/1089)) — U-Net skip connections.
- **@mtybadger** ([PR #122](https://github.com/openai/parameter-golf/pull/122)) — sliding window evaluation.
- **@shikhar1729** ([PR #364](https://github.com/openai/parameter-golf/pull/364)) — warmdown schedule.
- **@jfprincz, @newjordan** ([PR #315](https://github.com/openai/parameter-golf/pull/315), [PR #401](https://github.com/openai/parameter-golf/pull/401)) — EMA and logit softcap.
- **Prior SSM submission** [PR #1355](https://github.com/openai/parameter-golf/pull/1355) — the SP1024 Mamba-3 baseline this work builds against.

---

**If any of this is useful to you, I'd love to hear from you.** Specifically:

- **Answers to open questions** — if you have data or intuition on the LZMA compression mechanism, the Muon+SSM hypothesis, or the SP vocab transfer failure, I'd welcome the input.
- **Triton collaboration** — if you work on Triton kernel development or GPU performance engineering and see something worth pursuing in the kernel analysis above, let's talk.
- **Research follow-up** — if the structural findings here connect to work you're doing on SSM optimization, quantization, or small-scale scaling laws, I'm open to collaborating.

Reach out via the contact info at the bottom of [mradassaad.github.io](https://mradassaad.github.io).
