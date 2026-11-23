# Hill-climb log: Inkling 2 Mini on GPU 1

**Targets:** in priority order: c1 decode tok/s for small coding tasks, then prefill and turn-2 TTFT.

**Constraints:**
- 32K context stays stable.
- Zero VRAM spill (watchdog: >1 GiB of borrowed system RAM stops the run).
- System RAM under 6 GB.
- GPU 0 is off-limits.
- Lossless changes only.

**Method:** the skill's `bench.py`, with Windows adapters (launch/stop/probe/queue `.ps1`). Each run takes 6 decode samples (median) from the fixed prompt panel, at temperature 1.0 and low effort. Then a cold prefill at about 4K and 16K. Runs are interleaved as ctl / candidate / ctl.

**Accept rule:** the candidate must beat both neighbouring controls on c1, with no regression over 3%.

## Rejected earlier (don't rerun without a new reason; see ../RESULTS.md)

| Lever | Why it was rejected |
|---|---|
| DSpark draft | Lost to the neighbouring controls |
| MTP | No MTP layers in the GGUF |
| n-gram speculation | Slower |
| Expert offload to CPU | Slower |
| Bigger or smaller quant | Same speed (overhead-bound) |
| ub 1024 / 64K context | Spilled |
| reasoning-budget 0 | Loops |

## Experiments

### Round 1 (2026-11-21, 11:10-11:25). Order: ctl-2, bs-1, ctl-3, pdl-1, ctl-4, gopt-1, ctl-5, prio-1, ctl-6
Controls ranged 66.6-68.0 tok/s (about 2% spread; much quieter than overnight, likely because GPU 0 is idle).

| Run | c1 median | Neighbours | Verdict |
|---|---|---|---|
| bs-1 (`-bs`, backend sampling) | **69.44** | 68.00 / 67.85 | Provisional win, +2.2%. Repeat. Prefill unchanged. |
| pdl-1 (`GGML_CUDA_PDL=1`) | 65.67 | 67.85 / 67.75 | **Rejected**: -3% |
| gopt-1 (`GGML_CUDA_GRAPH_OPT=1`) | 67.57 | 67.75 / 66.61 | **Rejected**: inside noise |
| prio-1 (`--prio 2 --poll 100`) | **68.20** | 66.61 / 67.43 | Provisional win, +1.5%. Repeat. |

ctl-1 was voided: the user closed the server console by accident (now launched hidden).

### Round 2 (11:27-11:38). Order: bs-2, ctl-7, prio-2, ctl-8, up-1, ctl-9
| Run | c1 median | Neighbours | Verdict |
|---|---|---|---|
| bs-2 | **69.89** | 67.43 / 69.14 | **ACCEPTED** (2 out of 2 wins over neighbours, about +1-2%). Added `-bs` to the base. |
| prio-2 | 67.16 | 69.14 / 71.64 | **Rejected**: lost both. The round-1 win was noise. |
| up-1 (upstream b12606) | load error | - | Upstream can't read Unsloth's GGUF (`blk.24.attn_q.weight not found`): the two builds lay out inkling2 differently. |

Checked the ggml-org GGUF header (first 20 MB, full tensor list): 24 blocks, **no MTP tensors**. MTP is unavailable in both llama.cpp GGUF lines, so it's a dead end for now.
Next: requantize ggml-org Q8_0 to about Q3 with their imatrix, for a fair test of the upstream engine.

### Round 3 (base = ctl + `-bs`). Order: t4-1, ctl-10, t1-1, ctl-11, mmids-1, ctl-12, nofuse-1, ctl-13
Controls ranged 70.41-70.87 (spread under 1%).
| Run | c1 median | Neighbours | Verdict |
|---|---|---|---|
| t4-1 (`-t 4`) | **72.63** | (ctl-9 is the old base) / 70.41 | +3.2%. Left neighbour invalid, so repeat. |
| t1-1 (`-t 1`) | 71.88 | 70.41 / 70.48 | +2%, beats both. Compare against t2/t4. |
| mmids-1 (`GGML_CUDA_MM_IDS_SORT=1`) | 71.72 | 70.48 / 70.79 | +1.5%, beats both. Retest on top of the thread winner. |
| nofuse-1 (diagnostic) | 67.84 | 70.79 / 70.87 | Fusion is worth about 4%. Launch count matters somewhat, but it isn't the dominant cost. |

### Round 4. Order: t4-2, ctl-14, t2-1, ctl-15, t4-3, ctl-16
| Run | c1 median | Neighbours | Verdict |
|---|---|---|---|
| t4-2 | 70.84 | 70.87 / 71.95 | Lost. |
| t2-1 | 71.61 | 71.95 / 71.72 | Lost. |
| t4-3 | 71.09 | 71.72 / 70.41 | Inside noise. |

**Verdict:** thread count has no real effect, and t4-1's +3.2% was noise. The controls drifted 70.4 to 72.0 across this round, so the single +1.5-2% wins for t1 and mmids are inside drift and are **not accepted**.
**PLATEAU (config levers):** 3 consecutive levers landed inside noise. Accepted so far: `-bs` only.

### Engine test
Requantized ggml-org Q8_0 to Q3_K_M with `imatrix.gguf` (upstream llama-quantize, 2 min): 12.78 GiB, 4.21 bpw. Testing as `upq3` against ctl.

### Engine A/B. Order: (ctl-16), upq3-1, ctl-17, upq3-2, ctl-18
| Run | c1 | Neighbours | prefill 4K / 16K |
|---|---|---|---|
| upq3-1 | **77.59** | 70.41 / 71.14 | 1118 / 1013 |
| upq3-2 | **77.68** | 71.14 / 71.35 | 1113 / 1009 |
| controls (mean) | 71.0 | | 1146 / 1103 |

**Decode +9.2%, 2 out of 2, far outside noise** (all 12 upq3 samples are 76.0-78.4; every control sample is at or below 71.9).
**Labelled trade:** prefill at 16K is down about 8.5% (beyond the 3% rule). For small coding tasks decode dominates wall time, so it's accepted *pending the quality gates*.
Confound: this changes both the engine and the quant file (the upstream engine can't load Unsloth's format), so the gain belongs to the package.
**Gates:** needle at ~25K, plus the coding eval (HumanEval 164 + MBPP 100, low effort), against Q3_K_XL's 98.8% / 90%.

### Gates for upq3
- Needle at 25K: 3/3. Decode at depth 68.3 (vs 65.1). Prefill 839 (vs 970, -13%). Turn-2 0.54 s.
- Coding (low effort): HumanEval 96.3% (vs 98.8), MBPP 92% (vs 90). Combined 250/264 vs 252/264, the same within noise.
- **But:** about 16% more tokens per HumanEval answer (559 vs 482), so sec/task is 7.7 vs 7.2 (HE) and 6.2 vs 6.2 (MBPP). The 264-task suite took 31.6 vs 30.2 min.
- **Verdict:** a c1 win that does *not* translate into faster coding tasks, plus a prefill regression. **Not adopted as default.** Shipped as `start_inkling_upstream.ps1` (an option for chat-style use).

## STATUS: plateau (2026-11-21 12:30)

| Metric | Baseline (morning) | Best default now | Upstream option | Ceiling | How measured |
|---|---|---|---|---|---|
| c1 decode | 68.0 | ~70-71 (`-bs`) | 77.6 | ~208 | bench.py, median of 6, temp 1.0, low effort, interleaved controls |
| Prefill ~4K / ~13K | 1136 / 1107 | ~1150 / ~1100 | 1115 / 1010 | ~4400 | bench.py, cold unique prompt |
| Turn-2 TTFT at 25K | 0.46 s | 0.46 s | 0.54 s | - | needle, cache reuse |
| Coding suite | 30.2 min | 30.2 min | 31.6 min | - | eval_code.py, 264 tasks |
| Spill | 0 | 0 | 0 | - | watchdog (>1 GiB shared aborts) |

**What limits further gains (evidence):**
- Decode is at ~34% of the bandwidth ceiling, and quant size has no effect on speed.
- Graphs off is 4.6x slower. Fusion off is -4%. CPU threads, priority and polling have no effect.
- So the cost is inside the GPU kernels (MoE routing/matvec for 128 tiny experts and 48 layer passes), not CPU or launch overhead.
- A newer engine helps (+9%), which confirms kernel-side headroom.

**Remaining levers, ranked:**
1. Wait for upstream/Unsloth kernel work on inkling2 (free; re-run `queue.ps1 upq3 ctl` on new releases).
2. MTP speculative decoding, once a GGUF ships the MTP layers (est. +30-80% on code, based on the skill's 3090 evidence).
3. A different engine (vLLM/SGLang need Linux/WSL; EXL3 has an MTP-capable quant, `grant-ai/Inkling-2-Mini-EXL3-4bpw`). That's a large effort.
4. Kernel profiling with Nsight (out of scope for config tuning).

**Rejected this session:** PDL, GRAPH_OPT, prio/poll, threads 1/2/4, MM_IDS_SORT (inside drift), upstream engine on Unsloth's GGUF (incompatible).

### Phase 2 round 1: SITU op + clip fusion (own build, same binary, env switches)
**Voided:** srcunf-1..3, src-2..4 (15:07-15:25). All of them ran at ~28 tok/s, and so did the prebuilt in a spot check, while GPU 1 was contended.
Both GPUs drive a monitor (`display_active=Enabled`), so desktop rendering on GPU 1's screen time-slices the GPU. That probably also explains the overnight 71→50 drift.

Clean A/B (15:40). Order: srcunf-4, src-5, srcunf-5, src-6, srcunf-6
| Run | c1 | Neighbours | pf 4K / 16K |
|---|---|---|---|
| src-5 | **83.76** | 79.97 / 76.23 | 1194 / 1048 |
| src-6 | **86.51** | 76.23 / 78.53 | 1250 / 1113 |
| srcunf mean | 78.2 | | 1170 / 1032 |

**ACCEPTED: decode +8.8% (2/2), prefill about +4%, no regressions.** llama-bench agrees: fused 89.1 vs unfused 81.6 (+9.2%).
Correctness: test-backend-ops SITU 16/16; greedy output identical (3/3); perplexity identical (7.9796). Kernels/token 7,349 → 5,909.

### Environment change (2026-11-21 ~16:00)
The user moved the VS248 monitor off GPU 1, so both monitors are now on GPU 0 and **GPU 1 drives no display**.
Since then the controls agree within ±0.1% (80.06-80.16), against ±5-30% before.

### Phase 2 round 2: convolve fusions (head = CONCAT+state CPY, tail = SSM_CONV+RESHAPE+ADD+SCALE)
- Kernels/token: 5,909 → 5,557 (tail) → **5,365** (head). GPU time/token (median-based): 13.07 (upstream) → **10.90 ms**.
- Correctness: SITU 16/16; greedy output identical (3/3); perplexity identical (7.9796).
- llama-bench: fused tg128 **92.6** vs unfused 81.9 (+13.1%), pp512 1969 vs 1888 (+4.3%).

Server A/B (`srcunf` = all fusions off, same binary). Order: srcunf-7, src-7, srcunf-8, src-8, srcunf-9
| Run | c1 | Neighbours | pf 4K / 16K |
|---|---|---|---|
| src-7 | **89.59** | 80.16 / 80.16 | 1258 / 1126 |
| src-8 | **89.60** | 80.16 / 80.06 | 1258 / 1128 |

**ACCEPTED (all fusions so far): decode +11.8%, prefill +6-7%, no regressions.**
Gotcha: `ggml_can_fuse_subgraph` rejects views of non-constant tensors outside the subgraph (the conv-state cell). Pass the cell view as an output, since we write through it.

### Phase 2 round 3: control gates (TANH+MUL+SUB/ADD) + mix (stacked-prefix concat, no REPEAT at decode)
- Kernels/token: 5,365 → 4,981 (gates) → **4,726** (mix). GPU time/token (median-based) **10.08 ms** (upstream unfused 13.07).
- Still bit-identical: SITU 16/16, greedy output 3/3 identical, perplexity 7.9796 in both modes.
- llama-bench: tg128 **96.7** vs unfused 81.9 (+18%), pp512 2012 vs 1888 (+6.5%).

Server A/B. Order: srcunf-10, src-9, srcunf-11, src-10, srcunf-12
| Run | c1 | Neighbours | pf 4K / 16K |
|---|---|---|---|
| src-9  | **92.91** | 79.98 / 79.98 | 1296 / 1153 |
| src-10 | **93.18** | 79.98 / 80.17 | 1289 / 1153 |

**ACCEPTED: all exact fusions together give decode +16.3% (93.0 vs 80.0), prefill +8-10%.**
Against the user's current default (Unsloth Q3_K_XL + prebuilt, ~70-71 tok/s via server) that's about +31%.

### Phase 2 round 4: removed message() copies (exact) + GGML_OP_INKLING_MIX (fused depth attention, not bit-exact)
**Regression checks on the exact build (all passed):**
- Needle at 25K: 3/3, no spill. Decode at depth 79.3 vs 69.0 unfused.
- `--parallel 2` greedy output identical.
- llama-bench depth sweep: fused faster at 0/8K/16K for both pp and tg.

**Mix op correctness:**
- test-backend-ops 25/25 vs the CPU reference.
- KLD vs the exact graph is 0.00296 (top-1 97.66%), PPL ratio 1.0007 ± 0.0011.
- **Model rounding floor:** the exact graph with llama.cpp's standard CUDA fusions disabled (a pure op-order change) gives KLD 0.00294 / 97.67%. The mix op is AT the floor.
- TF32 hypothesis rejected: f32 cuBLAS gives KLD 0 vs default.

**Kernel iterations (llama-bench tg128):**
- v1 one-block-per-token sequential: 96.9 (pp +45%, but decode slower).
- v2 warp-per-state: 91.3 (worse: latency-bound on 1 SM).
- v3 all-threads-all-states register accumulate + single reduction: **100.4**.

Server A/B (`srcexact` = mix op off). Order: srcexact-1, src-11, srcexact-2, src-12, srcexact-3
| Run | c1 | Neighbours | pf 4K / 16K |
|---|---|---|---|
| src-11 | **96.18** | 94.25 / 94.25 | 1613 / 1397 |
| src-12 | **96.47** | 94.25 / 94.25 | 1578 / 1385 |

**ACCEPTED: decode +2.2%, prefill +22-26% over exact-only.**
**Total vs unfused upstream (80.0): decode 96.3 (+20%), prefill 4K ~1600 vs 1180 (+36%), 16K ~1390 vs 1065 (+31%).**

### Phase 2 round 5: closing the gap
**Diagnosis:** 3,793 of 4,058 kernels/token are < 4 us and cost 6.70 of 9.67 ms. Each costs ~1-1.5 us whatever it does.
PDL / GRAPH_OPT retested on the quiet GPU: no effect (98.2-98.5 vs 98.4-98.5), so the toll can't be overlapped away.

| Change | tg128 (llama-bench) | Exact? |
|---|---|---|
| control: concat-then-rms_norm (one kernel less); `inputs[j]` copies dropped; tanh-gate takes strided input | 100.4 → 101.7 | yes (bit-identical, verified) |
| mix op split into 2 multi-block kernels for T<16 (1 block/token left 69 SMs idle) | 101.7 → **105.2** | floor |
| control 2nd half fused: 2 fp32 matvecs + both gates → 1 warp-per-output kernel (T<=8 only; prompts keep GEMM, else pp -13%) | 105.2 → **111.6** | floor |

**Decode-path KLD** (perplexity with `-ub 1`, 8 chunks, vs the batched exact base): unfused 0.00313 (floor), full build **0.00291**, so at the floor.
Prefill-path KLD for the full build: 0.00294 (floor 0.00294).
| conv tail absorbs the trailing residual ADD (SSM_CONV RESHAPE ADD SCALE ADD -> 1), exact | 111.6 → 111.8 | yes (bit-identical, verified) |

Server A/B (`srcexact` = mix op + control fusion off): src-13/14 **105.5** vs srcexact 99.0-99.3. Prefill 4K 1613 vs 1277, 16K 1405 vs 1142.
**Total vs unfused upstream (server): decode 80.0 → 105.5 (+32%), prefill 4K 1180 → 1613 (+37%), 16K 1065 → 1405 (+32%).**
Kernels/token 7,349 → **3,706**. GPU time (median-based) 13.07 → **8.54 ms**. llama-bench tg128 81.9 → **111.8** (+37%).

### Gap analysis now (8.54 ms GPU/token, ~8.9 ms wall; ceiling ~3.9 ms)
| Category | ms | Notes |
|---|---|---|
| Weight matvecs (Q3_K/Q4_K/Q5_K/Q6_K) | ~3.1 | at bandwidth: the floor |
| quantize_q8_1 (449) | 0.49 | Sharing needs a persistent buffer outside the LIFO pool (alloc must not happen during graph capture). Est. -0.2 ms. PARKED |
| fp32 matvecs (control down 96 + router 48) | 0.52 | down could join the control fusion (2-stage) |
| mix op (2 kernels x 120) | 0.61 | |
| attention (vec + combine) | 0.37 | combine is unneeded at short context |
| conv head + tail (2 x 192) | 0.42 | one decode-step op would halve it |
| rms_norm+mul (154), clip, adds, set_rows, ... | ~1.5 | |
| wall - GPU (graph launch, sampling, sync) | ~0.4 | |
**Each remaining item is worth 2-5%.** Getting near the ceiling would take per-layer megakernels (one kernel per layer instead of ~70), which is research-level work.

## Phase 3: speculative decoding (DSpark = Inkling's MTP drafter), 2026-11-21 night
### PR #32218 review (Inkling 2 DSpark support, pwilkin, 1 commit, 17 files, +672/-120, open)
- **Read all 1,362 diff lines.** Scope: common/sampling.cpp, common/speculative.cpp, llama.h, llama-arch, llama-batch, llama-context, llama-hparams, llama-model, models/{dflash,dspark,inkling2,inkling2-loop,models.h}, tests/test-backend-sampler.cpp.
  No build-system, CI, script, network, process, env or file-I/O additions (automated scan + manual read).
- **Verified:**
  - Output-buffer size accounting after the sampled/probs relocation (`new_size` still sums all float/token counts).
  - `t_layer_inp` is sized LLAMA_MAX_LAYERS+1 (index 48 is fine).
  - Unsloth `dspark-Inkling-2-Mini-Q8_0.gguf` has all 10 required tensors and the metadata (block 8, sample_from_anchor False, SWA, softcap 30, target layers ending at 48).
  - Applies cleanly on our tree (`git apply --check`).
- **Flagged:**
  - (a) `valid &= sum == 1` (exact float equality) on the drafter backend-sampling path may discard proposals. Not hit in our tests: no "invalid proposal" warnings.
  - (b) The drafter head multiplies the full LM head twice per draft token (W*h + W*c). **Fixed** on our branch as W*(h+c). LLAMA_INKLING_DSPARK_TWO_HEADS=1 restores it.
- Applied as a working-tree patch. Our fusion work is saved separately in `src/analysis/inkling-fusion-wip.patch` (+ the untracked inkling-fused.cu/.cuh); the PR is in `src/analysis/pr32218-inkling2-dspark.diff`.

### Findings
- **32K context + DSpark spills** (~200 MB shared): 52 tok/s. 16K f16 and 24K q8_0 fit (no spill).
- **Profile of a verify step at n_max=7 (~34 ms):**
  - MoE expert-union in the 8-token verify: ~10 ms. Each token routes 8 of 128, the union is ~51/128, so ~6x the expert bytes.
  - Drafter double LM head: 7.3 ms.
  - fp32 matmuls at T=8: 3 ms. MMQ+fixup at T=8: 2 ms. q8_0 KV writes: 0.7 ms.
- **Sweep (8K ctx, temp 0, median of 3 prompts):**

| config | tok/s |
|---|---|
| no spec, q8 KV | 110.2 |
| no spec, f16 KV | 113.3-113.6 |
| dspark n=2 (two heads) | 131.7 |
| **dspark n=2, merged head** | **139.4 (+23%)** |
| dspark n=3 merged | 122.0 |
| dspark n=4 merged | 115.8 |
| n=5 / n=7 (two heads) | 99.7 / 95.2 |

  Acceptance at n=2: 681/1028 (mean ~2.3 tokens/step). Larger n adds little acceptance but grows the expert union, so n=2 is optimal on this MoE.
- **Temp 1.0:** ~111-120 tok/s (acceptance ~50%; single requests).
- **Greedy equivalence spec vs no-spec:** diverges early, but every checked divergence is a near-tie:
  - "implementation" 0.202 vs "algorithm" 0.187
  - "Content" 0.317 vs "Explain" 0.310
  - This is the rounding floor (verify-batch kernels vs single-token kernels), not a verification bug.
- **Long-run collapse (first eval attempt):** tasks 1-32 ran at 101-132 tok/s, then a sudden drop to 3-4 tok/s at task 33 with shared memory 502 MB (spill).
  A leak test (30 requests, spec on/off) showed flat memory and steady speed, so this was outside interference on GPU 1 (VRAM taken by another process). The culprit isn't identified.
  Headroom with DSpark at 16K is only ~0.9 GB.
- **Coding eval (re-run with a GPU-1 guard; min per-task 88.5 tok/s):** fused build + DSpark n=2, 16K f16:
  - HumanEval **97.0%** (vs 96.3% unfused no-spec), MBPP **92%** (=)
  - **4.7 s / 3.7 s per task vs 7.7 / 6.2 (-39% / -40%)**
  - Gen 123.5 tok/s vs 75.5.
- Shipped as `start_inkling_fast.ps1`.
- **32K + DSpark fits:** -ub 128, q8_0 KV, drafter KV q8_0, --spec-draft-n-ubatch 16. With a 25K-token prompt: no spill, needle 3/3, prefill 875 tok/s, decode at 25K depth 108 tok/s, short-context 131.4 tok/s.
  The spill at ub 256 came from the main model's compute buffer, not the drafter. `start_inkling_fast.ps1` now uses 32K.
- **Committed on `inkling-fusion` (local):** 4cebc23c9 PR #32218 as published / 6ca86622e fusion work / 5fa5237e4 drafter head fix. Patches in `src/analysis/patches/`.

## Phase 4: engine change, Strata (strata-inkling 0.1.66, release 0.2.20), 2026-11-23
**Why:** public reports of engines much closer to the ceiling (Maya: 128.7 tok/s plain on a 3090 at 781 launches/token; Strata lists a 5070 Ti at 88.0 / 155.5 tok/s for SQ-15 / SQ-10 and ~4,200 tok/s prefill).
**Setup:** everything in `D:\Models\Inki\strata` (engine tarball + control zip + SQ-15/SQ-10 packs, all SHA-256 verified against the release digests / HF LFS oids). No installer, nothing system-wide. GPU 1 only (`CUDA_VISIBLE_DEVICES=1`).
**Method:** `strata\ab.py` feeds the *same prompt token ids* (6 bench.py chat prompts + 4 small coding tasks, low-effort template, 400 new tokens) to llama.cpp (`/completion`) and to `strata-inkling --serve` (stdin line protocol). Decode tok/s excludes prefill on both sides. Control = `start_inkling_fast.ps1` (fused build + DSpark n=2, Q3_K_M-imat, 32K), measured before and after the Strata runs.

### Decode, short context (median tok/s; chat / code)
| Config | temp 0 | temp 1.0 | VRAM | Notes |
|---|---|---|---|---|
| llama.cpp fused + DSpark n=2 (control, before) | 112.9 / 133.1 | 92.9 / 120.0 | 15.4 GB | drafts accepted 50% / 74% |
| llama.cpp fused + DSpark n=2 (control, after) | 113.5 / 136.1 | - | | |
| Strata SQ-10, drafter off | 88.4 / 87.4 | - | 11.7 GB | plain decode is SLOWER than our fused build (~105) |
| **Strata SQ-10, drafter on, 32K fp16 KV** | **136.6 / 179.8** | **99.2 / 139.4** | 12.7 GB | +21% / +33% at temp 0, +7% / +16% at temp 1.0 |
| Strata SQ-10, drafter on, 64K k8v8 KV | 134.8 / 172.4 | 98.9 / 140.2 | 12.4 GB | 64K costs ~nothing at short context |
| Strata SQ-15, drafter on | 58.6 / 81.0 | - | 14.7 GB + host | **does not fit**: 340/3072 experts + KV in host RAM |
| Strata on Unsloth UD-Q3_K_XL GGUF | 55.0 / 55.3 (plain) | - | 14.5 GB + host | 1,013 experts in host RAM; rejects Unsloth's dspark GGUF (`missing tensor dspark.layers.0.self_attn.o_proj.weight`) |

Synthetic `--bench` (4K prompt, 256 tokens): SQ-10 84.3 plain / 160.3 drafted, prefill 2,834-3,407 tok/s; SQ-15 15.6 / 31.5 (hybrid).
**Where the gain comes from:** not plain decode. Strata's speculative step gives 2.0x over its own plain decode (ours: ~1.3x): a W=4 beam tree from the drafter, sized per step from the drafter's own probabilities (maximise expected tokens / verify cost, cost = 1 + 4.2 x (1 - (15/16)^M) for the MoE expert union), verified in one CUDA graph without touching committed KV/conv state, exact GPU argmax, one commit kernel. Temp > 0 uses a linear block draft + rejection sampling, hence the smaller gain.

### Long context (identical documents, low effort, temp 0)
| Prompt | Metric | llama.cpp control | Strata SQ-10 |
|---|---|---|---|
| 19.6K tok | prefill / decode at depth / turn total | 779 tok/s / 104.7 / 28.3 s | 2,972 / 154.1 / 8.1 s |
| 29.7K tok | prefill / decode at depth / turn total | 866 tok/s / 89.9 / 38.2 s | 2,770 / 130.2 / 12.8 s |
| 58.6K tok (64K ctx, `--kv k8v8`) | prefill / decode at depth | n/a (64K variant: ~250 tok/s, 77 at 46K) | 2,203 / 109.9, 13.6 GB VRAM, shared 86 MiB |
| all | needles / turn-2 cache | 3/3, prompt_ms 337-644 | 3/3, prompt_ms 98-144 |
bench.py cold prefill on Strata: 3,109 tok/s (3.3K), 3,127 tok/s (12.9K).
**64K with the default fp16 KV FAILS:** `fatal GPU error: VRAM exhausted growing elastic global KV cache to 49664 tokens` (engine dies, server reloads it). `--kv k8v8` fixes it.

### Coding suite (HumanEval 164 + MBPP 100, temp 1.0)
| Config | HumanEval | MBPP | avg tokens | sec/task | suite time | gen tok/s |
|---|---|---|---|---|---|---|
| llama.cpp fused + DSpark, low effort (2026-11-21) | 97.0% | 92% | 533 / 423 | 4.7 / 3.7 | 19.0 min | 123.5 / 123.8 |
| **Strata SQ-10, low effort** (`serve_think.py`) | 96.3% | 91% | 605 / 407 | 4.9 / 3.3 | 18.8 min | 136.7 / 131.2 |
| Strata SQ-10 as shipped (thinking forced off) | 88.4% | 87% | 123 / 77 | 0.9 / 0.6 | 3.4 min | 175.4 / 157.5 |
| llama.cpp fused + DSpark, Strata-style no-thinking | 86.6% | 81% | 118 / 93 | 1.1 / 0.85 | 4.4 min | 146.0 / 139.7 |
- SQ-10 (10.5 GB, published KLD 0.05) is as accurate as our Q3_K_M-imat on this suite (249 vs 251 of 264): the quant is not the problem.
- Stock Strata renders Inkling with `reasoning_effort=none` + a visible-text opener. That, not the engine, is why it looks 5x faster per task, and it costs 9-10 points on HumanEval on either engine.
- With thinking on, the coding suite takes the same wall time on both engines (+8-10% tok/s, but more tokens written). Same lesson as upq3: tok/s is not time per task.

### Verdict
- **Small coding tasks, low effort (the priority metric): no gain.** Same accuracy, same suite time. Keep `start_inkling_fast.ps1` as default for that.
- **Long prompts / agent use: ACCEPTED as an option** (`start_inkling_strata.ps1`): prompt reading 3.2x faster at 30K, decode at depth +45%, 64K context fully in VRAM, 2.9 GB more headroom. That is the Claude Code worker's bottleneck (23K-token system prompt: ~27 s -> est. ~8 s).
- Anthropic `/v1/messages` tool_use round trip works; 30/30 Windows paths intact in tool arguments (llama.cpp 12/12).
- **Not verified:** Claude Code end to end on Strata, vision, soak, concurrency, KLD of SQ-10 on our own text (only the vendor's 0.050 / 0.072), SQ-10 at temp 0 on the coding suite.
- **Measurement traps:** bench.py's c1 is invalid against Strata in thinking mode (thinking is not streamed, so first-visible-token timing inflates tok/s to 112-584); its `--long` needle reported 0/3 only because 256 max tokens were spent on hidden thinking (3/3 with 1024).

### Remaining levers, ranked
1. Port Strata's confidence-sized draft length to our llama.cpp DSpark path (the drafter's Markov head already yields probabilities; we use a fixed n=2). Est. +10-25% on code at temp 0.
2. Try Strata + SQ-10 as the Claude Code worker server (64K, k8v8) and measure real task wall time.
3. Tree (multi-branch) verify in llama.cpp: large effort; upstream PR #32218 is being rewritten, so wait.
