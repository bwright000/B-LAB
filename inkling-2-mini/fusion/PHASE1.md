# Phase 1: where the 7,764 ops per decoded token come from (2026-11-21)

**Method:**
- Dumped the decode graph with `GGML_SCHED_DEBUG=2`: llama-cli at b12606, ggml-org Q3_K_M (`sched.log`), analysed by `analyze_graph.py`.
- Read `src/models/inkling2.cpp`, `inkling2-loop.cpp` and `build_moe_ffn` in `src/llama-graph.cpp`.
- Counted calls per helper from signature ops: SOFT_MAX gives `mix` calls, SQR gives `clip` calls, SSM_CONV gives `convolve` calls, and the TANH pairs give `control` calls.

**Graph:** 7,764 non-view nodes per token, against ~7,350 GPU kernels measured by Nsight (some nodes are fused or no-ops).
The top op types are MUL_MAT 977, ADD 872, MUL 858, SCALE 629, CONCAT 548, RMS_NORM 514, CONT 480, CPY 337 and TANH 289.

**What the existing CUDA fusion pass already covers:** rms_norm+mul(+rope), add/mul chains, mul_mat+GLU, topk-moe routing, mul_mat+bias and the ssm scan. **None of Inkling's helpers are covered.**

## Ranked fusion targets

| # | Helper (source) | What it does | Calls/token | Ops/call | Ops/token | After fusion | **Saves (ops)** | Difficulty | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `clip()` inkling2.cpp:164 | L2-norm clip of Q, K, R per head: mul, sqr, sum_rows, sqrt, scale_bias, fill, div, clamp, mul | 144 | 8-9 | ~1,250 | 144 | **~1,100** | Easy: row-wise, ≤128 elements per row; a pattern-match fusion with no model change | Best savings-to-effort ratio |
| 2 | `convolve()` :195 | short causal conv + state store + residual + scale; full-attention layers add 4 dynamic taps | 192 | 5 (17 dynamic) | ~1,150 | 192 | **~950** | Medium: state semantics, dynamic taps, n_tokens=1 fast path | Overlaps PR #30139 (in-place recurrent state) |
| 3 | `control()` :240 (+`message()` cont) | loop gate: 2 rms_norm, concat, 3 matvec, 2 tanh, mul/sub/mul/add | 96 | 13 | ~1,250 | ~480 | **~770** | Medium: fuse norm+concat, and the tanh/mul/sub/add epilogue. Merging gain+forget weights saves 96 more matvecs. | Only in loop passes 2-3 |
| 4 | `mix()` :178 | depth attention over block outputs: concat chain, rms_norm, repeat, matvec, softmax, transpose+cont, matvec | 120 | 6 + n_blocks | ~980 | ~120-240 | **~800** | Medium-hard: variable block count. Better as a model-side change (persistent block buffer) plus one fused kernel. | Touches model code that PRs #32695/#32669 are changing |
| 5 | SITU activation, llama-graph.cpp:2667 | `beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb)` = 9 elementwise ops | 48 | 9 | 432 | 48 (or 0 if fused into mul_mat_id+GLU) | **~400** | **Easy:** add `GGML_GLU_OP_SITU` beside geglu/swiglu, which reuses the existing mul_mat+GLU fusion path | Cleanest to upstream. Also speeds up prefill. |
| 6 | Misc: attention gate sigmoid+mul, rel-bias permute/cont, tau muls, logit softcap | | | | ~300 | | ~150 | Easy-medium | Do last |

**Total addressable: ~4,200 of 7,764 ops.** At ~1 us per small kernel, that's about 4-5 ms of the ~13 ms per token, so **~13 → ~8.5-9 ms, roughly 110-118 tok/s** (estimate; kernels aren't all equal).

## Recommended order
1. **SITU GLU op (#5).** Smallest, self-contained and upstream-friendly. It exercises the whole loop (new op, CUDA kernel, CPU reference, test-backend-ops, A/B) at low risk. Expect +3%.
2. **clip fusion (#1).** The biggest win for the effort, as a pure CUDA-side pattern match with no model or format change. Expect about +8%.
3. **convolve decode fast path (#2).** Check PR #30139 first.
4. **control (#3), then mix (#4).** Coordinate with the open Inkling model PRs (#32695, #32669) so we don't build on code that's about to change.

## Upstream state (2026-11-21)
- Open: #32218 Inkling 2 DSpark support; #32695 Inkling-2-Mini looped arch; #32669 apply .scale tensors + fix correctness; #30139 read recurrent states in place.
- No open PR fuses any Inkling-specific op. Recent CUDA fusion PRs (#32731 top-k masking in sparse FA) don't touch these patterns.
