# Inkling 2 fusion project: llama.cpp source build (parked 2026-11-21)

**Goal:** cut the ~7,400 GPU kernels per decoded token. About 6,270 of them are tiny "glue" ops (≈5.8 of ~13 ms) and should be fused into a few custom CUDA kernels for Inkling's ops.
Target: ~100–110 tok/s (from ~77). Background: `../hillclimb/THEORY.md` ("Where the time actually goes").

## State: Phase 0 done
- Source: `llama.cpp/`, upstream ggml-org at tag **b12606** (commit `e412c6f46`, the same as the prebuilt `../llama-up`). Work branch: **`inkling-fusion`**.
- Build: `build.cmd`, the same command for the first and later incremental builds. It uses the already-installed VS 2022 Build Tools (MSVC 14.44), the bundled CMake and Ninja, and CUDA 12.8. Settings: `sm_120` only, Release. Output: `llama.cpp/build/bin/`.
- Verified:
  - `test-backend-ops` passes: ADD 167, MUL 143, SCALE 11, RMS_NORM 121 and MUL_MAT_ID 1077.
  - Decode matches the prebuilt: 77.59 vs 78.62 / 76.59 tok/s (`../hillclimb/ledger.jsonl`, src-1).
- Model for this work: `../gguf/ggml-org/Inkling-2-Mini-Q3_K_M-imat.gguf`, in the upstream format. The Unsloth GGUFs don't load in upstream.
- Open question: prefill at 4K was 987 vs ~1,120 tok/s for the prebuilt (16K was equal). It's a single cold sample, so re-measure. If it's real, compare the prebuilt's build flags (e.g. `GGML_CUDA_FA_ALL_QUANTS`, the arch string `120` vs `120a`).

## Phase 1: DONE (see `analysis/PHASE1.md`)
Ranked targets: clip (~1,100 ops), convolve (~950), control (~770), mix (~800), SITU (~400). That's ~4,200 of 7,764 ops addressable, for an estimated ~110-118 tok/s.
**Start Phase 2 with the SITU GLU op** (easy, upstream-friendly), then the clip fusion.

## Phase 2, round 1: SITU op + clip fusion (2026-11-21, uncommitted on branch `inkling-fusion`)
**Changes:**
- **SITU:** new `GGML_GLU_OP_SITU` / `ggml_situ()`, with a CPU reference (`ops.cpp`), a CUDA kernel (`unary.cu`) and 16 `test-backend-ops` cases. `build_moe_ffn` uses it.
- **clip:** a pattern-match fusion `ggml_cuda_try_fuse_inkling_clip` (`ggml-cuda/inkling-fused.cu`), hooked at the top of `ggml_cuda_try_fuse`. No model or format change. It reproduces SUM_ROWS's reduction order, so results are bit-identical.
- **A/B switches (TEMP, remove before upstreaming):** `LLAMA_INKLING_UNFUSED=1` restores the 9-op SITU graph. `GGML_CUDA_NO_INKLING_CLIP=1` disables the clip fusion.

**Verified (`analysis/verify_fusion.ps1`, same binary fused vs unfused):**
- `test-backend-ops -o SITU`: 16/16 pass.
- Greedy decode at temp 0: 3/3 prompts x 256 tokens **identical**.
- Perplexity (20 x 512, batched path): **7.9796 in both modes** (identical).
- Kernels/token (Nsight): **7,349 → 5,909** (-1,440, matching the estimate). The clip kernel runs 144x/token and SITU 48x/token.

**Open:**
- **Speed A/B DONE: decode +8.8% (fused ~85 vs unfused ~78 tok/s, 2/2 wins over neighbours), prefill about +4%** (hillclimb/PLAN.md). Benchmark only when the desktop is quiet: both GPUs drive a monitor.
- The SITU kernel's median is 9.2 us (measured under contention); expected ~2 us. Re-measure on an idle GPU, then look at it.
- No unit test for the clip fusion yet. It's covered end to end by the identical outputs.

## (Original Phase 1 checklist, kept for reference)
1. Dump the ggml graph for one decode token. Count the op sequences per layer and map them to Inkling features (softcaps, attention-residual, loop gates, sconv, value-norm, qk clip, relative positions, logit softcap).
2. Rank the patterns by kernels/token. Profile names to start from (per token): `scale_f32` ~720, `k_bin_bcast` add ~490 and mul ~570, `quantize_q8_1` ~385, `concat` ~380, `cpy` ~300, `rms_norm` ~240, `reduce_rows` ~190, `transpose` ~190.
3. Check what upstream is already doing (open PRs touching `ggml-cuda` fusion and inkling2) so we don't duplicate work.

## Then, per pattern (Phases 2–5)
fused kernel + hook in the ggml-cuda fusion pass → `test-backend-ops` → greedy output equivalence + KLD + `eval/eval_code.py` + needle → `hillclimb/queue.ps1 src ctl src ctl` → re-profile kernel count.
**Rule:** a fused kernel is only kept if it is bit-exact or within a tight tolerance, *and* it beats both neighbouring controls.

## Phase 2 status (2026-11-21 evening)
| Step | Kernels/token | Server decode | Notes |
|---|---|---|---|
| upstream (all fusions off) | 7,349 | 80.0 | `srcunf` lever |
| + SITU op + clip fusion | 5,909 | ~85 | round 1 |
| + convolve head/tail | 5,365 | 89.6 | round 2 |
| + control gates + mix stacking/no-repeat | **4,726** | **93.0** | round 3 (+16.3%) |

Everything is bit-identical to unfused (greedy output and perplexity). Launch with `..\start_inkling_fused.ps1`.
**Switches (TEMP):**
- `LLAMA_INKLING_UNFUSED=1` (graph-level: SITU, mix)
- `GGML_CUDA_NO_INKLING_CLIP=1`, `GGML_CUDA_NO_INKLING_CONV=1`, `GGML_CUDA_NO_INKLING_GATE=1` (CUDA fusions)
- `INKLING_FUSE_DEBUG=1` (prints why the concat/state matcher rejects)

**Remaining ideas (not exact any more, or bigger effort):**
1. Fully fused `mix` kernel (~600 more kernels): the matmuls make it non-bit-exact, so it needs a KLD/tolerance gate.
2. Share `quantize_q8_1` across matvecs with the same input (449/token).
3. Dynamic-tap CONT+MUL+ADD (96/token).
4. Unit tests for the CUDA fusions in test-backend-ops (needed before upstreaming).
