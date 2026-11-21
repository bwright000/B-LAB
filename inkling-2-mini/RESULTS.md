# Inkling 2 Mini: local coding model on one RTX 5070 Ti (GPU 1)

Updated 2026-11-21, after the overnight battery and a follow-up that used the inference-optimisation skill's method.

**To run it:** `powershell -File D:\Models\Inki\start_inkling.ps1`, then open http://127.0.0.1:8080. The OpenAI-compatible API is at `/v1/chat/completions`.

## Final config and numbers

UD-Q3_K_XL, fully on the GPU, flash attention, **32K context, `-ub 256`**, q8_0 KV cache, low reasoning effort by default.

| Metric | Value | How measured |
|---|---|---|
| Generation, short context | 70–77 tok/s | llama-bench, plus a median of 6 on a server panel |
| Generation at 25K depth | 65 tok/s | needle run |
| Prompt reading (cold, 25K tokens) | 970 tok/s (26 s) | needle run |
| Turn-2 wait on the same 25K document (prompt cache) | **0.46 s** | 24,936 of 25,159 tokens reused |
| Long-context retrieval | 3/3 needles at 12K and 25K | exact match |
| Coding accuracy (low effort) | HumanEval 98.8%, MBPP 90% | 164 + 100 problems |
| Stability | 64 tok/s flat for 45 min, 52 °C, 134 W | soak |
| Theoretical ceiling | ~208 tok/s, so we are at ~35% | 759 GB/s measured ÷ 3.65 GB read per token |

## The biggest finding: the old 64K config was spilling
At 64K context with `-ub 1024`, the GPU silently borrowed **12.4 GB of system RAM** (Windows Sysmem Fallback). Short benchmarks still looked fine, but real long prompts collapsed: 89–256 tok/s prompt reading, 37 tok/s generation and an 8.8 s turn-2 wait. Moving to 32K with `-ub 256` brought the borrowed memory to 0, and long-prompt speed rose 4–11x.
Check for spill with: `Get-Counter '\GPU Process Memory(*)\Shared Usage'`. About 440 MB per process is normal baseline; much more than that is spill.

## Why it can't go much faster with flags
- **Every quant runs at the same speed:** IQ3_XXS 77.6, Q3_K_XL 76.7 and IQ4_XS 76.1 tok/s, although they read 10.9–14.6 GB. If the limit were memory bandwidth, the smallest would be faster.
- **Turning CUDA graphs off drops speed from 77 to 16.6 tok/s.** That shows per-kernel launch overhead dominates: 48 layer passes and many small experts per token.
- **So the remaining ~65% gap to the ceiling is in llama.cpp's kernels for this model**, not in settings. It needs kernel fusion upstream, or a different engine.

## What was tested and rejected (with evidence)

| Lever | Result | Verdict |
|---|---|---|
| Medium / high reasoning effort | 2x slower per task, slightly *lower* accuracy | Low is best |
| `--reasoning-budget 0` (no thinking) | 34% HumanEval; the model loops until max_tokens | Never use |
| IQ4_XS (better perplexity: 5.72 vs 5.75) | Same speed, MBPP 93 vs 90, but only about 8K context fits | Optional for short tasks |
| IQ3_XXS | Same speed, slightly lower accuracy (97 / 88) | Fallback if you need 48K context (no spill) |
| Q4_K_M / Q4_K_XL with experts on CPU (0xSero's offload) | 55–69 tok/s | Slower; rejected |
| DSpark draft (speculative decoding) | Lost to the neighbouring controls in interleaved A/B runs | Rejected |
| MTP speculation (Tensorfold's suggestion) | Unsloth's GGUF has no MTP layers | Not possible |
| n-gram speculation | Slower | Rejected |
| Temperature 0 vs 1.0 | **+2–4% in all 5 paired runs**, same accuracy (40 problems) | Small real win, optional for code |
| 2 / 4 requests at once | 115 / 159 tok/s total (vs 69) | Useful for batch jobs, not chat |

## Hill-climb with the inference-optimisation skill (2026-11-21 midday)
The full log is in `hillclimb/PLAN.md`, the ceilings in `hillclimb/THEORY.md`, and the raw runs in `hillclimb/ledger.jsonl`. The method: the skill's `bench.py`, a median of 6, interleaved control/candidate/control, and a spill watchdog.
- **Accepted:** `-bs` (sample the next token on the GPU), +1-2%, 2 out of 2 wins. It's now in `start_inkling.ps1`.
- **Rejected (inside noise or slower):** `GGML_CUDA_PDL`, `GGML_CUDA_GRAPH_OPT`, `--prio/--poll`, `-t 1/2/4`, `GGML_CUDA_MM_IDS_SORT`. Kernel fusion is worth about 4% (diagnostic).
- **Upstream engine** (ggml-org b12606 plus a self-requantized Q3_K_M): **decode +9% (77.6 vs 71 tok/s)**, needle 3/3 and the same coding accuracy (250 vs 252 out of 264). **But** it writes about 16% more tokens per answer, so coding tasks take the same time, and long-prompt prefill is 9-13% slower. It's shipped as an option: `start_inkling_upstream.ps1`.
- **MTP:** neither Unsloth's nor ggml-org's GGUF contains the MTP layers (checked the tensor list), so it's a dead end in llama.cpp for now.
- **Plateau:** the remaining gap to the ~208 tok/s ceiling is inside the GPU kernels. CPU-side and launch-side levers measure as zero. The next real gains will come from upstream kernel work, or from a GGUF with MTP layers.

## Caveats
- Run-to-run drift was large (the same config measured 71 and then 50 tok/s, 5 minutes apart). It is likely caused by other work on GPU 0. Only paired or interleaved comparisons in this file are trustworthy.
- Prompt reading at depth is still well below the short-prompt 1,800 tok/s. That comes from attention cost, not spill.
- Untested: vision (`mmproj`), contexts above 32K without spill, and agent/tool-calling quality.

Raw data: `bench\ledger.jsonl` (follow-up), `bench\code_eval.md`, `bench\overnight_*.md`, `bench\*.log`. Scripts: `eval\eval_code.py`, `eval\followup.py`, `eval\ceiling.py`, `eval\bandwidth.py`.
