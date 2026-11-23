# Lessons for future Claude sessions: local LLM benchmarking on this machine

Written 2026-11-21 after the Inkling 2 Mini runs (`Inki/`). Read this before you plan a benchmark or estimate its runtime.

## Time estimates were wrong in both directions
- **Run 1 ("overnight"): finished in about 15 min.** llama-bench is fast; a sweep takes minutes, not hours.
- **Run 2 (estimated 7–8 h): took about 12 h.** The causes, worst first:
  1. **One config spilled out of VRAM, so a 2-min test took 2 h.** Q4_K_M perplexity ran with `-ngl 99` and no `--n-cpu-moe`. On Windows the driver silently pages the overflow to system RAM (Sysmem Fallback), so nothing errors, it just slows down 30–60x.
  2. **A pathological config ran a full eval for 3.4 h.** `--reasoning-budget 0` made the model loop until it hit max_tokens (4096) on most tasks, which pushed it to about 45 s per task instead of 7.
  3. **The arithmetic was missing.** 264 tasks x 6 runs at 7–15 s each already comes to about 5 h before the other phases.

## Rules
- **Estimate per run:** tasks x worst-case seconds per task, where the worst case is max_tokens / tok/s (4096 / 65 ≈ 63 s). Add 20% and state the figure.
- **Check for VRAM fit before each config.** After load, VRAM above about 15.5 GB on a 16 GB card means you are probably spilling. Run a 1-sample speed check, and if it is more than 30% below the known baseline, abort the config. Never assume `-ngl 99` fits.
- **llama-bench uses more VRAM than llama-server at the same context.** `-d 61440` spilled although the server fits 64K. Cap llama-bench depth at about 32K on 16 GB. Spill signature: VRAM full, 100% utilisation, low power draw (about 70 W).
- **Put a wall-clock timeout on every phase** (for example 2x the estimate). Kill the phase, log it and move on, so one bad config can't eat the night.
- **Canary-test pathological settings first.** Run 10 tasks before the full eval. If more than 20% are truncated, skip it.
- **Order the battery by value** (as done here), so overruns only cost low-priority tests.
- **Give an ETA range and the riskiest item**, not one number.

## Measure the real workload, and check for spill directly (2026-11-21)
- **I shipped a config that spilled 12.4 GB into system RAM** and didn't notice: 64K context, ub 1024. The "context fits" test only measured short replies, and those stayed fast. Long prompts ran 4–11x slower.
- **Spill check:** `Get-Counter '\GPU Process Memory(*)\Shared Usage'`, filtered by the server PID. About 440 MB is baseline. Check it *during* a long-prompt request, not just after load.
- **`-ub` (batch size) changes VRAM a lot:** at 32K, ub 512 spilled 1.8 GB and ub 256 spilled nothing.
- **Compute the ceiling first** (`Inki/eval/ceiling.py` + `bandwidth.py`). Then, if all quants run at the same speed, you're overhead-bound, and flags won't help much.
- **Drift here is huge** (71 → 50 tok/s in 5 minutes, same config). Use paired comparisons inside one server session, or interleaved control–candidate–control runs, never single before/after runs.
- **PowerShell gotchas:** a function's output includes any `Tee-Object` or echo, so `$p = Start-Srv ...` was truthy even when the server failed. Log with `Add-Content`. Also, `-not $job | Wait-Job` binds wrongly; write `-not (Wait-Job $job ...)`.

## Hill-climb lessons (2026-11-21, inference-optimisation skill)
- **Repeat every win.** t4 gave +3.2% once and then lost twice, and prio gave +1.5% once and then lost. Only `-bs` survived. Controls drifted ±1.5% within 30 minutes even on a quiet box.
- **tok/s is not time per task.** The upstream engine was +9% on tok/s, but it wrote 16% more tokens, so coding tasks took the same time. Always gate on the real task suite, not just bench.py.
- **Engine and file format go together.** Upstream llama.cpp and Unsloth's prebuilt can't load each other's inkling2 GGUFs. To compare engines, requantize the other line's Q8_0 to the same bpw (llama-quantize with `--imatrix`, about 2 min).
- **Check a GGUF for MTP layers without downloading it:** `curl -r 0-20000000` and grep the tensor names (the full tensor list sits in the first ~12 MB).
- **Windows harness:** launch servers via WMI (`Win32_Process.Create`) so they don't inherit the bench's stdout pipe. With `Start-Process`, bench.py hangs forever. Hide the console (`ShowWindow=0`) or the user may Ctrl+C it. PS 5.1 writes a BOM with `-Encoding utf8` (Python's json chokes on it), and `Add-Content` without `-Encoding` writes ANSI. Use `[IO.File]::WriteAllText(..., UTF8Encoding($false))`.
- **Don't run CPU-heavy jobs (quantizing) during benchmarks.** The model is overhead-bound, so CPU load skews decode.

## Engine-comparison lessons (2026-11-23, Strata)
- **Compare engines on identical prompt token ids**, not on each server's chat endpoint. Tokenize once, then feed the ids to both (`strata/ab.py`). Chat endpoints differ in template, thinking mode and sampling.
- **Check what the server does to thinking.** Stock Strata forces Inkling's thinking off. It looked 5x faster per task, and HumanEval fell from 96% to 88%. Run the coding suite before believing a per-task speedup.
- **A vendor's per-GPU number belongs to one pack.** Strata's 155 tok/s row for this card is the 10.5 GB SQ-10 pack. The 14.3 GB SQ-15 pack "for 16 GB cards" put 340 experts in system RAM here and ran at 59 to 81 tok/s. Read the load line for how many experts are in VRAM.
- **Streaming-based tok/s breaks when thinking is hidden.** bench.py times from the first visible token, so a server that does not stream thinking reported 112 to 584 tok/s. Use the engine's own decode timing.
- **Give needle tests enough output tokens.** 256 tokens were spent on hidden thinking and the test read 0/3; with 1,024 it was 3/3.
- **Test long context at the real length.** Strata's default KV loaded fine at 64K and then died at about 49K tokens. `--kv k8v8` fixed it.
- **Check test inputs before reporting a bug.** A "mangled Windows path" in a tool call was the shell collapsing `\\` in a heredoc, so the request contained a tab. Build JSON test bodies in Python.
- **Stopping a server this session did not start needs the user's OK.** Ask early; the GPU work is blocked until then.

## Gotchas in this environment
- Unsloth llama.cpp zips: use **cuda12-newer**. The cuda13 zips load only the CPU backend, because only the CUDA 12.8 runtime is installed.
- Build b12568 renamed `--no-mmap` to `--load-mode none` (`-lm none` in llama-bench).
- PowerShell: `$args` is reserved (don't assign to it), and you can't put a comment after a line-continuation backtick. Use an argument array.
- Kill processes **by PID you started**, never by name. Killing by name is blocked, and it could kill another session's server.
- A results file is locked while llama-bench writes to it. Wait, or write to per-phase files.
- Check that downloads are non-empty. A wrong HF URL produces a 0-byte file and no error (wikitext).
- GPU 0 is often used by another session. Use `CUDA_VISIBLE_DEVICES=1`, and coordinate through SendMessage.

## Model-specific findings (Inkling 2 Mini, small coding tasks)
- Use **low reasoning effort**: it is the most accurate and fastest. Medium/high take twice as long and score slightly worse.
- **Never** use `--reasoning-budget 0`: it loops.
- On 16 GB: Q3_K_XL fits **32K context with ub 256** with no spill, at about 70 tok/s (64K looked fine in short tests but was spilling). IQ3_XXS fits 48K. IQ4_XS is marginally better but only fits about 8K. Q4 doesn't fit.
