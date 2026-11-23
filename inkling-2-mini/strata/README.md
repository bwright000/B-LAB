# Engine comparison: Strata vs our fused llama.cpp build (2026-11-23)

We tested the [Strata](https://github.com/Niko1221/Strata) engine (`strata-inkling` 0.1.66, release 0.2.20) against our fused llama.cpp build on the same RTX 5070 Ti (16 GB, GPU 1). Both engines got the same prompts, and the llama.cpp control was measured before and after the Strata runs.

**Short version:** Strata is much faster on long prompts and fits 64K context in VRAM. For small coding tasks with thinking on, the two engines take the same time and score the same.

## Results

Control: `launch/start_inkling_fast.ps1` (fused kernels, DSpark n=2, Q3_K_M-imat, 32K context). Strata: its own SQ-10 pack (10.5 GB) with the built-in drafter.

| Metric | Fused llama.cpp | Strata SQ-10 | How measured |
|---|---|---|---|
| Decode, temp 0 (chat / code) | 113 / 136 tok/s | 137 / 180 | `ab.py`: identical prompt token ids, median of 6 chat and 4 code prompts, 400 tokens |
| Decode, temp 1.0 (chat / code) | 93 / 120 | 99 / 139 | same panel |
| Decode, drafter off | ~105 | 88 | Strata's plain decode is slower than ours |
| Prompt reading, 19.6K / 29.7K tokens | 779 / 866 tok/s | 2,972 / 2,770 | `needle.py`, same documents |
| Decode at 29.7K depth | 90 tok/s | 130 | same |
| Whole turn on a 29.7K prompt | 38.2 s | 12.8 s | same; 3/3 needles on both |
| 64K context | experts on CPU, ~250 tok/s reading | in VRAM (13.6 GB); 2,203 tok/s at 58.6K, 110 tok/s decode | `--kv k8v8`; 3/3 needles, no spill |
| Coding suite, low-effort thinking | 97.0% / 92%, 19.0 min | 96.3% / 91%, 18.8 min | HumanEval 164 + MBPP 100, temp 1.0; minutes are summed request time |
| Coding suite, no thinking | 86.6% / 81%, 4.4 min | 88.4% / 87%, 3.4 min | same suite |

## What we learned

- **The gain is in speculative decoding.** Strata's drafter step gives 2.0x over its own plain decode; ours gives about 1.3x. Strata drafts a small tree, sizes it each step from the drafter's own probabilities, and verifies it in one CUDA graph. At temperature above 0 it falls back to a linear draft, so the gain is smaller.
- **Only SQ-10 fits a 16 GB card.** SQ-15 puts 340 of 3,072 experts in system RAM (59 / 81 tok/s). Unsloth's UD-Q3_K_XL GGUF puts 1,013 there (55 tok/s plain), and Strata rejects Unsloth's DSpark drafter GGUF.
- **SQ-10 is accurate enough for coding.** With thinking on it passes 249 of 264 tasks; our Q3 quant passes 251.
- **Stock Strata turns Inkling's thinking off.** Its server renders the chat template with no reasoning effort and starts the reply directly in visible text. That is about 5x faster per task, but it costs 9 to 10 points on HumanEval on either engine. `serve_think.py` puts low-effort thinking back.
- **With thinking on, faster decode did not shorten coding tasks.** Strata generated 8 to 10% faster but wrote more tokens.
- **64K needs 8-bit KV.** With the default fp16 KV the engine stopped with "VRAM exhausted" at about 49K tokens.
- **Tool calls work.** An Anthropic `/v1/messages` tool_use round trip succeeded, and 30 of 30 Windows paths came back intact.

## Not verified

- Claude Code end to end on Strata.
- Vision, a long soak, and concurrent requests.
- SQ-10's divergence from the original weights on our own text. The vendor reports KLD 0.050 / 0.072.

## Files

| File | Purpose |
|---|---|
| `ab.py` | Token-level A/B. `prep` tokenizes the panel through llama.cpp, `llama` benchmarks `/completion`, `strata` drives `strata-inkling --serve` over stdin |
| `prompts.json` | The 10 templated prompts and their token ids |
| `results/*.jsonl` | One row per prompt and run, with the generated token ids |
| `needle.py` | Needle retrieval plus a turn-2 cache test against any OpenAI-compatible server |
| `toolpath.py` | Checks that Windows paths survive an Anthropic-API tool call |
| `serve.ps1` | Starts and stops Strata's HTTP server (detached, GPU 1); `-Think` uses `serve_think.py` |
| `serve_think.py` | Runs Strata's `server.py` with low-effort thinking instead of forced no-thinking |
| `sq10.json`, `sq10_64k.json` | Run configs: 32K with fp16 KV, 64K with `--kv k8v8` |
| `eval_code_nothink_llama.py` | The coding eval against llama.cpp with Strata-style no-thinking prompts |
| `bench_strata.json` | Config for the hill-climb `bench.py` (prefill numbers only; see the caveat below) |

Coding-suite rows are in `../eval/results/` (`strata_sq10_low`, `strata_sq10_nothink`, `llama_fused_nothink`). The full log is Phase 4 of `../hillclimb/PLAN.md`. The launch script is `../launch/start_inkling_strata.ps1`.

## Reproducing

1. Download from the Strata v0.2.20 release: `strata-inkling-0.1.66-windows-x86_64-cuda-cuda12.tar.gz` (engine) and `strata-control-0.1.66-windows-x86_64.zip` (HTTP server with its own Python). Extract them to `engine\` and `control\` beside these scripts. No installer is needed.
2. Download `Inkling-2-Mini-SQ-10.strata`, `tokenizer.json` and `chat_template.jinja` from `strata-engine/Inkling-2-Mini-Strata` into `packs\`.
3. Build the tokenizer folder with `control\tools\inkling_tokenizer.py` (`convert_files(tokenizer.json, chat_template.jinja, tok\sq10)`).
4. Adjust the absolute paths in `sq10*.json` and the scripts (`D:\Models\Inki\strata\...`).
5. Engine-level A/B: with llama.cpp running, `python ab.py prep`, then `python ab.py llama --label ctl`. Stop llama.cpp, then `python ab.py strata --label sq10 -- --native packs\Inkling-2-Mini-SQ-10.strata --gpu 0 --max-context 32768 --draft auto`.
6. HTTP server: `serve.ps1 -Config sq10_64k.json -Think`, then run `needle.py 8092 75000` or the coding eval with `--port 8092`.

## Measurement caveats

- The hill-climb `bench.py` decode figure is wrong against Strata in thinking mode. Thinking tokens are not streamed, so timing from the first visible token gave 112 to 584 tok/s. Use `ab.py` for decode.
- Its long-context needle reported 0/3 because the 256-token limit was spent on hidden thinking. With 1,024 tokens it is 3/3.
- Single machine, single session. The decode comparison has controls on both sides; the long-context and coding numbers are single runs.
