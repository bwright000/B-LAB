# Inkling 2 Mini on one RTX 5070 Ti (16 GB)

We ran Thinking Machines' Inkling 2 Mini (26B MoE: 128 experts, top-8, with layers 6–17 looped 3x) on a single consumer Blackwell card. Then we made it fast, and wired it into Claude Code as a local coding worker.

- **Hardware:** RTX 5070 Ti 16 GB, sm_120, on PCIe x4, Windows 11.
- **Quant:** Unsloth's UD-Q3_K_XL.
- **Engine:** llama.cpp.
- **Dates:** November 2026.

## Headline

| Stage | Decode tok/s, short context | What changed |
|---|---|---|
| Unsloth prebuilt llama.cpp | ~77 | Baseline; 32K context with `-ub 256` gives no VRAM spill |
| Fused CUDA kernels (`fusion/patches/0002`) | 105.5 (server), 111.8 (llama-bench) | Fewer tiny kernels per token in the looped layers |
| + DSpark speculative decoding, n_max=2 | **~139** at temperature 0 | The model's own drafter ([llama.cpp PR #32218](https://github.com/ggml-org/llama.cpp/pull/32218)) plus `0003` (one LM-head matmul per draft token) |
| 64K-context variant | 110 short, 77 at 46K depth | `--n-cpu-moe 2` |
| Strata engine + SQ-10 pack (option, 2026-11-23) | 137 chat / 180 code at temperature 0 (ours on the same prompts: 113 / 136) | A different engine and quant; see [`strata/`](strata/) |

- **Overall:** coding tasks run about 40% faster end to end, with the same HumanEval and MBPP accuracy.
- **Quality checks:**
  - Fusion KLD is about 0.003, the same as the model's own rounding floor (with fusion switched off it measures 0.0029).
  - DSpark is lossless; every greedy divergence we checked was a near-tie.
- **Ceiling:** about 3.9 ms per token (weights at the measured 759 GB/s take ~3.1 ms). The rest of the gap is launch overhead from tiny kernels.

## Engine comparison: Strata (2026-11-23)

We ran the [Strata](https://github.com/Niko1221/Strata) engine with its SQ-10 pack against the fused build on the same card and prompts. Details and scripts are in [`strata/`](strata/).

- **Long prompts:** Strata reads a 29.7K-token prompt at 2,770 tok/s against 866, and decodes at 130 tok/s at that depth against 90.
- **64K context:** it fits fully in VRAM with 8-bit KV (13.6 GB), reading at 2,203 tok/s.
- **Small coding tasks with low-effort thinking:** no gain. HumanEval 96.3% and MBPP 91% in 18.8 min, against 97.0% and 92% in 19.0 min.
- **Caveat:** stock Strata turns Inkling's thinking off, which is faster per task but drops HumanEval to 88.4%.

## What's here

| Path | Contents |
|---|---|
| [`LESSONS.md`](LESSONS.md) | Short lessons for next time: runtime-estimate mistakes, silent VRAM spill on Windows, environment gotchas. **Read this first.** |
| [`RESULTS.md`](RESULTS.md) | First-phase results before fusion: quant comparison, the 64K spill discovery, rejected levers |
| [`hillclimb/`](hillclimb/) | Hill-climb log (`PLAN.md`), ceiling maths (`THEORY.md`), every run (`ledger.jsonl`), queue and launch scripts |
| [`fusion/`](fusion/) | Fused-kernel work for llama.cpp: plan (`README.md`, `PHASE1.md`), patches, build script, verification scripts, nsys kernel summaries |
| [`eval/`](eval/) | Coding-accuracy harness (HumanEval and MBPP), ceiling and bandwidth calculators, per-config results |
| [`strata/`](strata/) | Engine comparison against Strata: token-level A/B harness, needle and tool-call tests, server wrapper, per-prompt results |
| [`launch/`](launch/) | Server launch scripts. `start_inkling_fast.ps1` is the recommended one: fused build, DSpark n=2, 32K. `start_inkling_strata.ps1` is the Strata option for long prompts and 64K |
| [`claude-code/`](claude-code/) | Using the local model from Claude Code (VS Code), and `inkling.ps1`, which lets a lead Claude agent delegate small tasks to Inkling with a chosen toolbox |

## Reproducing

1. Check out llama.cpp at `e412c6f46` (the base we used) and apply PR #32218.
2. Then `git am fusion/patches/0002-*.patch fusion/patches/0003-*.patch`.
3. Build with CUDA for `sm_120` (`fusion/build.cmd`).
4. GGUFs: Unsloth's Inkling 2 Mini UD-Q3_K_XL, and the DSpark drafter at Q8_0.
5. The scripts use absolute Windows paths (`D:\Models\Inki\...`). Adjust them to your layout.

The patches are experimental and not upstreamed. Runtime switches for A/B testing (`LLAMA_INKLING_UNFUSED`, `GGML_CUDA_NO_INKLING_*`) are described in `fusion/README.md`.

## Caveats

- These are single-machine numbers, and the other GPU was sometimes busy. Only the interleaved control/candidate/control comparisons are trustworthy.
- Anthropic does not support Claude Code with non-Claude models. It works here, but some features may misbehave.
