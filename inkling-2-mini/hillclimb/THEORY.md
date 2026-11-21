# Ceilings: Inkling 2 Mini UD-Q3_K_XL on RTX 5070 Ti (GPU 1)

## Measured hardware (`eval/bandwidth.py`, median of 10)
- Copy bandwidth: **759 GB/s** (the spec sheet says 896).
- FP16 GEMM: 95–97 TFLOPS.
- PCIe link: **Gen5 x4** (max x16). The second slot is likely chipset-wired. Spill and host transfers are about 4x slower than on an x16 slot.
- CPU: Ryzen 9 9950X. Power plan: "3XS-Overclock" (a performance profile, left unchanged).

## Decode (memory bound), from real GGUF tensor sizes (`eval/ceiling.py`)

| Component | GB per token |
|---|---:|
| Dense (attention, norms, router; the 12 looped layers count 3x) | 1.724 |
| Routed experts (8/128 of each layer, looped layers 3x) | 1.607 |
| LM head (201K vocab) | 0.318 |
| KV at 32K (8 global-attention passes x 16 KiB/token, q8_0) | ~0.03 (negligible) |
| **Total** | **3.65** |

**Decode ceiling: 759 / 3.65 ≈ 208 tok/s.** The baseline of about 70 tok/s is about 34% of it.

## Evidence that the bottleneck is per-token overhead, not bandwidth
- IQ3_XXS (10.9 GB), Q3_K_XL (12.7 GB) and IQ4_XS (14.6 GB) all decode at 76–78 tok/s.
- Turning CUDA graphs off takes it from 77 to 16.6 tok/s, so launch overhead is dominant.
- Implied step time is 13 ms at 77 tok/s, against 4.8 ms of pure memory traffic. That leaves about 8 ms of overhead or kernel inefficiency per token.
- So the levers to try are the ones that cut per-step overhead: sampling on the GPU, PDL, the graph optimiser, CPU scheduling, and newer kernels.

## Prefill (compute bound)
- About 2 x 5.4B effective active params ≈ 10.8 GFLOP per token, plus attention.
- At 50% of 96 TFLOPS, that gives **≈4,400 tok/s** at short context. Measured: 1,800 (pp512) and about 1,000 at 12–25K depth (ub 256).

## KV
- Global attention: 8 passes x 2 x 8 heads x 128 dim x 1 byte (q8_0) = 16 KiB per token, so 0.5 GiB at 32K.
- Sliding window (1,024 tokens): 40 passes, about 80 MiB, fixed.
- What limits context on 16 GB is the compute buffer (`-ub`) and the weights, not the KV.

## Where the time actually goes (Nsight Systems, 64-token decode, `profile/tg64*.nsys-rep`)
The profiler adds about 0.4 us per kernel (16.1 ms/token under profiling vs 13 ms real). Real-time estimates:

| Category | ms/token (real, approx.) | kernels/token |
|---|---|---|
| Weight matvecs (experts + dense) | ~5.4 (about 680 GB/s, ~90% of measured bandwidth) | ~385 |
| Small fp32 matvecs (loop gates, conv, attention-residual) | ~1.6 | ~640 |
| Attention | ~0.3 | ~96 |
| **Glue ops** (add/mul/scale/copy/concat/norm/quantize...) | **~5.8** | **~6,270** |
| Total | ~13 | **~7,400** |

**The gap to the ceiling (4.8 ms) is about 7,000 tiny kernels at about 1 us each**, not memory. Weight reads are already near peak.
Fusing Inkling's custom ops (softcaps, attention-residual, loop gates, sconv, clipping) would be the lever. Realistic: 130-170 tok/s. That's engine work.

**Upstream b12606 profile:** 7,349 kernels/token, the same as Unsloth. Its +9% comes from weight kernels about 0.9 ms faster (Q3_K_M compresses the dense/attention tensors more than Unsloth's UD mix, which keeps Q8_0 dense). That's a quant choice, not more fusion.

**MTP = DSpark:** `mtp_config` is the 3-layer DSpark drafter. The llama.cpp converter skips the `mtp.` tensors by design, and the drafter ships separately (Unsloth's dspark GGUF). Upstream DSpark for inkling2 is PR #32218 (still open). DSpark gave no gain here despite about 2.5 tokens accepted per step. Hypothesis: the drafter's own kernel overhead (also Inkling-style layers) eats the win. Profile it next to confirm.
