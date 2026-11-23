# OPTION (2026-11-23): Strata engine (strata-inkling 0.1.66, release 0.2.20) + its SQ-10 pack, 64K context, low-effort thinking.
# Everything lives in D:\Models\Inki\strata (engine\, control\, packs\); nothing is installed system-wide.
# vs start_inkling_fast.ps1 (fused llama.cpp + DSpark, 32K), measured on GPU 1 with identical prompts (hillclimb\PLAN.md, Phase 4):
#   prompt reading 2,770 tok/s at 30K (vs 866), 2,200 tok/s at 59K; decode 130 tok/s at 30K depth (vs 90); 64K fits fully in VRAM (13.6 GB).
#   short-context decode, temp 0: chat 135 / code 172 tok/s (vs 113 / 136); temp 1.0: 99 / 140 (vs 93 / 120).
#   coding suite with low-effort thinking: HumanEval 96.3% / MBPP 91% in 18.8 min (vs 97.0% / 92% in 19.0 min): same accuracy, same wall time.
# Notes:
#   - serve_think.py keeps low-effort thinking on. Stock Strata forces thinking OFF for Inkling (5x faster per task, but 88.4% / 87%).
#   - --kv k8v8 (8-bit KV) is required for 64K: the default fp16 KV runs out of VRAM at ~49K tokens and kills the engine.
#   - SQ-15 and the Unsloth GGUFs do NOT fit 16 GB in this engine (experts go to host RAM: 55-81 tok/s). Use SQ-10 only.
#   - NOT yet tested end to end with Claude Code (only: /v1/messages tool_use round trip, 30/30 Windows paths intact).
# OpenAI API at /v1/chat/completions, Anthropic API at /v1/messages.   -Lazy: bind the API but load the model on the first request.
param([int]$Port = 8080, [switch]$Lazy)
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$d = 'D:\Models\Inki\strata'
$env:STRATA_HOME = "$d\home"
$serverArgs = @('-I', "$d\serve_think.py", '--engine', 'strata', '--config', "$d\sq10_64k.json",
                '--host', '127.0.0.1', '--port', "$Port", '--gpu', '1', '--no-api-key')
if ($Lazy) { $serverArgs += '--lazy' }
& "$d\control\python\python.exe" @serverArgs
