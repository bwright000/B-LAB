# FASTEST: our llama.cpp build (branch inkling-fusion: fused kernels + PR #32218 Inkling 2 DSpark, merged drafter head)
# + ggml-org Q3_K_M + Unsloth DSpark drafter (speculative decoding, 2 draft tokens/step).
# Coding benchmark: HumanEval 97.0% / MBPP 92% (same as no-spec), 4.7 s / 3.7 s per task vs 7.7 / 6.2 unfused.
# 32K context: q8_0 KV, -ub 128 (main compute buffer was what spilled at ub 256), drafter KV q8_0 + draft ubatch 16.
# Verified: no spill with a 25K-token prompt, needle 3/3, 875 tok/s prompt reading, 108 tok/s generation at 25K depth.
# Keep GPU 1 free of other jobs: only ~0.9 GB of headroom. Chat UI + OpenAI API at http://127.0.0.1:8080
$env:CUDA_VISIBLE_DEVICES = '1'
$d = 'D:\Models\Inki'
$serverArgs = @(
  '-m', "$d\gguf\ggml-org\Inkling-2-Mini-Q3_K_M-imat.gguf",
  '-ngl', '99', '-fa', 'on',
  '-c', '32768', '-ub', '128',                      # 32K context; ub 128 keeps the compute buffer small enough (ub 256 spills)
  '-ctk', 'q8_0', '-ctv', 'q8_0',                   # q8_0 KV: half the memory of f16
  '--parallel', '1', '--load-mode', 'none', '--jinja',
  '-bs',                                            # sample on the GPU
  '--chat-template-kwargs', '{\"reasoning_effort\":\"low\"}',
  '--temp', '1.0', '--top-p', '1.0', '--top-k', '0', '--min-p', '0.0',
  '-md', "$d\gguf\dspark-Inkling-2-Mini-Q8_0.gguf", # DSpark = Inkling's MTP drafter
  '--spec-type', 'draft-dspark', '-ngld', 'all',
  '--spec-draft-p-min', '0',                        # required by the Inkling2 DSpark path (no confidence head)
  '--spec-draft-n-max', '2',                        # 2 is optimal: more drafts grow the MoE expert union in verify
  '-ctkd', 'q8_0', '-ctvd', 'q8_0', '--spec-draft-n-ubatch', '16',   # drafter KV in q8_0, small drafter batch (>= block 8)
  '--host', '127.0.0.1', '--port', '8080'
)
& "$d\src\llama.cpp\build\bin\llama-server.exe" @serverArgs
