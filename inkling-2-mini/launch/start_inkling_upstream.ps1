# ALTERNATIVE: upstream llama.cpp b12606 engine + self-requantized ggml-org Q3_K_M. Decode +9% (77.6 vs 71 tok/s),
# but long-prompt prefill -9..13% and no faster per coding task (writes ~16% more tokens). Good for chat-style use.
# Chat UI + OpenAI-compatible API at http://127.0.0.1:8080   (tuned 2026-11-21, see RESULTS.md and hillclimb\PLAN.md)
$env:CUDA_VISIBLE_DEVICES = '1'                     # GPU 1 only (GPU 0 is used by other work)
$d = 'D:\Models\Inki'
$serverArgs = @(
  '-m', "$d\gguf\ggml-org\Inkling-2-Mini-Q3_K_M-imat.gguf",
  '-ngl', '99',                                     # whole model on the GPU
  '-fa', 'on',                                      # flash attention
  '-c', '32768', '-ub', '256',                      # 32K context, small batch: the largest setup with ZERO spill into system RAM
  '-ctk', 'q8_0', '-ctv', 'q8_0',                   # q8 KV cache: half the memory, no measurable speed cost
  '--parallel', '1',                                # one chat at a time = fixed memory use
  '--load-mode', 'none',                            # don't keep a copy of the file in system RAM
  '--jinja',
  '-bs',                                            # backend sampling: pick the token on the GPU (+1-2%, hill-climb 2/2 wins)
  '--chat-template-kwargs', '{\"reasoning_effort\":\"low\"}',   # low effort = most accurate AND fastest for small coding tasks
  '--temp', '1.0', '--top-p', '1.0', '--top-k', '0', '--min-p', '0.0',   # Unsloth's sampling (temp 0 is ~3% faster, also fine for code)
  '--host', '127.0.0.1', '--port', '8080'
)
& "$d\llama-up\llama-server.exe" @serverArgs
