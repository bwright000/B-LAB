# Correctness checks for the inkling-fusion build: fused vs unfused in the SAME binary (env switches), GPU 1 only.
$ErrorActionPreference = 'Continue'
$env:CUDA_VISIBLE_DEVICES = '1'
$bin   = 'D:\Models\Inki\src\llama.cpp\build\bin'
$model = 'D:\Models\Inki\gguf\ggml-org\Inkling-2-Mini-Q3_K_M-imat.gguf'
$o     = 'D:\Models\Inki\src\analysis'
$unfusedVars = 'LLAMA_INKLING_UNFUSED', 'GGML_CUDA_NO_INKLING_CLIP', 'GGML_CUDA_NO_INKLING_CONV', 'GGML_CUDA_NO_INKLING_GATE'
function Set-Unfused($on) { foreach ($v in $unfusedVars) { if ($on) { Set-Item "env:$v" '1' } else { Remove-Item "env:$v" -ErrorAction SilentlyContinue } } }

"== 1. test-backend-ops (SITU vs CPU reference) =="
& "$bin\test-backend-ops.exe" test -o SITU -b CUDA0 2>&1 | Select-String 'tests passed|FAIL' | Select-Object -Last 3

"== 2a. greedy decode equivalence (server, temp 0, 3 prompts x 256 tokens) =="
$prompts = @('Write a Python function that merges overlapping intervals, with tests.',
             'Explain how a hash map handles collisions, briefly.',
             'Write a bash one-liner that counts lines in all .py files recursively.')
$res = @{}
foreach ($mode in 'unfused', 'fused') {
  Set-Unfused ($mode -eq 'unfused')
  $p = Start-Process "$bin\llama-server.exe" -ArgumentList @('-m',$model,'-ngl','99','-fa','on','-c','8192','--jinja','--port','8093','--load-mode','none') -PassThru -WindowStyle Hidden -RedirectStandardError "$o\verify_$mode.err"
  for ($i=0; $i -lt 90; $i++) { Start-Sleep 2; try { if ((Invoke-RestMethod http://127.0.0.1:8093/health).status -eq 'ok') { break } } catch {} }
  $res[$mode] = foreach ($pr in $prompts) {
    $body = @{ messages=@(@{role='user';content=$pr}); max_tokens=256; temperature=0.0; seed=1; chat_template_kwargs=@{reasoning_effort='low'} } | ConvertTo-Json -Depth 5
    $r = Invoke-RestMethod http://127.0.0.1:8093/v1/chat/completions -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 300
    ($r.choices[0].message.reasoning_content + '|' + $r.choices[0].message.content)
  }
  Stop-Process -Id $p.Id -Force; Start-Sleep 4
}
for ($k=0; $k -lt $prompts.Count; $k++) {
  $same = $res['unfused'][$k] -ceq $res['fused'][$k]
  "prompt {0}: {1}  (len {2} vs {3})" -f ($k+1), $(if ($same) {'IDENTICAL'} else {'DIFFERENT'}), $res['unfused'][$k].Length, $res['fused'][$k].Length
}

"== 2b. perplexity equivalence (batched prefill path, 20 chunks x 512) =="
foreach ($mode in 'unfused', 'fused') {
  Set-Unfused ($mode -eq 'unfused')
  $ppl = & "$bin\llama-perplexity.exe" -m $model -f D:\Models\Inki\eval\data\wiki.test.raw -c 512 --chunks 20 -ngl 99 -fa on --load-mode none 2>&1 | Select-String 'Final estimate'
  "{0,-8} {1}" -f $mode, ($ppl.Line -replace '.*Final estimate: ','')
}
Set-Unfused $false
