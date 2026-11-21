# Regression checks for the fused build that verify_fusion.ps1 does not cover. GPU 1 only.
$env:CUDA_VISIBLE_DEVICES = '1'
$bin = 'D:\Models\Inki\src\llama.cpp\build\bin'; $m = 'D:\Models\Inki\gguf\ggml-org\Inkling-2-Mini-Q3_K_M-imat.gguf'
$py  = 'D:\Models\Inki\eval\.venv\Scripts\python.exe'; $o = 'D:\Models\Inki\src\analysis'
$vars = 'LLAMA_INKLING_UNFUSED', 'GGML_CUDA_NO_INKLING_CLIP', 'GGML_CUDA_NO_INKLING_CONV', 'GGML_CUDA_NO_INKLING_GATE'
function Set-Unfused($on) { foreach ($v in $vars) { if ($on) { Set-Item "env:$v" '1' } else { Remove-Item "env:$v" -ErrorAction SilentlyContinue } } }
function Shared($procId) { $c = (Get-Counter '\GPU Process Memory(*)\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples | Where-Object { $_.InstanceName -match "pid_$($procId)_" }; [math]::Round((($c | Measure-Object CookedValue -Sum).Sum)/1MB) }
function Start-Srv([string[]]$extra) {
  $a = @('-m',$m,'-ngl','99','-fa','on','-c','32768','-ub','256','-ctk','q8_0','-ctv','q8_0','--load-mode','none','--jinja','--port','8091') + $extra
  $p = Start-Process "$bin\llama-server.exe" -ArgumentList $a -PassThru -WindowStyle Hidden -RedirectStandardError "$o\regr.err"
  for ($i=0; $i -lt 90; $i++) { Start-Sleep 2; try { if ((Invoke-RestMethod http://127.0.0.1:8091/health).status -eq 'ok') { return $p } } catch {} }
  throw 'server failed'
}

"== 1. long context: needle at ~25K, decode at depth, VRAM spill (production config, 32K ctx) =="
foreach ($mode in 'unfused', 'fused') {
  Set-Unfused ($mode -eq 'unfused'); $p = Start-Srv @('--parallel','1')
  $peak = 0; $job = Start-Job { param($py) & $py -I D:\Models\Inki\eval\followup.py needle --ctx 32768 } -ArgumentList $py
  while ($job.State -eq 'Running') { Start-Sleep 3; $peak = [math]::Max($peak, (Shared $p.Id)) }
  $r = Receive-Job $job | Select-Object -Last 1
  $vram = nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader
  "{0,-8} VRAM {1} | shared peak {2} MB (~440 = baseline) | {3}" -f $mode, $vram, $peak, ($r -replace "'t': '[\d:]+', 'label': 'needle_32768', ", '')
  Stop-Process -Id $p.Id -Force; Start-Sleep 4
}

"== 2. two concurrent requests (--parallel 2, multi-sequence code paths), greedy, fused vs unfused =="
$prompts = @('Write a Python function that merges overlapping intervals, with tests.', 'Explain how a hash map handles collisions, briefly.')
$out = @{}
foreach ($mode in 'unfused', 'fused') {
  Set-Unfused ($mode -eq 'unfused'); $p = Start-Srv @('--parallel','2','-c','32768')
  $jobs = foreach ($pr in $prompts) { Start-Job { param($pr) $b = @{ messages=@(@{role='user';content=$pr}); max_tokens=200; temperature=0.0; chat_template_kwargs=@{reasoning_effort='low'} } | ConvertTo-Json -Depth 5; $r = Invoke-RestMethod http://127.0.0.1:8091/v1/chat/completions -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 300; $r.choices[0].message.reasoning_content + '|' + $r.choices[0].message.content } -ArgumentList $pr }
  $out[$mode] = $jobs | Wait-Job | Receive-Job
  Stop-Process -Id $p.Id -Force; Start-Sleep 4
}
for ($k=0; $k -lt 2; $k++) { "request {0}: {1}" -f ($k+1), $(if ($out['unfused'][$k] -ceq $out['fused'][$k]) {'IDENTICAL'} else {"DIFFERENT (len $($out['unfused'][$k].Length) vs $($out['fused'][$k].Length))"}) }

"== 3. speed deep into context (llama-bench, depth 0/8K/16K) =="
foreach ($mode in 'unfused', 'fused') {
  Set-Unfused ($mode -eq 'unfused')
  "-- $mode"; & "$bin\llama-bench.exe" -m $m -ngl 99 -fa 1 -lm none -ub 256 -ctk q8_0 -ctv q8_0 -d 0,8192,16384 -p 512 -n 64 -r 2 2>$null | Select-String 'pp512|tg64' | ForEach-Object { ($_.Line -replace '.*\|\s+(pp512|tg64)', '$1') -replace '\s{2,}', ' ' }
}
Set-Unfused $false
