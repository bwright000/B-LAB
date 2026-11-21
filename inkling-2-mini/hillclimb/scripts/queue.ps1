# Windows stand-in for the skill's queue.sh: one GPU job at a time, interleaved controls.
#   queue.ps1 ctl bs ctl pdl ctl      -> each run gets a unique name (ctl-1, bs-1, ctl-2 ...), config written to configs/
param([Parameter(ValueFromRemainingArguments)][string[]]$Runs)
$h = Split-Path -Parent $MyInvocation.MyCommand.Path; Set-Location $h
$lock = "$h\queue.lock"; if (Test-Path $lock) { "queue already running ($lock)"; exit 1 }; Set-Content $lock $PID
try {
  $L = Get-Content "$h\levers.json" -Raw | ConvertFrom-Json
  $py = 'D:\Models\Inki\eval\.venv\Scripts\python.exe'; $bench = "$env:USERPROFILE\.claude\skills\inference-optimisation\scripts\bench.py"
  $ps = @('powershell','-NoProfile','-ExecutionPolicy','Bypass','-File')
  foreach ($r in $Runs) {
    $n = 1; while (Test-Path "$h\configs\$r-$n.json") { $n++ }; $name = "$r-$n"; $lv = $L.$r
    $cfg = [ordered]@{
      name = $name; base_url = 'http://127.0.0.1:8091'; model = 'inkling'; health_timeout_s = 240
      launch = $ps + @("$h\launch.ps1", '-Config', "$h\configs\$name.json"); stop = $ps + @("$h\stop.ps1")
      concurrency = 1; prefill_lengths = @(4096, 16384); temperature = 1.0
      memory = [ordered]@{ cap_gib = 1.0; probe = $ps + @("$h\probe.ps1"); interval_s = 3 }
      server_args = @($L._base_args) + @($lv.add | Where-Object { $_ }); env = $lv.env; binary = $lv.binary; why = $lv.why
    }
    if ($lv.model) { $sa = [string[]]$cfg.server_args; $sa[[array]::IndexOf($sa, '-m') + 1] = $lv.model; $cfg.server_args = $sa }   # lever swaps the model file
    [IO.File]::WriteAllText("$h\configs\$name.json", ($cfg | ConvertTo-Json -Depth 5), (New-Object Text.UTF8Encoding $false))   # no BOM (PS 5.1 adds one)
    "$(Get-Date -f HH:mm:ss) start $name" | Add-Content "$h\queue.log"
    $j = Start-Process $py -ArgumentList @('-I', $bench, 'run', "$h\configs\$name.json", '--skip-cN') -PassThru -NoNewWindow -RedirectStandardOutput "$h\runs-$name.log" -RedirectStandardError "$h\runs-$name.err"
    if (-not $j.WaitForExit(15 * 60000)) { Stop-Process -Id $j.Id -Force; "$(Get-Date -f HH:mm:ss) TIMEOUT $name" | Add-Content "$h\queue.log" }
    & "$h\stop.ps1"
    "$(Get-Date -f HH:mm:ss) done $name" | Add-Content "$h\queue.log"
  }
} finally { Remove-Item $lock -ErrorAction SilentlyContinue; "QUEUE_DONE $Runs" | Add-Content "$h\queue.log" }
