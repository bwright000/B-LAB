# Memory probe for bench.py's watchdog: prints bytes of *system RAM borrowed by the GPU* (WDDM "Shared Usage")
# for the server process, minus the ~440 MB baseline every CUDA process has. >1 GiB = VRAM spill -> watchdog stops the run.
$h = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path "$h\server.pid")) { 0; exit }
$id = [int](Get-Content "$h\server.pid")
$s = (Get-Counter '\GPU Process Memory(*)\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples | Where-Object { $_.InstanceName -match "pid_$($id)_" }
$b = [int64](($s | Measure-Object CookedValue -Sum).Sum) - 460MB
[math]::Max(0, $b)
