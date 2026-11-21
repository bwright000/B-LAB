# Starts the local Inkling 2 Mini server (if not already up) and opens this folder in a new VS Code window,
# where the Claude Code extension talks to the local model instead of Anthropic.
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$up = $false
try { $up = (Invoke-RestMethod http://127.0.0.1:8080/health -TimeoutSec 2).status -eq 'ok' } catch {}
if (-not $up) {
    Write-Host 'Starting Inkling 2 Mini server on GPU 1 (keep GPU 1 free of other jobs)...'
    Start-Process powershell -ArgumentList '-NoProfile', '-File', 'D:\Models\Inki\start_inkling_fast.ps1' -WindowStyle Minimized
    for ($i = 0; $i -lt 90 -and -not $up; $i++) { Start-Sleep 2; try { $up = (Invoke-RestMethod http://127.0.0.1:8080/health -TimeoutSec 2).status -eq 'ok' } catch {} }
    if (-not $up) { throw 'Server did not come up within 3 minutes - check the minimized server window.' }
}
Write-Host 'Server ready at http://127.0.0.1:8080 - opening VS Code...'
code --new-window $here
