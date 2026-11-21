# Stop the server started by launch.ps1 (by PID only). bench.py calls this after its requests finish, so the server is idle.
$h = Split-Path -Parent $MyInvocation.MyCommand.Path
if (Test-Path "$h\server.pid") {
  $id = [int](Get-Content "$h\server.pid")
  $p = Get-Process -Id $id -ErrorAction SilentlyContinue
  if ($p -and $p.ProcessName -eq 'llama-server') { Stop-Process -Id $id -Force; $p.WaitForExit(15000) | Out-Null; Start-Sleep 3 }
  Remove-Item "$h\server.pid"
}
