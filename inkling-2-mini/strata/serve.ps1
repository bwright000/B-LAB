# Start/stop Strata's HTTP server (OpenAI + Anthropic API) around strata-inkling on GPU 1, isolated under this folder.
#   serve.ps1 -Config sq10.json [-Port 8092]     start (detached via WMI, hidden console; PID in serve.pid)
#   serve.ps1 -Stop                               stop the server this script started (by recorded PID tree only)
#   serve.ps1 -Config sq10.json -Think            same, but through serve_think.py (low-effort thinking instead of Strata's forced no-thinking)
param([string]$Config = 'sq10.json', [int]$Port = 8092, [switch]$Stop, [switch]$Think)
$h = Split-Path -Parent $MyInvocation.MyCommand.Path
function Stop-Mine {
  if (Test-Path "$h\serve.pid") {
    $id = [int](Get-Content "$h\serve.pid")
    $kids = Get-CimInstance Win32_Process -Filter "ParentProcessId=$id"
    foreach ($k in $kids) { if ($k.Name -eq 'strata-inkling.exe') { Stop-Process -Id $k.ProcessId -Force -ErrorAction SilentlyContinue } }
    $p = Get-Process -Id $id -ErrorAction SilentlyContinue
    if ($p -and $p.ProcessName -eq 'python') { Stop-Process -Id $id -Force; $p.WaitForExit(15000) | Out-Null }
    Remove-Item "$h\serve.pid"; Start-Sleep 2
  }
}
Stop-Mine
if ($Stop) { 'stopped'; exit }
$name = [IO.Path]::GetFileNameWithoutExtension($Config)
$lines = @('@echo off', "set STRATA_HOME=$h\home", 'set CUDA_DEVICE_ORDER=PCI_BUS_ID',
  "`"$h\control\python\python.exe`" -I `"$(if ($Think) { "$h\serve_think.py" } else { "$h\control\serve\server.py" })`" --engine strata --config `"$h\$Config`" --host 127.0.0.1 --port $Port --gpu 1 --no-api-key > `"$h\home\serve-$name.out`" 2> `"$h\home\serve-$name.err`"")
$cmdFile = "$h\home\serve-$name.cmd"; [IO.File]::WriteAllLines($cmdFile, $lines)
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$r = Invoke-CimMethod Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd.exe /c `"$cmdFile`""; CurrentDirectory = "$h\control"; ProcessStartupInformation = $si }
for ($i = 0; $i -lt 20; $i++) {
  $s = Get-CimInstance Win32_Process -Filter "ParentProcessId=$($r.ProcessId) AND Name='python.exe'"
  if ($s) { Set-Content "$h\serve.pid" $s.ProcessId; "started python pid $($s.ProcessId) on port $Port"; break }; Start-Sleep -Milliseconds 500 }
