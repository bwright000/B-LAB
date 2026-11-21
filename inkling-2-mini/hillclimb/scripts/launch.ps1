# Windows adapter for the skill's bench.py "launch" hook. Reads server_args/env/binary from the config JSON,
# writes a .cmd, and starts it via WMI (Win32_Process.Create) so llama-server inherits NO handles from bench.py
# (Start-Process leaked bench.py's stdout pipe into the server and bench.py waited forever). Returns immediately.
param([Parameter(Mandatory)][string]$Config)
$h = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$h\stop.ps1"
$c = Get-Content $Config -Raw | ConvertFrom-Json
$bin = if ($c.binary) { $c.binary } else { 'D:\Models\Inki\llama\llama-server.exe' }
$q = { param($a) if ($a -match '[\s"]') { '"' + ($a -replace '"','\"') + '"' } else { $a } }
$lines = @('@echo off', 'set CUDA_VISIBLE_DEVICES=1')
if ($c.env) { $c.env.PSObject.Properties | ForEach-Object { $lines += "set $($_.Name)=$($_.Value)" } }
$lines += ('"' + $bin + '" ' + (($c.server_args | ForEach-Object { & $q $_ }) -join ' ') + " > `"$h\server-$($c.name).out`" 2> `"$h\server-$($c.name).err`"")
$cmdFile = "$h\configs\$($c.name).cmd"; [IO.File]::WriteAllLines($cmdFile, $lines)
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }   # hidden console window
$r = Invoke-CimMethod Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd.exe /c `"$cmdFile`""; CurrentDirectory = $h; ProcessStartupInformation = $si }
for ($i = 0; $i -lt 20; $i++) {   # record the llama-server PID (child of that cmd.exe)
  $s = Get-CimInstance Win32_Process -Filter "ParentProcessId=$($r.ProcessId) AND Name='llama-server.exe'"
  if ($s) { Set-Content "$h\server.pid" $s.ProcessId; break }; Start-Sleep -Milliseconds 500 }
