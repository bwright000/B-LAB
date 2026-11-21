<#
.SYNOPSIS
  Delegate one small, well-briefed coding task to the local Inkling 2 Mini worker (Claude Code headless,
  pointed at the local llama-server on GPU 1) with only the tools the caller chooses. Returns JSON.

.EXAMPLE
  powershell -NoProfile -File D:\Models\Inki\local-claude\inkling.ps1 -Dir C:\code\myrepo-worktree `
    -Tools Read,Edit -Task "In src/x.py, rename helper _foo to _bar and update its two callers in the same file."

.NOTES
  - The worker never commits and gets no Bash unless -AllowBash names exact command patterns.
  - Edits land in -Dir; the caller reviews `changed_files` / the diff and runs the tests itself.
  - One worker at a time (the server runs one slot); a lock file enforces it.
#>
param(
    [Parameter(Mandatory)][string]$Task,                 # the brief (or a path to a brief file with -TaskIsFile)
    [switch]$TaskIsFile,
    [string]$Dir = (Get-Location).Path,                  # working directory (prefer a lane worktree for edits)
    [string[]]$Tools = @('Read', 'Grep', 'Glob'),        # built-in tools the worker may use
    [string[]]$AllowBash = @(),                          # e.g. 'uv run python -m pytest tests/test_x.py *'
    [int]$MaxTurns = 12,
    [int]$TimeoutSec = 600,
    [switch]$Bare,                                       # smallest prompt (~1K tokens), least guidance
    [string]$ServerScript = 'D:\Models\Inki\start_inkling_fast.ps1',
    [double]$MinFreeRamGB = 5.78                         # RAM to leave free for other work
)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
function Out-Result($obj) { $obj | ConvertTo-Json -Depth 6; exit $(if ($obj.ok) { 0 } else { 1 }) }

$Tools = @($Tools | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })   # -File passes "A,B" as one string
if ($TaskIsFile) { $Task = [IO.File]::ReadAllText((Resolve-Path $Task)) }
if (-not (Test-Path $Dir)) { Out-Result @{ ok = $false; error = "Dir not found: $Dir" } }
$Dir = (Resolve-Path $Dir).Path

# --- one worker at a time --------------------------------------------------------------------------
$lock = Join-Path $here 'inkling-worker.lock'
if (Test-Path $lock) {
    $holder = Get-Content $lock -ErrorAction SilentlyContinue
    if ($holder -and (Get-Process -Id ([int]$holder) -ErrorAction SilentlyContinue)) { Out-Result @{ ok = $false; error = "another Inkling worker is running (PID $holder); retry later" } }
}
Set-Content $lock $PID
try {
    # --- server: reuse if healthy, else start it behind the RAM and VRAM gates ---------------------
    $healthy = { try { (Invoke-RestMethod http://127.0.0.1:8080/health -TimeoutSec 2).status -eq 'ok' } catch { $false } }
    if (-not (& $healthy)) {
        $freeGB = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
        if ($freeGB -lt ($MinFreeRamGB + 4)) { Out-Result @{ ok = $false; error = ("only {0:N1} GB RAM free; the server needs ~4 GB above the {1} GB spare" -f $freeGB, $MinFreeRamGB) } }
        $usedMiB = [int](nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
        if ($usedMiB -gt 1500) { Out-Result @{ ok = $false; error = "GPU 1 already has $usedMiB MiB in use by another job; the server needs it free" } }
        Start-Process powershell -ArgumentList '-NoProfile', '-File', $ServerScript -WindowStyle Minimized | Out-Null
        for ($i = 0; $i -lt 90 -and -not (& $healthy); $i++) { Start-Sleep 2 }
        if (-not (& $healthy)) { Out-Result @{ ok = $false; error = 'server did not become healthy within 3 minutes' } }
    }

    # --- the Claude Code binary bundled with the newest installed VS Code extension ----------------
    $ext = Get-ChildItem "$env:USERPROFILE\.vscode\extensions" -Directory -Filter 'anthropic.claude-code-*' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $cc = if ($ext) { Join-Path $ext.FullName 'resources\native-binary\claude.exe' } else { $null }
    if (-not $cc -or -not (Test-Path $cc)) { Out-Result @{ ok = $false; error = 'Claude Code binary not found (VS Code extension anthropic.claude-code)' } }

    # --- worker invocation ------------------------------------------------------------------------
    # Mention shell rules only when a shell exists: naming commands primes the model to call a Bash tool
    # it does not have, and llama-server's tool grammar then fails the request with a 500.
    $guard = @('You are a worker agent delegated one small task by a lead agent. Do exactly the task and nothing else.',
               "Your only tools are: $((@($Tools) + $(if ($AllowBash.Count) { 'Bash' } else { @() })) -join ', '). Do not call any other tool.")
    if ($AllowBash.Count) {
        $guard += 'Run only the commands you were allowed. Never run git commands that change state; never commit; never kill processes by image name. Use `uv run python`, never bare `python`.'
    }
    $guard += 'British English in comments and docstrings. Keep edits minimal and in the files named.',
              'Finish with a short report: what you changed (file and function), and anything you could not do.'
    $guard = $guard -join ' '
    $toolList = @($Tools) + @($AllowBash | ForEach-Object { "Bash($_)" })
    [string[]]$a = @('-p', $Task, '--max-turns', "$MaxTurns", '--output-format', 'json',
                     '--tools', ((@($Tools) + $(if ($AllowBash.Count) { 'Bash' } else { @() })) -join ','),
                     '--allowedTools', ($toolList -join ','), '--append-system-prompt', $guard,
                     '--strict-mcp-config')
    if ($Tools -match '^(Edit|Write)$') { $a += @('--permission-mode', 'acceptEdits') }
    if ($Bare) { $a += '--bare' }

    $before = @(git -C $Dir status --porcelain 2>$null)
    $env:ANTHROPIC_BASE_URL = 'http://127.0.0.1:8080'; $env:ANTHROPIC_AUTH_TOKEN = 'local-inkling'
    $env:ANTHROPIC_MODEL = 'inkling-2-mini'; $env:CLAUDE_CODE_SUBAGENT_MODEL = 'inkling-2-mini'
    $env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = '1'; $env:CLAUDE_CODE_MAX_CONTEXT_TOKENS = '32768'
    $env:CLAUDE_CODE_MAX_OUTPUT_TOKENS = '4096'; $env:CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS = '1'

    $outFile = Join-Path $env:TEMP "inkling-worker-$PID.json"
    $t0 = Get-Date
    $proc = Start-Process $cc -ArgumentList ($a | ForEach-Object { if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ } }) `
            -WorkingDirectory $Dir -NoNewWindow -PassThru -RedirectStandardOutput $outFile -RedirectStandardError "$outFile.err"
    if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
        Stop-Process -Id $proc.Id -Force    # our own child only, by PID
        Out-Result @{ ok = $false; error = "worker timed out after $TimeoutSec s"; changed_files = @(git -C $Dir status --porcelain 2>$null | Where-Object { $before -notcontains $_ }) }
    }
    $raw = Get-Content $outFile -Raw -Encoding UTF8 -ErrorAction SilentlyContinue
    Remove-Item $outFile, "$outFile.err" -ErrorAction SilentlyContinue
    $j = $null; try { $j = ($raw.Trim() -split "`n")[-1] | ConvertFrom-Json } catch {}
    $after = @(git -C $Dir status --porcelain 2>$null)
    Out-Result @{
        ok            = [bool]($j -and -not $j.is_error)
        result        = $(if ($j) { $j.result } else { $raw })
        changed_files = @($after | Where-Object { $before -notcontains $_ })
        turns         = $(if ($j) { $j.num_turns } else { $null })
        seconds       = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        prompt_tokens = $(if ($j) { $j.usage.input_tokens + $j.usage.cache_read_input_tokens } else { $null })
        tools         = $toolList
        dir           = $Dir
    }
}
finally { Remove-Item $lock -ErrorAction SilentlyContinue }
