# Local Claude Code workspace (Inkling 2 Mini)

In this folder, the Claude Code VS Code extension uses the **local Inkling 2 Mini server** instead of Anthropic.
Everywhere else, Claude Code is unchanged.

## Use
1. Run `open-local-claude.ps1`. It starts the server (`..\start_inkling_fast.ps1`, GPU 1) if needed, then opens this folder in a new VS Code window.
2. Open Claude Code in that window. The first time, accept the folder-trust prompt, because project settings apply only to trusted folders.
3. Check it's local: run `/status` (base URL http://127.0.0.1:8080) or `/context` (context use against 32K).

To work on another project with the local model, either copy `.vscode/settings.json` and `.claude/settings.local.json` into it, or add that project's folder to this workspace.

## As a worker for the main Claude agent (`inkling.ps1`)
The main (Anthropic) Claude agent hands Inkling one small task with a chosen toolbox:
`inkling.ps1 -Dir <worktree> -Tools Read,Edit [-Bare] [-AllowBash '<pattern>'] -Task "<brief>"`.

- **What it does:** starts the server if needed, after checking that GPU 1 is free and RAM allows it.
- **Concurrency:** one worker at a time.
- **Output:** JSON with `ok`, `result`, `changed_files`, `turns`, `seconds` and `prompt_tokens`.
- **Project skill:** a short `.claude/skills/inkling-worker/SKILL.md` in the target repo tells the lead agent when to delegate, how to brief, which toolbox to give, and to review the diff and run tests before accepting.
- **Gotcha:** if the worker calls a tool it wasn't given, llama-server's tool grammar fails the request with a 500. Never mention shell commands in the prompt unless Bash is allowed.

## How it is wired
- **`.vscode/settings.json`:** `claudeCode.environmentVariables` holds the base URL and a dummy token. The extension checks these before launching. Scope: this workspace only.
- **`.claude/settings.local.json`:** sets the model name for all tiers and subagents, the 32K context, a 4K output cap, no telemetry or beta headers, and turns off WebFetch.
- **Server:** llama-server's Anthropic-compatible `/v1/messages`, with Inkling's tool-call parser. A tool_use round trip has been verified.

## Limits
- **Not supported by Anthropic.** Some Claude Code features may misbehave with a non-Claude model.
- **Not available here:** Remote Control, voice, claude.ai connectors (Gmail, Drive, ...), and web search/fetch.
- **32K context:** Claude Code's own system prompt and 22 tool definitions use **~23K tokens**, leaving ~9K for files and conversation. The first message takes ~27 s to read that prompt; later turns hit the prompt cache.
  For more room, start `..\start_inkling_64k.ps1` instead (64K, 2 layers of experts in RAM: 110 tok/s, but slower prompt reading at ~250 tok/s) and set `CLAUDE_CODE_MAX_CONTEXT_TOKENS` to 65536.
- **GPU 1 must be free of other jobs:** ~0.9 GB of headroom. If speed drops sharply, check `nvidia-smi`.
- **Speed:** ~130 tok/s for short answers. Reading a large prompt takes time: about 875 tok/s, so ~10 s for 8K tokens of files.
