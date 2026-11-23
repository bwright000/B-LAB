"""Does a Windows path survive an Anthropic-API tool call? toolpath.py <port> <model>"""
import json, sys, requests
port, model = sys.argv[1], sys.argv[2]; temp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0; reps = int(sys.argv[4]) if len(sys.argv) > 4 else 1
tools = [{"name": "Read", "description": "Read a file from disk and return its contents.",
          "input_schema": {"type": "object", "properties": {"file_path": {"type": "string", "description": "Absolute path"}}, "required": ["file_path"]}}]
ok = 0
paths = [r"D:\notes\todo.txt", r"C:\new\bin\test.py", r"D:\Models\B-LAB\README.md", r"C:\Users\ben\repo\src\utils\table.rs", "D:/notes/todo.txt", r"C:\temp\notes\final\report.md"] * reps
for p in paths:
    for attempt in range(2):
        r = requests.post(f"http://127.0.0.1:{port}/v1/messages", headers={"anthropic-version": "2023-06-01", "x-api-key": "local"},
                          json={"model": model, "max_tokens": 1024, "temperature": temp, "tools": tools,
                                "messages": [{"role": "user", "content": f"Read the file {p} with the Read tool."}]}, timeout=180)
        try:
            c = r.json().get("content", [])
        except Exception:
            c = []
        got = next((b["input"].get("file_path") for b in c if b.get("type") == "tool_use"), None)
        if got is not None or attempt:
            break
    good = got == p; ok += good
    print(("OK  " if good else "BAD ") + f"{p!r} -> {got!r}" + ("" if got is not None else f"   (HTTP {r.status_code}: {r.text[:160]!r})"))
print(f"{ok}/{len(paths)} paths intact")
