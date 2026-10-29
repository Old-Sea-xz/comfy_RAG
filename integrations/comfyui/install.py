"""Install the queue extension into an explicitly selected, stopped ComfyUI."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
HELPERS = ("prompt_memory_cleanup.py", "prompt_queue_recovery.py")


def planned_files(root: Path) -> dict[str, bytes]:
    manifest = json.loads((HERE / "upstream.json").read_text())
    originals = {}
    for name, expected in manifest["base_sha256"].items():
        content = (root / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"{name} differs from supported ComfyUI {manifest['revision']}; no files changed")
        originals[name] = content.decode()
    main = originals["main.py"]
    worker = next(n for n in ast.parse(main).body
                  if isinstance(n, ast.FunctionDef) and n.name == "prompt_worker")
    lines = main.splitlines(keepends=True)
    lines[worker.lineno - 1:worker.end_lineno] = [(HERE / "prompt_worker.py").read_text()]
    main = "".join(lines)
    changes = {
        "import comfy.model_patcher\n":
            "import comfy.model_patcher\n"
            "from prompt_memory_cleanup import release_prompt_memory, MemoryReleaseTimeout\n"
            "from prompt_queue_recovery import restart_with_queue_checkpoint, restore_queue_checkpoint\n",
        "    threading.Thread(target=prompt_worker,":
            "    restore_queue_checkpoint(prompt_server)\n    threading.Thread(target=prompt_worker,",
    }
    for old, new in changes.items():
        if main.count(old) != 1:
            raise ValueError("Unsupported main.py integration anchor; no files changed")
        main = main.replace(old, new, 1)
    server = originals["server.py"]
    for old, new in {
        "        self.prompt_queue = execution.PromptQueue(self)\n":
            "        self.prompt_queue = execution.PromptQueue(self)\n"
            "        self.queue_cleanup_state = {\"supported\": True, \"phase\": \"idle\", \"boot_id\": str(uuid.uuid4())}\n",
        "        @routes.get(\"/queue\")\n":
            "        @routes.get(\"/queue_cleanup\")\n"
            "        async def get_queue_cleanup(request):\n"
            "            return web.json_response(dict(self.queue_cleanup_state))\n\n"
            "        @routes.get(\"/queue\")\n",
    }.items():
        if server.count(old) != 1:
            raise ValueError("Unsupported server.py integration anchor; no files changed")
        server = server.replace(old, new, 1)
    result = {"main.py": main.encode(), "server.py": server.encode()}
    result.update({name: (HERE / name).read_bytes() for name in HELPERS})
    for name, content in result.items():
        compile(content, name, "exec")
        if name in HELPERS and (root / name).exists():
            raise ValueError(f"Existing {name} would be overwritten; no files changed")
        if (root / (name + ".comfy-rag-backup")).exists():
            raise ValueError(f"Backup for {name} already exists; no files changed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Write changes; default only validates")
    args = parser.parse_args()
    root = args.comfyui.resolve()
    files = planned_files(root)
    if not args.apply:
        print("Validated queue integration; 4 files planned. Stop this ComfyUI before using --apply.")
        return
    for name in ("main.py", "server.py"):
        backup = root / (name + ".comfy-rag-backup")
        with backup.open("xb") as handle:
            handle.write((root / name).read_bytes())
    for name, content in files.items():
        (root / name).write_bytes(content)
    print("Queue integration installed. Install psutil and restart ComfyUI; verify GET /queue_cleanup.")


if __name__ == "__main__":
    main()
