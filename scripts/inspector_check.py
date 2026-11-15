"""Black-box check with the official MCP Inspector CLI, over stdio and Streamable HTTP.

Runs the Inspector (Node, via npx) against the real server: list and call every tool,
read the resources, render the prompt, and run the Inspector's strict schema-portability
check on the tool list. Needs Node 18+.

    uv run python scripts/inspector_check.py --data-dir data
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bench_latency import start_http  # noqa: E402

from catalog_mcp.auth import generate_token  # noqa: E402

INSPECTOR = "@modelcontextprotocol/inspector@2.13.0"


def inspector(target: list[str], *args: str) -> tuple[int, str, str]:
    npx = shutil.which("npx") or "npx"
    p = subprocess.run([npx, "-y", INSPECTOR, "--cli", *target, *args], capture_output=True, text=True,
                       encoding="utf-8", timeout=300)
    return p.returncode, p.stdout, p.stderr


def checks(pid: str, pid2: str) -> list[tuple[str, list[str], callable]]:
    return [
        ("tools/list (--strict)", ["--method", "tools/list", "--strict"],
         lambda o: {t["name"] for t in o["tools"]} == {"search_products", "get_product", "compare_products", "find_near_duplicates"}),
        ("tools/call search_products", ["--method", "tools/call", "--tool-name", "search_products",
                                        "--tool-arg", "query=countertop ice maker", "--tool-arg", "limit=3"],
         lambda o: len(o["structuredContent"]["results"]) == 3),
        ("tools/call get_product", ["--method", "tools/call", "--tool-name", "get_product", "--tool-arg", f"product_id={pid}"],
         lambda o: o["structuredContent"]["product_id"] == pid),
        ("tools/call compare_products", ["--method", "tools/call", "--tool-name", "compare_products",
                                         "--tool-arg", f'product_ids=["{pid}", "{pid2}"]'],
         lambda o: len(o["structuredContent"]["products"]) == 2),
        ("tools/call with an unknown id -> tool error", ["--method", "tools/call", "--tool-name", "get_product",
                                                       "--tool-arg", "product_id=B000000000"],
         "tool_is_error"),
        ("tools/call find_near_duplicates", ["--method", "tools/call", "--tool-name", "find_near_duplicates",
                                             "--tool-arg", f"product_id={pid}"],
         lambda o: "candidates" in o["structuredContent"]),
        ("resources/list", ["--method", "resources/list"],
         lambda o: [r["uri"] for r in o["resources"]] == ["catalog://stats"]),
        ("resources/templates/list", ["--method", "resources/templates/list"],
         lambda o: o["resourceTemplates"][0]["uriTemplate"] == "catalog://categories/{category}/stats"),
        ("resources/read", ["--method", "resources/read", "--uri", "catalog://categories/Video_Games/stats"],
         lambda o: json.loads(o["contents"][0]["text"])["category"] == "Video_Games"),
        ("prompts/get", ["--method", "prompts/get", "--prompt-name", "write_product_comparison",
                         "--prompt-args", f"product_ids={pid},B000000001", "audience=a student"],
         lambda o: pid in o["messages"][0]["content"]["text"]),
    ]


def run_suite(label: str, target: list[str], pid: str, pid2: str) -> int:
    failures = 0
    for name, args, ok in checks(pid, pid2):
        code, out, err = inspector(target, *args)
        try:
            if isinstance(ok, str):  # expected failure: the Inspector reports the tool error
                passed = code != 0 and ok in (out + err)
            else:
                passed = code == 0 and ok(json.loads(out))
        except Exception:  # noqa: BLE001
            passed = False
        failures += not passed
        print(f"  {'PASS' if passed else 'FAIL'} [{label}] {name}" + ("" if passed else f"\n    {(err or out)[:400]}"))
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    exe = shutil.which("catalog-mcp") or "catalog-mcp"
    stdio = [exe, "serve", "--no-warm-up"] + (["--data-dir", args.data_dir] if args.data_dir else [])
    code, out, err = inspector(stdio, "--method", "tools/call", "--tool-name", "search_products",
                               "--tool-arg", "query=DA29-00020B", "--tool-arg", "mode=keyword", "--tool-arg", "limit=2")
    if code != 0:
        sys.exit(f"inspector could not reach the server:\n{err}")
    pid, pid2 = (r["product_id"] for r in json.loads(out)["structuredContent"]["results"])

    print(f"MCP Inspector {INSPECTOR.rsplit('@', 1)[1]}")
    failures = run_suite("stdio", stdio, pid, pid2)
    token = generate_token()
    proc, url = start_http(args.data_dir, token)
    try:
        failures += run_suite("http", [url, "--transport", "http", "--header", f"Authorization: Bearer {token}"], pid, pid2)
        code, _, _ = inspector([url, "--transport", "http"], "--method", "tools/list")
        print(f"  {'PASS' if code != 0 else 'FAIL'} [http] rejected without token")
        failures += code == 0
    finally:
        proc.terminate()
    total = 2 * len(checks(pid, pid2)) + 1
    print(f"\n{total - failures}/{total} Inspector checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
