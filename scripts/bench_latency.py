"""Latency per tool, measured at the client (full MCP round-trip), over stdio and HTTP.

Each tool gets N calls with varied arguments (queries from the search eval set, random
product ids). The HTTP server is started here as a subprocess with a random token.

    uv run python scripts/bench_latency.py --data-dir data --calls 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import httpx2
import numpy as np

from catalog_mcp.auth import generate_token
from catalog_mcp.client import http_client, stdio_client
from catalog_mcp.config import paths

ROOT = Path(__file__).resolve().parents[1]


def workload(data_dir: str | None, n: int, seed: int = 0) -> dict[str, list]:
    rng = random.Random(seed)
    queries = [q["query"] for q in json.loads((ROOT / "eval" / "search_queries.json").read_text(encoding="utf-8"))["queries"]]
    con = duckdb.connect(str(paths(data_dir).db), read_only=True)
    ids = [r[0] for r in con.execute(f"SELECT product_id FROM products USING SAMPLE {n * 4} ROWS (reservoir, {seed})").fetchall()]
    con.close()
    q = lambda i: queries[i % len(queries)]  # noqa: E731
    return {
        "search_products (hybrid)": [("search_products", {"query": q(i), "mode": "hybrid"}) for i in range(n)],
        "search_products (keyword)": [("search_products", {"query": q(i), "mode": "keyword"}) for i in range(n)],
        "search_products (semantic)": [("search_products", {"query": q(i), "mode": "semantic"}) for i in range(n)],
        "search_products (hybrid + filters)": [("search_products", {"query": q(i), "category": rng.choice(["Appliances", "Video_Games"]), "max_price": 100}) for i in range(n)],
        "get_product": [("get_product", {"product_id": ids[i]}) for i in range(n)],
        "compare_products (3 ids)": [("compare_products", {"product_ids": ids[n + 3 * i: n + 3 * i + 3]}) for i in range(n)],
        "find_near_duplicates": [("find_near_duplicates", {"product_id": ids[i]}) for i in range(n)],
        "resource catalog://stats": [("resource", "catalog://stats") for _ in range(n)],
    }


async def bench(client, work: dict[str, list]) -> dict[str, dict]:
    out = {}
    async with client as c:
        for name, calls in work.items():  # warm-up: model load, caches, connection
            for kind, args in calls[:5]:
                await (c.read_resource(args) if kind == "resource" else c.call_tool(kind, args))
        for name, calls in work.items():
            times, errors = [], 0
            for kind, args in calls:
                t = time.perf_counter()
                if kind == "resource":
                    await c.read_resource(args)
                else:
                    r = await c.call_tool(kind, args)
                    errors += bool(r.is_error)
                times.append((time.perf_counter() - t) * 1000)
            a = np.array(times)
            out[name] = {"n": len(a), "errors": errors, "p50_ms": round(float(np.percentile(a, 50)), 1),
                         "p95_ms": round(float(np.percentile(a, 95)), 1), "p99_ms": round(float(np.percentile(a, 99)), 1),
                         "mean_ms": round(float(a.mean()), 1)}
            print(f"  {name:36} p50 {out[name]['p50_ms']:7.1f}  p95 {out[name]['p95_ms']:7.1f} ms  errors {errors}")
    return out


def start_http(data_dir: str | None, token: str) -> tuple[subprocess.Popen, str]:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cmd = [sys.executable, "-m", "catalog_mcp.cli", "serve", "--transport", "http", "--port", str(port)]
    if data_dir:
        cmd += ["--data-dir", data_dir]
    proc = subprocess.Popen(cmd, env={**os.environ, "CATALOG_MCP_TOKEN": token},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            if httpx2.get(f"{base}/healthz").status_code == 200:
                break
        except httpx2.HTTPError:
            time.sleep(0.1)
    return proc, f"{base}/mcp"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "docs"))
    args = ap.parse_args()
    work = workload(args.data_dir, args.calls)

    results = {}
    print("stdio:")
    results["stdio"] = asyncio.run(bench(stdio_client(args.data_dir), work))
    token = generate_token()
    proc, url = start_http(args.data_dir, token)
    try:
        print("streamable http:")
        results["http"] = asyncio.run(bench(http_client(url, token), work))
    finally:
        proc.terminate()

    import platform

    import torch

    meta = {"calls_per_tool": args.calls, "cpu": platform.processor(), "python": platform.python_version(),
            "embedding_device": "cuda" if torch.cuda.is_available() else "cpu"}
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "latency.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2), encoding="utf-8")
    lines = ["| Tool | stdio p50 | stdio p95 | HTTP p50 | HTTP p95 |", "|---|---|---|---|---|"]
    for name in work:
        s, h = results["stdio"][name], results["http"][name]
        lines.append(f"| {name} | {s['p50_ms']} | **{s['p95_ms']}** | {h['p50_ms']} | **{h['p95_ms']}** |")
    (out / "latency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    import matplotlib.pyplot as plt

    names = list(work)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(names))
    ax.barh(y - 0.2, [results["stdio"][n]["p95_ms"] for n in names], 0.4, label="stdio", color="#3b7dd8")
    ax.barh(y + 0.2, [results["http"][n]["p95_ms"] for n in names], 0.4, label="Streamable HTTP", color="#1f9e74")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("p95 latency (ms), client round-trip")
    ax.set_title(f"p95 latency per tool ({args.calls} calls each)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "latency_p95.png", dpi=160)


if __name__ == "__main__":
    main()
