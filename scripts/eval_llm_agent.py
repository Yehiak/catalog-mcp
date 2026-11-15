"""LLM-agent eval: can a real model use the tools from their descriptions alone?

A local model (default: Qwen 3.8 27B through Ollama) gets the tool list exactly as the MCP
server advertises it (names, descriptions, JSON schemas) and 30 user requests in plain
English. An agent loop forwards its tool calls to the server over stdio and returns the
results, up to --max-turns rounds. Per task we record:

- success: the task's check passes (right tool(s), right arguments, answer grounded in results)
- first-tool accuracy: the first tool the model picked is the expected one
- valid-call rate: share of the model's tool calls the server accepted (no validation error)

    ollama serve
    uv run python scripts/eval_llm_agent.py --data-dir data
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx2

from catalog_mcp.client import stdio_client

ROOT = Path(__file__).resolve().parents[1]
ASIN = re.compile(r"\b[A-Z0-9]{10}\b")
SYSTEM = (
    "You are a shopping assistant with access to a product catalog through tools. "
    "Use the tools to answer; never invent products, prices or ids. "
    "When you mention a product, include its product_id. Be concise."
)


@dataclass
class Run:
    calls: list[dict] = field(default_factory=list)  # {"name", "args", "error", "result"}
    answer: str = ""

    def called(self, name: str) -> list[dict]:
        return [c for c in self.calls if c["name"] == name and not c["error"]]

    def results(self, name: str = "search_products") -> list[dict]:
        return [r for c in self.called(name) for r in (c["result"] or {}).get("results", [])]

    def ids_in_answer(self) -> set[str]:
        return set(ASIN.findall(self.answer))


@dataclass
class Task:
    id: str
    prompt: str
    first_tool: str
    check: Callable[[Run], bool]


def first_id(text: str) -> str | None:
    """The first product id in an answer: models lead with the verdict."""
    m = ASIN.search(text)
    return m.group(0) if m else None


def _any_arg(run: Run, tool: str, pred) -> bool:
    return any(pred(c["args"]) for c in run.called(tool))


async def fixtures(c) -> dict[str, dict]:
    """Real products to talk about, found through the server itself."""
    async def top(query, n=1, **kw):
        r = await c.call_tool("search_products", {"query": query, "limit": n, **kw})
        return r.structured_content["results"]

    async def get(pid):
        return (await c.call_tool("get_product", {"product_id": pid})).structured_content

    ice = await top("countertop ice maker", 2, min_price=50)
    charge = await top("PS5 controller charging station", 3)
    filt = (await top("DA29-00020B refrigerator water filter", 1))[0]
    zelda = (await top("Legend of Zelda Breath of the Wild Nintendo Switch", 1))[0]
    wheel = (await top("racing wheel with pedals", 1))[0]
    weighted = next(p for p in [await get(r["product_id"]) for r in await top("chest freezer", 5, category="Appliances")]
                    if "Item Weight" in p["details"])
    return {"ice": ice, "charge": charge, "filter": filt, "zelda": zelda, "wheel": wheel, "weighted": weighted}


def tasks(fx: dict) -> list[Task]:
    i1, i2 = fx["ice"][0], fx["ice"][1]
    ch = fx["charge"]
    cheaper = min((i1, i2), key=lambda p: p["price"])
    most_rev = max(ch, key=lambda p: p["rating_number"])
    w = fx["weighted"]
    weight_num = re.search(r"[\d.]+", w["details"]["Item Weight"]).group(0)
    T = Task
    return [
        # --- search (10)
        T("s01", "Find me a replacement refrigerator water filter with part number W10295370A.", "search_products",
          lambda r: any("W10295370" in x["title"].upper() for x in r.results()) and "W10295370" in r.answer.upper().replace("-", "")),
        T("s02", "What Mario Kart games do you have for under $50?", "search_products",
          lambda r: _any_arg(r, "search_products", lambda a: (a.get("max_price") or 1e9) <= 50) and "mario kart" in r.answer.lower()),
        T("s03", "Show me video game controllers rated at least 4.5 stars.", "search_products",
          lambda r: _any_arg(r, "search_products", lambda a: (a.get("min_rating") or 0) >= 4.5)),
        T("s04", "I need a countertop ice maker under $150. What are my options?", "search_products",
          lambda r: _any_arg(r, "search_products", lambda a: (a.get("max_price") or 1e9) <= 150) and bool(r.ids_in_answer())),
        T("s05", "List Samsung refrigerator water filters.", "search_products",
          lambda r: _any_arg(r, "search_products", lambda a: "samsung" in (str(a.get("brand")) + a.get("query", "")).lower())),
        T("s06", "Find a racing wheel with pedals that works with PS5.", "search_products",
          lambda r: bool(r.ids_in_answer() & {x["product_id"] for x in r.results()})),
        T("s07", "Are there chest freezers in the catalog? List three with their ids.", "search_products",
          lambda r: len(r.ids_in_answer() & {x["product_id"] for x in r.results()}) >= 3),
        T("s08", "Find the part DA29-00020B.", "search_products",
          lambda r: bool({x["product_id"] for x in r.results() if "DA29-00020B" in x["title"].upper() or "DA2900020B" in x["title"].upper()} & r.ids_in_answer())),
        T("s09", "Recommend a well-rated gaming headset for Xbox that costs less than $40.", "search_products",
          lambda r: _any_arg(r, "search_products", lambda a: (a.get("max_price") or 1e9) <= 40)),
        T("s10", "Search only the Appliances category for dishwasher rack wheels.", "search_products",
          lambda r: _any_arg(r, "search_products", lambda a: a.get("category") == "Appliances")),
        # --- details (5)
        T("d01", f"What are the main features of product {i1['product_id']}?", "get_product",
          lambda r: _any_arg(r, "get_product", lambda a: a.get("product_id") == i1["product_id"])),
        T("d02", f"What's the price and average rating of {i2['product_id']}?", "get_product",
          lambda r: str(i2["average_rating"]) in r.answer and f"{i2['price']:.2f}".rstrip("0").rstrip(".") in r.answer),
        T("d03", f"Which brand makes product {fx['wheel']['product_id']}?", "get_product",
          lambda r: (fx["wheel"]["brand"] or "").lower() in r.answer.lower()),
        T("d04", f"How much does product {w['product_id']} weigh?", "get_product",
          lambda r: weight_num in r.answer),
        T("d05", "Tell me about product B000000000.", "get_product",
          lambda r: bool(re.search(r"not (be )?found|no product|doesn't exist|does not exist|couldn't find|could not find|unable to find|invalid", r.answer, re.I))),
        # --- compare (6)
        T("c01", f"Compare {i1['product_id']} and {i2['product_id']}.", "compare_products",
          lambda r: _any_arg(r, "compare_products", lambda a: set(a.get("product_ids", [])) == {i1["product_id"], i2["product_id"]})),
        T("c02", f"Which is cheaper, {i1['product_id']} or {i2['product_id']}?", "compare_products",
          lambda r: cheaper["product_id"] in r.answer),
        T("c03", f"Of {ch[0]['product_id']}, {ch[1]['product_id']} and {ch[2]['product_id']}, which one has the most reviews?", "compare_products",
          lambda r: first_id(r.answer) == most_rev["product_id"]),
        T("c04", "Find two countertop ice makers and compare them for me.", "search_products",
          lambda r: bool(r.called("search_products")) and any(len(c["args"].get("product_ids", [])) >= 2 for c in r.called("compare_products"))),
        T("c05", "Compare the top three PS5 controller charging stations and tell me which is best rated.", "search_products",
          lambda r: any(len(c["args"].get("product_ids", [])) >= 3 for c in r.called("compare_products"))),
        T("c06", f"Put {i1['product_id']} and {i2['product_id']} side by side: capacity, weight and price.", "compare_products",
          lambda r: bool(r.called("compare_products")) and bool(re.search(r"pound|lbs?\b", r.answer, re.I))),
        # --- near-duplicates (5)
        T("u01", f"Is product {fx['filter']['product_id']} also listed by other sellers?", "find_near_duplicates",
          lambda r: _any_arg(r, "find_near_duplicates", lambda a: a.get("product_id") == fx["filter"]["product_id"])),
        T("u02", "I'm about to add a listing titled 'Replacement Water Filter for Samsung DA29-00020B, 2 Pack'. Does something like it already exist?", "find_near_duplicates",
          lambda r: _any_arg(r, "find_near_duplicates", lambda a: bool(a.get("title")) and not a.get("product_id"))),
        T("u03", f"Are there duplicate listings of {fx['zelda']['product_id']}? Editions for other consoles don't count.", "find_near_duplicates",
          lambda r: _any_arg(r, "find_near_duplicates", lambda a: a.get("product_id") == fx["zelda"]["product_id"])),
        T("u04", f"Find near-duplicates of {fx['wheel']['product_id']}, but only very close matches (similarity at least 0.93).", "find_near_duplicates",
          lambda r: _any_arg(r, "find_near_duplicates", lambda a: abs((a.get("min_similarity") or 0) - 0.93) < 1e-6)),
        T("u05", "Find duplicate listings of the WP99003147 dishwasher rack wheel.", "search_products",
          lambda r: bool(r.called("search_products")) and bool(r.called("find_near_duplicates"))),
        # --- other (4)
        T("o01", "Do you have a 'zzqx unicorn blender 9000'?", "search_products",
          lambda r: bool(r.called("search_products")) and bool(re.search(r"\bno\b|not|couldn't|could not|none|unable", r.answer, re.I))),
        T("o02", "Suggest PC or console games about open-world space exploration.", "search_products",
          lambda r: bool(r.ids_in_answer() & {x["product_id"] for x in r.results() if x["category"] == "Video_Games"})),
        T("o03", "What's the cheapest DualSense controller you can find?", "search_products",
          lambda r: bool(r.ids_in_answer() & {x["product_id"] for x in r.results()})),
        T("o04", "List Whirlpool dryer parts.", "search_products",
          lambda r: _any_arg(r, "search_products", lambda a: "whirlpool" in (str(a.get("brand")) + a.get("query", "")).lower())
          and bool(r.ids_in_answer())),
    ]


def ollama_tools(mcp_tools) -> list[dict]:
    return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                              "parameters": t.input_schema}} for t in mcp_tools]


async def agent(c, http, model, tools, prompt, max_turns, think) -> Run:
    run = Run()
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    for _ in range(max_turns):
        resp = await http.post("/api/chat", json={"model": model, "messages": messages, "tools": tools, "stream": False,
                                                  "think": think, "options": {"temperature": 0, "num_ctx": 16384}})
        resp.raise_for_status()
        msg = resp.json()["message"]
        messages.append(msg)
        if not msg.get("tool_calls"):
            run.answer = msg.get("content", "")
            return run
        for tc in msg["tool_calls"]:
            name, args = tc["function"]["name"], tc["function"].get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args or "{}")
            r = await c.call_tool(name, args)
            text = r.content[0].text if r.content else ""
            run.calls.append({"name": name, "args": args, "error": bool(r.is_error),
                              "result": None if r.is_error else r.structured_content})
            messages.append({"role": "tool", "tool_name": name, "content": text[:6000]})
    run.answer = "(no final answer within the turn limit)"
    return run


async def main_async(args) -> dict:
    env = {"CUDA_VISIBLE_DEVICES": "-1"}  # leave the GPU to the LLM; query embedding is fast on CPU
    async with stdio_client(args.data_dir, env=env) as c, httpx2.AsyncClient(base_url=args.ollama, timeout=600) as http:
        tools = ollama_tools((await c.list_tools()).tools)
        all_tasks = tasks(await fixtures(c))
        rows = []
        for t in all_tasks:
            start = time.perf_counter()
            try:
                run = await agent(c, http, args.model, tools, t.prompt, args.max_turns, args.think)
                ok = bool(t.check(run))
            except Exception as e:  # noqa: BLE001
                run, ok = Run(answer=f"{type(e).__name__}: {e}"), False
            row = {"id": t.id, "prompt": t.prompt, "success": ok,
                   "first_tool_ok": bool(run.calls) and run.calls[0]["name"] == t.first_tool,
                   "tool_calls": len(run.calls), "invalid_calls": sum(c["error"] for c in run.calls),
                   "calls": [{"name": c["name"], "args": c["args"], "error": c["error"]} for c in run.calls],
                   "answer": run.answer, "seconds": round(time.perf_counter() - start, 1)}
            rows.append(row)
            print(f"{'PASS' if ok else 'FAIL'} {t.id} first_tool={'ok' if row['first_tool_ok'] else 'X'} "
                  f"calls={row['tool_calls']} invalid={row['invalid_calls']} {row['seconds']}s", flush=True)
    n = len(rows)
    total_calls = sum(r["tool_calls"] for r in rows)
    return {
        "model": args.model, "think": args.think, "tasks": n,
        "success_rate": sum(r["success"] for r in rows) / n,
        "first_tool_accuracy": sum(r["first_tool_ok"] for r in rows) / n,
        "valid_call_rate": 1 - sum(r["invalid_calls"] for r in rows) / max(total_calls, 1),
        "tool_calls": total_calls, "mean_seconds_per_task": sum(r["seconds"] for r in rows) / n,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--model", default="qwen3.8")
    ap.add_argument("--ollama", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--think", action="store_true", help="Let the model reason before answering (slower).")
    ap.add_argument("--out", default=str(ROOT / "docs"))
    args = ap.parse_args()
    if not args.ollama.startswith("http"):
        args.ollama = f"http://{args.ollama}"
    report = asyncio.run(main_async(args))
    Path(args.out).mkdir(exist_ok=True)
    (Path(args.out) / "llm_agent.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nmodel={report['model']} think={report['think']}: success {report['success_rate']:.1%}, "
          f"first-tool {report['first_tool_accuracy']:.1%}, valid calls {report['valid_call_rate']:.1%} "
          f"({report['tool_calls']} calls), {report['mean_seconds_per_task']:.1f}s/task")


if __name__ == "__main__":
    main()
