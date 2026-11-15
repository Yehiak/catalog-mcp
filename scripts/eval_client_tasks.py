"""Tool-call success rate: 30 scripted client tasks against the real catalog.

Each task is what an MCP client would do for one user request: one or more tool calls
(later calls use ids from earlier results) plus checks on the results. A task succeeds
only if every call returns without a protocol error and every check passes. Tasks that
send bad input succeed when the server answers with a clean, explanatory tool error.

    uv run python scripts/eval_client_tasks.py --data-dir data --transport stdio
    uv run python scripts/eval_client_tasks.py --transport http --url http://127.0.0.1:8000/mcp
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

from catalog_mcp.client import http_client, stdio_client

ROOT = Path(__file__).resolve().parents[1]


class Check(Exception):
    pass


def expect(cond, msg):
    if not cond:
        raise Check(msg)


class Session:
    def __init__(self, client):
        self.c = client
        self.calls = 0

    async def tool(self, name, args, ok=True):
        self.calls += 1
        r = await self.c.call_tool(name, args)
        if ok:
            expect(not r.is_error, f"{name} failed: {r.content[0].text[:200] if r.content else ''}")
            return r.structured_content
        expect(r.is_error, f"{name} should have failed for {args}")
        return r.content[0].text

    async def search(self, query, **kw):
        return (await self.tool("search_products", {"query": query, **kw}))["results"]

    async def resource(self, uri):
        self.calls += 1
        return json.loads((await self.c.read_resource(uri)).contents[0].text)


def titled(results, pattern, n=None):
    rx = re.compile(pattern, re.I)
    return [r for r in (results[:n] if n else results) if rx.search(r["title"])]


# ---------------------------------------------------------------- the tasks

async def t01(s):  # exact part number lookup
    res = await s.search("W10295370A", limit=5)
    expect(titled(res, r"W10295370", 3), "no W10295370 listing in top 3")

async def t02(s):  # named game
    res = await s.search("Mario Kart 8 Deluxe", limit=5)
    expect(titled(res, r"mario kart 8", 3), "no Mario Kart 8 in top 3")

async def t03(s):  # descriptive query, semantic
    res = await s.search("small machine that makes ice cubes for drinks", mode="semantic", limit=5)
    expect(len(titled(res, r"ice")) >= 3, "fewer than 3 ice-related results")

async def t04(s):  # category filter
    res = await s.search("controller", category="Video_Games", limit=10)
    expect(res and all(r["category"] == "Video_Games" for r in res), "category filter leaked")

async def t05(s):  # price range filter
    res = await s.search("water filter", min_price=10, max_price=25, limit=10)
    expect(res and all(r["price"] is not None and 10 <= r["price"] <= 25 for r in res), "price filter violated")

async def t06(s):  # rating filter
    res = await s.search("gaming headset", min_rating=4.5, limit=10)
    expect(res and all(r["average_rating"] >= 4.5 for r in res), "rating filter violated")

async def t07(s):  # brand filter
    res = await s.search("refrigerator water filter", brand="samsung", limit=10)
    expect(res and all("samsung" in f"{r['brand']} {r['store']}".lower() for r in res), "brand filter violated")

async def t08(s):  # keyword mode, hyphenated part number
    res = await s.search("DA29-00020B", mode="keyword", limit=5)
    expect(titled(res, r"DA-?29-?00020B", 3), "DA29-00020B not in top 3")

async def t09(s):  # results are ranked and limited
    res = await s.search("dishwasher rack wheel", limit=7)
    expect(len(res) == 7, f"expected 7 results, got {len(res)}")
    expect(all(a["score"] >= b["score"] for a, b in zip(res, res[1:])), "scores not descending")

async def t10(s):  # filter combination that matches nothing -> empty, not an error
    res = await s.search("ice maker", category="Video_Games", min_price=5000, limit=5)
    expect(res == [], "expected no results")

async def t11(s):  # search -> get details
    pid = (await s.search("countertop ice maker", limit=1))[0]["product_id"]
    p = await s.tool("get_product", {"product_id": pid})
    expect(p["product_id"] == pid and p["features"], "missing features")

async def t12(s):  # details carry the spec table
    pid = (await s.search("Samsung DC31-00054A drain pump", limit=1))[0]["product_id"]
    p = await s.tool("get_product", {"product_id": pid})
    expect(p["details"], "empty details")

async def t13(s):  # description truncation honoured
    pid = (await s.search("Elden Ring PS5", limit=1))[0]["product_id"]
    p = await s.tool("get_product", {"product_id": pid, "max_description_chars": 50})
    expect(len(p["description"]) <= 50, "description not truncated")

async def t14(s):  # category path present for a game
    pid = (await s.search("Animal Crossing New Horizons", limit=1))[0]["product_id"]
    p = await s.tool("get_product", {"product_id": pid})
    expect(p["category"] == "Video_Games" and p["categories"], "category info missing")

async def t15(s):  # compare two ice makers
    ids = [r["product_id"] for r in await s.search("countertop ice maker", limit=2)]
    c = await s.tool("compare_products", {"product_ids": ids})
    expect(len(c["products"]) == 2 and all(len(row["values"]) == 2 for row in c["specs"]), "bad shape")

async def t16(s):  # cheapest is really the cheapest
    res = await s.search("refrigerator water filter", min_price=1, limit=4)
    c = await s.tool("compare_products", {"product_ids": [r["product_id"] for r in res]})
    cheapest = min(res, key=lambda r: r["price"])["product_id"]
    expect(c["cheapest_id"] == cheapest, "cheapest_id wrong")

async def t17(s):  # most reviewed is really the most reviewed
    res = await s.search("PS5 controller charging station", limit=5)
    c = await s.tool("compare_products", {"product_ids": [r["product_id"] for r in res]})
    expect(c["most_reviewed_id"] == max(res, key=lambda r: r["rating_number"])["product_id"], "most_reviewed_id wrong")

async def t18(s):  # comparison exposes shared specs beyond the core fields
    ids = [r["product_id"] for r in await s.search("chest freezer", category="Appliances", limit=3)]
    c = await s.tool("compare_products", {"product_ids": ids})
    expect(len(c["specs"]) > 5, "no specs beyond the core fields")

async def t19(s):  # compare across categories still works
    a = (await s.search("Minecraft", limit=1))[0]["product_id"]
    b = (await s.search("ice maker", limit=1))[0]["product_id"]
    c = await s.tool("compare_products", {"product_ids": [a, b]})
    expect({p["category"] for p in c["products"]} == {"Appliances", "Video_Games"}, "categories wrong")

async def t20(s):  # dedup: same part from several sellers
    pid = (await s.search("DA29-00020B refrigerator water filter", limit=1))[0]["product_id"]
    d = await s.tool("find_near_duplicates", {"product_id": pid})
    expect(any(c["verdict"] == "likely_duplicate" and c["shared_model_numbers"] for c in d["candidates"]),
           "no likely duplicate with a shared part number")

async def t21(s):  # dedup: platform editions are variants, not duplicates
    pid = (await s.search("Legend of Zelda Breath of the Wild Nintendo Switch", limit=1))[0]["product_id"]
    d = await s.tool("find_near_duplicates", {"product_id": pid, "min_similarity": 0.8})
    for c in d["candidates"]:
        if re.search(r"wii u", c["title"], re.I):
            expect(c["verdict"] == "variant", "Wii U edition not flagged as variant")

async def t22(s):  # dedup a draft listing title before adding it
    d = await s.tool("find_near_duplicates", {"title": "Replacement Water Filter for Whirlpool W10295370A EDR1RXD1"})
    expect(d["query_product_id"] is None and d["candidates"], "no candidates for a common filter")

async def t23(s):  # dedup respects the similarity floor and the limit
    pid = (await s.search("dishwasher rack wheel", limit=1))[0]["product_id"]
    d = await s.tool("find_near_duplicates", {"product_id": pid, "min_similarity": 0.9, "limit": 3})
    expect(len(d["candidates"]) <= 3 and all(c["similarity"] >= 0.9 for c in d["candidates"]), "floor/limit ignored")

async def t24(s):  # dedup stays in the product's category by default
    pid = (await s.search("Nintendo Switch carrying case", limit=1))[0]["product_id"]
    d = await s.tool("find_near_duplicates", {"product_id": pid})
    expect(all(c["category"] == "Video_Games" for c in d["candidates"]), "left the category")

async def t25(s):  # overview resource
    st = await s.resource("catalog://stats")
    expect(st["products"] > 200_000 and set(st["categories"]) == {"Appliances", "Video_Games"}, "bad overview")

async def t26(s):  # category stats resource
    st = await s.resource("catalog://categories/Appliances/stats")
    expect(st["price_usd"]["p10"] <= st["price_usd"]["median"] <= st["price_usd"]["p90"], "price percentiles unordered")

async def t27(s):  # prompt template renders with the arguments
    ids = [r["product_id"] for r in await s.search("racing wheel with pedals", limit=2)]
    s.calls += 1
    p = await s.c.get_prompt("write_product_comparison", {"product_ids": ",".join(ids), "audience": "a sim racer"})
    text = p.messages[0].content.text
    expect(all(i in text for i in ids) and "sim racer" in text, "prompt missing arguments")

async def t28(s):  # bad id format -> clean validation error
    msg = await s.tool("get_product", {"product_id": "abc"}, ok=False)
    expect("product_id" in msg, "error does not name the field")

async def t29(s):  # unknown category -> error that lists valid ones
    msg = await s.tool("search_products", {"query": "x", "category": "Books"}, ok=False)
    expect("Appliances" in msg and "Video_Games" in msg, "error does not list valid categories")

async def t30(s):  # typo in an argument name -> rejected, not silently ignored
    msg = await s.tool("search_products", {"query": "ice maker", "max_prise": 20}, ok=False)
    expect("max_prise" in msg, "unknown argument not reported")


TASKS = [t01, t02, t03, t04, t05, t06, t07, t08, t09, t10, t11, t12, t13, t14, t15,
         t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30]


async def run(client) -> dict:
    results = []
    async with client as c:
        await c.call_tool("search_products", {"query": "warm up", "limit": 1})
        for task in TASKS:
            s = Session(c)
            t = time.perf_counter()
            try:
                await task(s)
                ok, err = True, None
            except Check as e:
                ok, err = False, str(e)
            except Exception as e:  # noqa: BLE001 - a protocol failure is a failed task
                ok, err = False, f"{type(e).__name__}: {e}"
            results.append({"task": task.__name__, "doc": (task.__doc__ or "").strip(), "ok": ok, "error": err,
                            "tool_calls": s.calls, "seconds": round(time.perf_counter() - t, 3)})
    return {"tasks": len(results), "passed": sum(r["ok"] for r in results),
            "tool_calls": sum(r["tool_calls"] for r in results), "results": results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    ap.add_argument("--url", default="http://127.0.0.1:8000/mcp")
    ap.add_argument("--out", default=str(ROOT / "docs"))
    args = ap.parse_args()
    client = (stdio_client(args.data_dir) if args.transport == "stdio"
              else http_client(args.url, os.environ.get("CATALOG_MCP_TOKEN")))
    report = asyncio.run(run(client))
    Path(args.out).mkdir(exist_ok=True)
    (Path(args.out) / f"client_tasks_{args.transport}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in report["results"]:
        print(f"{'PASS' if r['ok'] else 'FAIL'} {r['task']} ({r['tool_calls']} calls, {r['seconds']}s)"
              + (f"  <- {r['error']}" if r["error"] else ""))
    print(f"\n{report['passed']}/{report['tasks']} tasks passed "
          f"({report['passed'] / report['tasks']:.1%}), {report['tool_calls']} MCP calls, transport={args.transport}")


if __name__ == "__main__":
    main()
