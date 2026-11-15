"""Search quality: capped recall@10 and MRR@10 for keyword vs semantic vs hybrid.

Relevance labels come from SQL rules in eval/search_queries.json (no retriever output is
used to write them). Capped recall@10 = |relevant in top 10| / min(10, |relevant|), so a
query with 500 relevant items is judged on whether the top 10 are all relevant.

    uv run python scripts/eval_search.py --data-dir data
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from catalog_mcp.catalog import Catalog

ROOT = Path(__file__).resolve().parents[1]
MODES = ("keyword", "semantic", "hybrid")
K = 10


def relevant_sets(cat: Catalog, queries: list[dict]) -> dict[str, set[str]]:
    con = cat.con
    con.execute(
        """CREATE OR REPLACE TEMP VIEW labeled AS
           SELECT product_id, title, brand, category, category_path,
                  regexp_replace(upper(title || ' ' || CAST(details AS VARCHAR)), '[^A-Z0-9]', '', 'g') AS norm
           FROM products"""
    )
    out = {}
    for q in queries:
        rows = con.execute(f"SELECT product_id FROM labeled WHERE {q['where']}").fetchall()
        out[q["id"]] = {r[0] for r in rows}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--queries", default=str(ROOT / "eval" / "search_queries.json"))
    ap.add_argument("--out", default=str(ROOT / "docs"))
    args = ap.parse_args()

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))["queries"]
    cat = Catalog(args.data_dir)
    cat.warm_up()
    rel = relevant_sets(cat, queries)
    empty = [q["id"] for q in queries if not rel[q["id"]]]
    if empty:
        raise SystemExit(f"queries with no relevant items (fix the rule): {empty}")

    per_query = []
    for q in queries:
        R = rel[q["id"]]
        row = {"id": q["id"], "type": q["type"], "query": q["query"], "relevant": len(R)}
        for mode in MODES:
            top = [pid for pid, _ in cat.search(q["query"], mode, K)]
            hits = [pid in R for pid in top]
            row[f"{mode}_recall"] = sum(hits) / min(K, len(R))
            row[f"{mode}_mrr"] = next((1 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
        per_query.append(row)

    groups = defaultdict(list)
    for r in per_query:
        groups[r["type"]].append(r)
        groups["all"].append(r)
    summary = {
        g: {m: {"recall@10": statistics.mean(r[f"{m}_recall"] for r in rows),
                "mrr@10": statistics.mean(r[f"{m}_mrr"] for r in rows)} for m in MODES} | {"n": len(rows)}
        for g, rows in groups.items()
    }

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "search_eval.json").write_text(json.dumps({"summary": summary, "per_query": per_query}, indent=2), encoding="utf-8")

    order = ["exact", "keyword", "paraphrase", "all"]
    lines = ["| Query type | n | Keyword (BM25) | Semantic (bge-small) | Hybrid (RRF) |", "|---|---|---|---|---|"]
    for g in order:
        s = summary[g]
        best = max(MODES, key=lambda m: s[m]["recall@10"])
        cells = [f"**{s[m]['recall@10']:.3f}**" if m == best else f"{s[m]['recall@10']:.3f}" for m in MODES]
        lines.append(f"| {g} | {s['n']} | " + " | ".join(cells) + " |")
    lines += ["", "MRR@10:", "", "| Query type | Keyword | Semantic | Hybrid |", "|---|---|---|---|"]
    for g in order:
        s = summary[g]
        lines.append(f"| {g} | " + " | ".join(f"{s[m]['mrr@10']:.3f}" for m in MODES) + " |")
    table = "\n".join(lines)
    (out / "search_eval.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print("\nrelevant-set sizes: min", min(r["relevant"] for r in per_query),
          "median", statistics.median(r["relevant"] for r in per_query),
          "max", max(r["relevant"] for r in per_query))
    worst = sorted(per_query, key=lambda r: r["hybrid_recall"])[:8]
    print("\nlowest hybrid recall:")
    for r in worst:
        print(f"  {r['id']} {r['query'][:55]:55} |R|={r['relevant']:5}  kw={r['keyword_recall']:.1f} "
              f"sem={r['semantic_recall']:.1f} hyb={r['hybrid_recall']:.1f}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    width = 0.26
    colors = {"keyword": "#8a8f98", "semantic": "#3b7dd8", "hybrid": "#1f9e74"}
    for i, m in enumerate(MODES):
        vals = [summary[g][m]["recall@10"] for g in order]
        bars = ax.bar([x + (i - 1) * width for x in range(len(order))], vals, width, label=m, color=colors[m])
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(range(len(order)), [f"{g}\n(n={summary[g]['n']})" for g in order])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("capped recall@10")
    ax.set_title("Search recall@10 by query type")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "search_recall.png", dpi=160)


if __name__ == "__main__":
    main()
