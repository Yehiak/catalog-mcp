"""Read side of the catalog: keyword / semantic / hybrid search, lookups, comparison, dedup.

This module knows nothing about MCP. The server (server.py) is a thin layer on top of it,
which keeps the logic easy to test and to benchmark without a client.
"""

from __future__ import annotations

import atexit
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import faiss
import numpy as np

from .config import paths as resolve_paths
from .embed import Embedder, SentenceTransformerEmbedder, load_index
from .ingest import connect

SearchMode = Literal["keyword", "semantic", "hybrid"]

RRF_K = 60  # standard reciprocal-rank-fusion constant
HYBRID_POOL = 50  # candidates taken from each retriever before fusion

SUMMARY_COLUMNS = "product_id, title, brand, store, category, price, average_rating, rating_number"


class CatalogNotBuilt(RuntimeError):
    pass


class ProductNotFound(KeyError):
    pass


@dataclass(frozen=True)
class Filters:
    category: str | None = None
    brand: str | None = None
    min_price: float | None = None
    max_price: float | None = None
    min_rating: float | None = None

    def is_empty(self) -> bool:
        return all(v is None for v in (self.category, self.brand, self.min_price, self.max_price, self.min_rating))

    def sql(self) -> tuple[str, list[Any]]:
        clauses, params = [], []
        if self.category is not None:
            clauses.append("category = ?")
            params.append(self.category)
        if self.brand is not None:
            clauses.append("(brand ILIKE ? OR store ILIKE ?)")
            params += [f"%{self.brand}%", f"%{self.brand}%"]
        if self.min_price is not None:
            clauses.append("price >= ?")
            params.append(self.min_price)
        if self.max_price is not None:
            clauses.append("price <= ?")
            params.append(self.max_price)
        if self.min_rating is not None:
            clauses.append("average_rating >= ?")
            params.append(self.min_rating)
        return (" AND ".join(clauses) or "TRUE"), params


# --- text signals used by near-duplicate detection ---------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")
# Part / model numbers: codes with at least one digit and one letter (W10295370A,
# DA97-07549B) or long pure-digit codes (99003147). Wattages like 1500W are dropped below.
_MODEL_NO = re.compile(r"\b(?:(?=[A-Z0-9\-]*\d)(?=[A-Z0-9\-]*[A-Z])[A-Z0-9][A-Z0-9\-]{3,}|\d{7,})\b")
_UNIT = re.compile(r"^\d+(W|V|MM|CM|IN|LB|LBS|OZ|GB|TB|MB|MAH|HZ|FT|QT|PCS|PACK)$")
_STOP = frozenset("a an and the for of with to in on by from new pack set compatible replacement".split())
_PLATFORMS = {
    "ps5": r"\b(ps5|playstation\s*5)\b",
    "ps4": r"\b(ps4|playstation\s*4)\b",
    "ps3": r"\b(ps3|playstation\s*3)\b",
    "xbox-series": r"\bxbox\s*series\s*[xs]\b",
    "xbox-one": r"\bxbox\s*one\b",
    "xbox-360": r"\bxbox\s*360\b",
    "switch": r"\b(nintendo\s*)?switch\b",
    "3ds": r"\b3ds\b",
    "wii-u": r"\bwii\s*u\b",
    "pc": r"\b(pc|windows)\b",
}
_PLATFORM_RE = {k: re.compile(v, re.I) for k, v in _PLATFORMS.items()}


def title_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def model_numbers(*texts: str | None) -> set[str]:
    found: set[str] = set()
    for t in texts:
        if t:
            found |= {m.replace("-", "") for m in _MODEL_NO.findall(t.upper())}
    return {m for m in found if len(m) >= 5 and not _UNIT.match(m)}


def platforms(text: str) -> set[str]:
    found = {k for k, rx in _PLATFORM_RE.items() if rx.search(text)}
    if "xbox-series" in found:
        found.discard("xbox-one")
    return found


def flatten_details(raw: str | dict | None, max_len: int = 200) -> dict[str, str]:
    """Amazon `details` values can be strings, numbers or nested dicts; make them flat strings."""
    if not raw:
        return {}
    data = json.loads(raw) if isinstance(raw, str) else raw
    out: dict[str, str] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = "; ".join(f"{ik}: {iv}" for ik, iv in v.items())
        elif isinstance(v, list):
            v = "; ".join(map(str, v))
        s = str(v).strip()
        if s:
            out[str(k).strip()] = s[:max_len]
    return out


# --- the catalog -------------------------------------------------------------------------


class Catalog:
    def __init__(self, data_dir: str | Path | None = None, embedder: Embedder | None = None):
        self.paths = resolve_paths(data_dir)
        if not self.paths.exists():
            raise CatalogNotBuilt(
                f"No catalog found in {self.paths.root}. Build one first with `catalog-mcp build`."
            )
        self.info = json.loads(self.paths.build_info.read_text(encoding="utf-8"))
        self._con = connect(self.paths.db, read_only=True)
        self._con.execute("SET enable_progress_bar = false")
        self.index = load_index(self.paths.index)
        self.embedder = embedder or SentenceTransformerEmbedder(self.info["embedding_model"])
        ids = self._con.execute("SELECT product_id FROM products ORDER BY vec_id").fetchall()
        self.ids = np.array([r[0] for r in ids], dtype=object)
        self.vec_of = {pid: i for i, pid in enumerate(self.ids)}
        self.categories: tuple[str, ...] = tuple(self.info["categories"])
        self._local = threading.local()
        self._sel_cache: dict[Filters, Any] = {}
        self._sel_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="catalog")
        # Close DuckDB before interpreter teardown (avoids noisy leak reports at exit).
        atexit.register(self.close)

    # DuckDB connections are not thread-safe; give each worker thread its own cursor.
    @property
    def con(self):
        cur = getattr(self._local, "cur", None)
        if cur is None:
            cur = self._con.cursor()
            cur.execute("SET enable_progress_bar = false")
            self._local.cur = cur
        return cur

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._con.close()

    def warm_up(self) -> None:
        """Load the embedding model now so the first semantic query is not slow."""
        self.embedder.encode_query("warm up")

    def __len__(self) -> int:
        return len(self.ids)

    # ---- search ----

    def keyword(self, query: str, k: int = 10, filters: Filters = Filters()) -> list[tuple[str, float]]:
        where, params = filters.sql()
        # Also match the hyphen-free form of any part number in the query (see ingest.add_codes).
        codes = model_numbers(query)
        if codes:
            query = f"{query} {' '.join(sorted(codes)).lower()}"
        rows = self.con.execute(
            f"""
            SELECT product_id, score FROM (
                SELECT product_id, fts_main_products.match_bm25(product_id, ?) AS score
                FROM products WHERE {where}
            ) WHERE score IS NOT NULL
            ORDER BY score DESC LIMIT ?
            """,
            [query, *params, k],
        ).fetchall()
        return [(pid, float(s)) for pid, s in rows]

    def _allowed(self, filters: Filters):
        """FAISS selector restricting vector search to rows that pass the SQL filters.

        Returns None (no filter), False (nothing passes) or a cached IDSelector.
        """
        if filters.is_empty():
            return None
        with self._sel_lock:
            if filters not in self._sel_cache:
                where, params = filters.sql()
                ids = self.con.execute(f"SELECT vec_id FROM products WHERE {where}", params).fetchnumpy()["vec_id"]
                if len(self._sel_cache) >= 64:
                    self._sel_cache.pop(next(iter(self._sel_cache)))
                self._sel_cache[filters] = faiss.IDSelectorBatch(ids.astype(np.int64)) if len(ids) else False
            return self._sel_cache[filters]

    def _vector_search(self, vec: np.ndarray, k: int, filters: Filters, exclude: str | None = None):
        sel = self._allowed(filters)
        if sel is False:
            return []
        extra = 1 if exclude else 0
        params = faiss.SearchParameters(sel=sel) if sel is not None else None
        scores, idx = self.index.search(vec.reshape(1, -1), k + extra, params=params)
        out = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            pid = self.ids[i]
            if pid != exclude:
                out.append((pid, float(s)))
        return out[:k]

    def semantic(self, query: str, k: int = 10, filters: Filters = Filters()) -> list[tuple[str, float]]:
        return self._vector_search(self.embedder.encode_query(query), k, filters)

    def hybrid(self, query: str, k: int = 10, filters: Filters = Filters()) -> list[tuple[str, float]]:
        pool = max(HYBRID_POOL, k)
        # The two retrievers are independent (DuckDB and FAISS/torch both release the GIL),
        # so run them side by side: latency is max(keyword, semantic), not the sum.
        semantic = self._pool.submit(self.semantic, query, pool, filters)
        keyword = self.keyword(query, pool, filters)
        fused: dict[str, float] = {}
        for ranked in (keyword, semantic.result()):
            for rank, (pid, _) in enumerate(ranked):
                fused[pid] = fused.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)
        return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]

    def search(self, query: str, mode: SearchMode = "hybrid", k: int = 10, filters: Filters = Filters()):
        fn = {"keyword": self.keyword, "semantic": self.semantic, "hybrid": self.hybrid}[mode]
        return fn(query, k, filters)

    # ---- lookups ----

    def summaries(self, product_ids: Iterable[str]) -> list[dict]:
        ids = list(product_ids)
        if not ids:
            return []
        rows = self.con.execute(
            f"SELECT {SUMMARY_COLUMNS} FROM products WHERE product_id IN (SELECT unnest(?))", [ids]
        ).fetchall()
        cols = [c.strip() for c in SUMMARY_COLUMNS.split(",")]
        by_id = {r[0]: dict(zip(cols, r)) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def get(self, product_id: str) -> dict:
        cur = self.con.execute(
            "SELECT product_id, title, brand, store, category, main_category, price, average_rating, "
            "rating_number, features, description, categories, details, image_url FROM products "
            "WHERE product_id = ?",
            [product_id],
        )
        row = cur.fetchone()
        if row is None:
            raise ProductNotFound(product_id)
        cols = [d[0] for d in cur.description]
        product = dict(zip(cols, row))
        product["details"] = flatten_details(product["details"])
        return product

    def exists(self, product_id: str) -> bool:
        return product_id in self.vec_of

    # ---- comparison ----

    def compare(self, product_ids: list[str], max_specs: int = 25) -> dict:
        products = [self.get(pid) for pid in product_ids]
        n = len(products)
        rows: list[dict] = []
        for label, key in (("Brand", "brand"), ("Store", "store"), ("Price (USD)", "price"),
                           ("Average rating", "average_rating"), ("Number of ratings", "rating_number")):
            rows.append({"name": label, "values": [_fmt(p[key]) for p in products]})

        skip = {"Best Sellers Rank", "Date First Available", "Is Discontinued By Manufacturer", "Pricing"}
        counts: dict[str, int] = {}
        for p in products:
            for key in p["details"]:
                if key not in skip:
                    counts[key] = counts.get(key, 0) + 1
        shared = [k for k, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if c >= min(2, n)]
        for key in shared[:max_specs]:
            rows.append({"name": key, "values": [p["details"].get(key) for p in products]})

        priced = [p for p in products if p["price"] is not None]
        rated = [p for p in products if p["average_rating"] is not None]
        return {
            "products": [{k: p[k] for k in SUMMARY_COLUMNS.replace(" ", "").split(",")} for p in products],
            "specs": rows,
            "cheapest_id": min(priced, key=lambda p: p["price"])["product_id"] if priced else None,
            "best_rated_id": max(rated, key=lambda p: _bayes_rating(p))["product_id"] if rated else None,
            "most_reviewed_id": max(products, key=lambda p: p["rating_number"] or 0)["product_id"],
        }

    # ---- near-duplicates ----

    def near_duplicates(
        self,
        product_id: str | None = None,
        title: str | None = None,
        k: int = 10,
        min_similarity: float = 0.85,
        same_category: bool = True,
    ) -> dict:
        if (product_id is None) == (title is None):
            raise ValueError("Give exactly one of product_id or title.")
        if product_id is not None:
            if product_id not in self.vec_of:
                raise ProductNotFound(product_id)
            base = self.get(product_id)
            vec = self.index.reconstruct(int(self.vec_of[product_id]))
            base_title, base_details, category = base["title"], base["details"], base["category"]
        else:
            vec = self.embedder.encode_docs([title])[0]
            base_title, base_details, category = title, {}, None
        filters = Filters(category=category if same_category else None)
        hits = self._vector_search(np.asarray(vec, dtype=np.float32), max(k * 3, 30), filters, exclude=product_id)

        base_tokens = title_tokens(base_title)
        base_models = model_numbers(base_title, base_details.get("Part Number"), base_details.get("Item model number"))
        base_platforms = platforms(base_title)
        cands = [(pid, s) for pid, s in hits if s >= min_similarity]
        details = {pid: self.get(pid) for pid, _ in cands}
        out = []
        for pid, sim in cands:
            p = details[pid]
            models = model_numbers(p["title"], p["details"].get("Part Number"), p["details"].get("Item model number"))
            shared_models = sorted(base_models & models)
            plats = platforms(p["title"])
            platform_conflict = bool(base_platforms and plats and not (base_platforms & plats))
            overlap = jaccard(base_tokens, title_tokens(p["title"]))
            if platform_conflict:
                verdict = "variant"
            elif shared_models:
                verdict = "likely_duplicate"
            elif sim >= 0.95 and overlap >= 0.6:
                verdict = "likely_duplicate"
            else:
                verdict = "possible_duplicate"
            out.append({
                **{c: p[c] for c in SUMMARY_COLUMNS.replace(" ", "").split(",")},
                "similarity": round(sim, 4),
                "title_overlap": round(overlap, 3),
                "shared_model_numbers": shared_models,
                "same_store": bool(p["store"] and base.get("store") == p["store"]) if product_id else False,
                "platform_conflict": platform_conflict,
                "verdict": verdict,
            })
        order = {"likely_duplicate": 0, "possible_duplicate": 1, "variant": 2}
        out.sort(key=lambda d: (order[d["verdict"]], -d["similarity"]))
        return {"query_product_id": product_id, "query_title": base_title, "candidates": out[:k]}

    # ---- stats (MCP resources) ----

    def overview(self) -> dict:
        return {
            "dataset": self.info["dataset"],
            "products": self.info["products"],
            "categories": self.info["products_per_category"],
            "embedding_model": self.info["embedding_model"],
            "built_at": self.info["built_at"],
        }

    def category_stats(self, category: str) -> dict:
        c = self.con
        n, with_price, p10, p50, p90, avg_rating, med_reviews = c.execute(
            """SELECT count(*), count(price), quantile_cont(price, 0.1), quantile_cont(price, 0.5),
                      quantile_cont(price, 0.9), avg(average_rating), median(rating_number)
               FROM products WHERE category = ?""",
            [category],
        ).fetchone()
        hist = c.execute(
            """SELECT CAST(least(floor(average_rating), 4) AS INT) AS b, count(*) FROM products
               WHERE category = ? AND average_rating IS NOT NULL GROUP BY 1 ORDER BY 1""",
            [category],
        ).fetchall()
        labels = {1: "1.0-1.9", 2: "2.0-2.9", 3: "3.0-3.9", 4: "4.0-5.0", 0: "below 1.0"}
        brands = c.execute(
            "SELECT brand, count(*) FROM products WHERE category = ? AND brand IS NOT NULL "
            "GROUP BY 1 ORDER BY 2 DESC LIMIT 10",
            [category],
        ).fetchall()
        subcats = c.execute(
            "SELECT categories[2], count(*) FROM products WHERE category = ? AND len(categories) >= 2 "
            "GROUP BY 1 ORDER BY 2 DESC LIMIT 10",
            [category],
        ).fetchall()
        r = lambda x, d=2: None if x is None else round(float(x), d)  # noqa: E731
        return {
            "category": category,
            "products": n,
            "with_price": with_price,
            "price_usd": {"p10": r(p10), "median": r(p50), "p90": r(p90)},
            "average_rating": r(avg_rating, 3),
            "median_ratings_per_product": r(med_reviews, 1),
            "rating_histogram": {labels[b]: cnt for b, cnt in hist},
            "top_brands": [{"brand": b, "products": cnt} for b, cnt in brands],
            "top_subcategories": [{"subcategory": s, "products": cnt} for s, cnt in subcats],
        }


def _fmt(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, float):
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return str(v)


def _bayes_rating(p: dict, prior: float = 4.0, weight: int = 20) -> float:
    """Average rating shrunk toward a prior, so 5.0 from 2 ratings doesn't beat 4.7 from 2,000."""
    n = p["rating_number"] or 0
    return (prior * weight + (p["average_rating"] or 0) * n) / (weight + n)
