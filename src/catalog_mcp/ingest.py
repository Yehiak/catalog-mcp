"""Build the catalog: raw JSONL -> DuckDB table + full-text index -> embeddings -> FAISS."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import duckdb

from .config import DATASET_REPO, DEFAULT_MODEL, Paths
from .embed import Embedder, SentenceTransformerEmbedder, build_index, save_index

RAW_COLUMNS = {
    "main_category": "VARCHAR",
    "title": "VARCHAR",
    "average_rating": "DOUBLE",
    "rating_number": "BIGINT",
    "features": "VARCHAR[]",
    "description": "VARCHAR[]",
    "price": "VARCHAR",
    "images": "JSON",
    "store": "VARCHAR",
    "categories": "VARCHAR[]",
    "details": "JSON",
    "parent_asin": "VARCHAR",
}

# Columns covered by the BM25 index. DuckDB scores all listed fields together.
FTS_FIELDS = ("title", "brand", "codes", "category_path", "features_text", "description_text")
# DuckDB's default tokenizer drops every non-letter, which deletes part numbers like
# W10295370A. Keep digits.
FTS_IGNORE = r"(\.|[^a-z0-9])+"


def download_category(category: str, raw_dir: Path) -> Path:
    """Fetch one category's item metadata from the Hugging Face Hub (cached on re-runs)."""
    from huggingface_hub import hf_hub_download

    filename = f"raw/meta_categories/meta_{category}.jsonl"
    return Path(hf_hub_download(DATASET_REPO, filename, repo_type="dataset", local_dir=raw_dir))


def connect(db_path: Path | str, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(db_path), read_only=read_only)
    con.execute("INSTALL fts")
    con.execute("LOAD fts")
    return con


def load_products(con: duckdb.DuckDBPyConnection, files: dict[str, Path]) -> int:
    """Create the `products` table from one JSONL file per category.

    Cleaning happens in a TEMP table, then `products` is written once: rewriting a table
    in place leaves dead blocks in the DuckDB file and roughly doubles its size.
    """
    from .catalog import model_numbers

    columns = "{" + ", ".join(f"'{k}': '{v}'" for k, v in RAW_COLUMNS.items()) + "}"
    selects = []
    for category, path in files.items():
        src = str(path).replace("'", "''")
        selects.append(
            f"SELECT '{category}' AS category, * FROM read_json('{src}', format='newline_delimited', "
            f"columns={columns}, ignore_errors=true, maximum_object_size=67108864)"
        )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE staged AS
        WITH raw AS ({' UNION ALL '.join(selects)})
        SELECT
            parent_asin AS product_id,
            category,
            NULLIF(trim(main_category), '') AS main_category,
            trim(regexp_replace(title, '\\s+', ' ', 'g')) AS title,
            NULLIF(trim(store), '') AS store,
            NULLIF(trim(coalesce(details->>'$.Brand', details->>'$.Manufacturer', store)), '') AS brand,
            CASE WHEN TRY_CAST(price AS DOUBLE) > 0 THEN TRY_CAST(price AS DOUBLE) END AS price,
            average_rating,
            coalesce(rating_number, 0) AS rating_number,
            coalesce(features, []) AS features,
            coalesce(description, []) AS description,
            coalesce(categories, []) AS categories,
            array_to_string(coalesce(categories, []), ' > ') AS category_path,
            array_to_string(coalesce(features, []), ' ') AS features_text,
            array_to_string(coalesce(description, []), ' ') AS description_text,
            coalesce(details, '{{}}') AS details,
            images->>'$[0].large' AS image_url
        FROM raw
        WHERE parent_asin IS NOT NULL AND title IS NOT NULL AND length(trim(title)) > 0
        QUALIFY row_number() OVER (PARTITION BY parent_asin ORDER BY rating_number DESC) = 1
        """
    )

    # `codes`: part/model numbers with hyphens removed, so 'DA29-00020B' also matches 'DA2900020B'.
    def codes(title: str | None, part: str | None, model: str | None) -> str:
        return " ".join(sorted(model_numbers(title, part, model))).lower()

    # null_handling="special": still call the function when some arguments are NULL.
    con.create_function("part_codes", codes, ["VARCHAR", "VARCHAR", "VARCHAR"], "VARCHAR", null_handling="special")
    con.execute("DROP TABLE IF EXISTS products")
    # vec_id = row position in the FAISS index (0..n-1, dense, stable order).
    con.execute(
        """CREATE TABLE products AS
           SELECT row_number() OVER (ORDER BY product_id) - 1 AS vec_id, *,
                  part_codes(title, details->>'$."Part Number"', details->>'$."Item model number"') AS codes
           FROM staged ORDER BY product_id"""
    )
    con.remove_function("part_codes")
    con.execute("DROP TABLE staged")
    con.execute("CREATE UNIQUE INDEX products_pk ON products(product_id)")
    return con.execute("SELECT count(*) FROM products").fetchone()[0]


def create_fts(con: duckdb.DuckDBPyConnection) -> None:
    fields = ", ".join(f"'{f}'" for f in FTS_FIELDS)
    con.execute(
        f"PRAGMA create_fts_index('products', 'product_id', {fields}, "
        f"stemmer='porter', stopwords='english', ignore='{FTS_IGNORE}', lower=1, strip_accents=1, overwrite=1)"
    )


def embedding_text(title: str, brand: str | None, category_path: str, features: Sequence[str]) -> str:
    """What gets embedded for each product: the parts a shopper would describe."""
    parts = [title]
    if brand:
        parts.append(f"Brand: {brand}.")
    if category_path:
        parts.append(f"Category: {category_path}.")
    if features:
        parts.append(" ".join(f[:200] for f in features[:3]))
    return " ".join(parts)[:1200]


def build(
    paths: Paths,
    categories: Sequence[str],
    *,
    files: dict[str, Path] | None = None,
    embedder: Embedder | None = None,
    batch_size: int = 512,
    log: Callable[[str], None] = print,
) -> dict:
    """Run the whole pipeline and write build_info.json. Returns the build info."""
    paths.root.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    t = time.perf_counter()
    if files is None:
        files = {}
        for c in categories:
            log(f"downloading {c} ...")
            files[c] = download_category(c, paths.raw)
    timings["download_s"] = time.perf_counter() - t

    tmp_db = paths.db.with_suffix(".building.duckdb")
    tmp_db.unlink(missing_ok=True)
    con = connect(tmp_db)
    try:
        t = time.perf_counter()
        n = load_products(con, files)
        timings["load_s"] = time.perf_counter() - t
        log(f"loaded {n:,} products in {timings['load_s']:.1f}s")

        t = time.perf_counter()
        create_fts(con)
        timings["fts_s"] = time.perf_counter() - t
        log(f"built full-text index in {timings['fts_s']:.1f}s")

        rows = con.execute(
            "SELECT title, brand, category_path, features FROM products ORDER BY vec_id"
        ).fetchall()
        texts = [embedding_text(*r) for r in rows]

        embedder = embedder or SentenceTransformerEmbedder(DEFAULT_MODEL)
        t = time.perf_counter()
        vectors = embedder.encode_docs(texts, batch_size=batch_size, progress=True)
        timings["embed_s"] = time.perf_counter() - t
        log(f"embedded {len(texts):,} products in {timings['embed_s']:.1f}s")

        per_category = dict(con.execute("SELECT category, count(*) FROM products GROUP BY 1 ORDER BY 1").fetchall())
    finally:
        con.close()

    index = build_index(vectors)
    tmp_index = paths.index.with_suffix(".building.faiss")
    save_index(index, tmp_index)

    # Swap the finished files in only once everything succeeded.
    for final, tmp in ((paths.db, tmp_db), (paths.index, tmp_index)):
        final.unlink(missing_ok=True)
        tmp.replace(final)

    info = {
        "dataset": DATASET_REPO,
        "categories": list(files),
        "products": n,
        "products_per_category": per_category,
        "embedding_model": embedder.name,
        "embedding_dim": int(vectors.shape[1]),
        "index_type": type(index).__name__,
        "db_bytes": paths.db.stat().st_size,
        "index_bytes": paths.index.stat().st_size,
        "timings": {k: round(v, 2) for k, v in timings.items()},
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    paths.build_info.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
