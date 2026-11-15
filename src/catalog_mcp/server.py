"""The MCP server: tools, resources and prompts on top of `Catalog`.

- Tools are actions the model decides to call (search, look up, compare, dedupe).
- Resources are read-only documents the client app can fetch by URI (catalog stats).
- Prompts are reusable message templates the user picks (write a product comparison).
"""

from __future__ import annotations

import inspect
import json
import logging
import threading
from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError, ToolError
from mcp.server.mcpserver.tools import Tool
from mcp.types import ToolAnnotations
from pydantic import ConfigDict, Field

from . import __version__
from .catalog import Catalog, CatalogNotBuilt, Filters, ProductNotFound
from .models import Comparison, Duplicates, Product, SearchResult

log = logging.getLogger(__name__)

INSTRUCTIONS = """\
Product catalog built from Amazon Reviews 2023 item metadata (McAuley Lab).
Typical flow: search_products -> get_product for details -> compare_products for a side-by-side
table. Use find_near_duplicates to spot repeated listings of the same item. Product ids are
Amazon parent ASINs (10 characters, e.g. B07XYZ1234). Read catalog://stats for what is loaded."""

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)

ProductId = Annotated[str, Field(min_length=10, max_length=10, pattern=r"^[A-Z0-9]{10}$", description="Amazon parent ASIN, e.g. 'B07XYZ1234'.")]


class _LazyCatalog:
    """Opens the catalog on first use, so the server starts (and lists its tools) instantly."""

    def __init__(self, data_dir=None, catalog: Catalog | None = None):
        self.data_dir = data_dir
        self._catalog = catalog
        self._lock = threading.Lock()

    def get(self) -> Catalog:
        if self._catalog is None:
            with self._lock:
                if self._catalog is None:
                    try:
                        self._catalog = Catalog(self.data_dir)
                    except CatalogNotBuilt as e:
                        raise ToolError(str(e)) from e
        return self._catalog


def strict_tool(fn, **kwargs) -> Tool:
    """Register `fn` as a tool whose arguments reject unknown keys.

    By default the SDK ignores arguments it doesn't know, so a misspelled `max_prise`
    would silently do nothing. Forbidding extras turns that into a clear error the
    model can correct, and the schema advertises it with additionalProperties: false.
    """
    tool = Tool.from_function(fn, description=inspect.cleandoc(fn.__doc__ or ""), annotations=READ_ONLY, **kwargs)
    base = tool.fn_metadata.arg_model

    class StrictArgs(base):  # type: ignore[misc, valid-type]
        model_config = ConfigDict(**base.model_config, extra="forbid")

    StrictArgs.__name__ = base.__name__
    tool.fn_metadata.arg_model = StrictArgs
    tool.parameters = {**tool.parameters, "additionalProperties": False}
    return tool


def create_server(data_dir=None, catalog: Catalog | None = None, warm_up: bool = False) -> MCPServer:
    lazy = _LazyCatalog(data_dir, catalog)
    tools: list[Tool] = []

    def tool(**kwargs):
        def register(fn):
            tools.append(strict_tool(fn, **kwargs))
            return fn
        return register

    def check_category(cat: Catalog, category: str | None) -> None:
        if category is not None and category not in cat.categories:
            raise ToolError(f"Unknown category {category!r}. Loaded categories: {', '.join(cat.categories)}.")

    # ------------------------------------------------------------------ tools

    @tool(title="Search products")
    def search_products(
        query: Annotated[str, Field(min_length=1, max_length=300, description="What the user is looking for, in their own words or as keywords.")],
        mode: Annotated[Literal["hybrid", "semantic", "keyword"], Field(description="'hybrid' (default, best overall) fuses keyword and semantic ranks; 'keyword' is BM25 full-text, best for exact model or part numbers; 'semantic' matches meaning, best for descriptive queries.")] = "hybrid",
        limit: Annotated[int, Field(ge=1, le=50, description="Number of results (1-50).")] = 10,
        category: Annotated[str | None, Field(description="Restrict to one category, e.g. 'Appliances' or 'Video_Games'. See catalog://stats.")] = None,
        brand: Annotated[str | None, Field(min_length=1, max_length=80, description="Case-insensitive substring of the brand or store name.")] = None,
        min_price: Annotated[float | None, Field(ge=0, description="Minimum price in USD. Products without a price are excluded when set.")] = None,
        max_price: Annotated[float | None, Field(ge=0, description="Maximum price in USD. Products without a price are excluded when set.")] = None,
        min_rating: Annotated[float | None, Field(ge=0, le=5, description="Minimum average star rating (0-5).")] = None,
    ) -> SearchResult:
        """Search the product catalog and return the best matches with id, title, brand, price and rating.

        Use this first to find product ids. Filters are applied before ranking."""
        cat = lazy.get()
        check_category(cat, category)
        if min_price is not None and max_price is not None and min_price > max_price:
            raise ToolError("min_price must be less than or equal to max_price.")
        filters = Filters(category=category, brand=brand, min_price=min_price, max_price=max_price, min_rating=min_rating)
        ranked = cat.search(query, mode, limit, filters)
        scores = dict(ranked)
        hits = [{**s, "score": round(scores[s["product_id"]], 6)} for s in cat.summaries(pid for pid, _ in ranked)]
        return SearchResult(query=query, mode=mode, count=len(hits), results=hits)

    @tool(title="Get product")
    def get_product(
        product_id: ProductId,
        max_description_chars: Annotated[int, Field(ge=0, le=20000, description="Truncate the description to this many characters (0 = omit).")] = 2000,
    ) -> Product:
        """Get the full record for one product: features, description, category path and spec table."""
        cat = lazy.get()
        try:
            p = cat.get(product_id)
        except ProductNotFound:
            raise ToolError(f"No product with id {product_id!r}. Use search_products to find valid ids.") from None
        desc = " ".join(p.pop("description") or [])
        return Product(**p, description=desc[:max_description_chars])

    @tool(title="Compare products")
    def compare_products(
        product_ids: Annotated[list[ProductId], Field(min_length=2, max_length=6, description="2 to 6 distinct product ids to compare.")],
        max_specs: Annotated[int, Field(ge=0, le=60, description="Maximum number of spec rows beyond the core fields.")] = 25,
    ) -> Comparison:
        """Compare 2-6 products side by side: price, rating, brand and every spec that at least two of them list.

        Also names the cheapest, best-rated (adjusted for rating count) and most-reviewed product."""
        if len(set(product_ids)) != len(product_ids):
            raise ToolError("product_ids contains duplicates; give distinct ids.")
        cat = lazy.get()
        missing = [p for p in product_ids if not cat.exists(p)]
        if missing:
            raise ToolError(f"Unknown product ids: {', '.join(missing)}. Use search_products to find valid ids.")
        return Comparison(**cat.compare(product_ids, max_specs=max_specs))

    @tool(title="Find near-duplicate listings")
    def find_near_duplicates(
        product_id: Annotated[ProductId | None, Field(description="Find listings that duplicate this product. Give this or `title`, not both.")] = None,
        title: Annotated[str | None, Field(min_length=3, max_length=500, description="A new listing title to check against the catalog before adding it. Give this or `product_id`.")] = None,
        limit: Annotated[int, Field(ge=1, le=50, description="Maximum number of candidates.")] = 10,
        min_similarity: Annotated[float, Field(ge=0.5, le=1.0, description="Cosine similarity cut-off (0.5-1.0). 0.85 is a good default; raise it for fewer, surer hits.")] = 0.85,
        same_category: Annotated[bool, Field(description="Only look in the query product's category.")] = True,
    ) -> Duplicates:
        """Find listings that are probably the same item as a product (or a draft title), e.g. one part sold by several sellers.

        Each candidate carries the evidence: embedding similarity, title word overlap, shared model or
        part numbers, and a platform check so a PS5 and a Switch edition of one game count as variants,
        not duplicates."""
        if (product_id is None) == (title is None):
            raise ToolError("Give exactly one of product_id or title.")
        cat = lazy.get()
        try:
            res = cat.near_duplicates(product_id=product_id, title=title, k=limit,
                                      min_similarity=min_similarity, same_category=same_category)
        except ProductNotFound:
            raise ToolError(f"No product with id {product_id!r}. Use search_products to find valid ids.") from None
        return Duplicates(**res)

    mcp = MCPServer(name="catalog-mcp", title="Product Catalog", version=__version__,
                    instructions=INSTRUCTIONS, tools=tools)

    # -------------------------------------------------------------- resources

    @mcp.resource("catalog://stats", title="Catalog overview", mime_type="application/json",
                  description="What is loaded: dataset, product count per category, embedding model, build time.")
    def catalog_stats() -> str:
        return json.dumps(lazy.get().overview(), indent=2)

    @mcp.resource("catalog://categories/{category}/stats", title="Category statistics", mime_type="application/json",
                  description="Statistics for one category: price percentiles, rating histogram, top brands and subcategories.")
    def category_stats(category: str) -> str:
        cat = lazy.get()
        if category not in cat.categories:
            raise ResourceNotFoundError(f"Unknown category {category!r}. Loaded: {', '.join(cat.categories)}.")
        return json.dumps(cat.category_stats(category), indent=2)

    # ---------------------------------------------------------------- prompts

    @mcp.prompt(title="Write a product comparison")
    def write_product_comparison(
        product_ids: Annotated[str, Field(description="Comma-separated product ids (2-6), e.g. 'B07XYZ1234, B08ABC5678'.")],
        audience: Annotated[str, Field(description="Who the write-up is for, e.g. 'first-time buyer' or 'appliance repair technician'.")] = "a general shopper",
        length: Annotated[Literal["short", "medium", "long"], Field(description="Target length: short (~150 words), medium (~350), long (~700).")] = "medium",
    ) -> str:
        """Write a buyer-friendly comparison of 2-6 catalog products, grounded in their real specs and ratings."""
        ids = [p.strip() for p in product_ids.split(",") if p.strip()]
        words = {"short": 150, "medium": 350, "long": 700}[length]
        return (
            f"Write a product comparison for {audience}, about {words} words.\n\n"
            f"Products: {', '.join(ids)}\n\n"
            "Steps:\n"
            f"1. Call compare_products with product_ids={json.dumps(ids)}.\n"
            "2. If a spec that matters for the decision is missing, call get_product for that product.\n"
            "3. Write the comparison:\n"
            "   - one-line verdict first (which to buy, for whom);\n"
            "   - a markdown table of the specs that actually differ;\n"
            "   - strengths and weaknesses of each product, citing price, average rating AND rating count;\n"
            "   - call out missing data (no price, few ratings) instead of guessing.\n"
            "Only use facts returned by the tools. Refer to products by short name and id."
        )

    if warm_up:
        def _warm():
            try:
                lazy.get().warm_up()
            except Exception:  # noqa: BLE001 - warm-up is best effort
                log.exception("warm-up failed")
        threading.Thread(target=_warm, daemon=True).start()

    return mcp
