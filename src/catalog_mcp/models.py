"""Pydantic schemas for tool outputs.

The MCP SDK turns these into each tool's `outputSchema`, and every result is validated
against them before it leaves the server. Clients get typed `structuredContent` plus a
JSON text copy for clients that only read text.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProductSummary(_Strict):
    product_id: str = Field(description="Amazon parent ASIN. Pass it to get_product, compare_products or find_near_duplicates.")
    title: str
    brand: str | None = Field(description="Brand from the product details, falling back to the store name.")
    store: str | None
    category: str = Field(description="Catalog category, e.g. 'Appliances' or 'Video_Games'.")
    price: float | None = Field(description="Price in USD when the listing had one (about half do).")
    average_rating: float | None = Field(ge=0, le=5)
    rating_number: int = Field(ge=0, description="How many ratings the average is based on.")


class SearchHit(ProductSummary):
    score: float = Field(description="Relevance score. BM25 for keyword, cosine for semantic, RRF for hybrid. Higher is better; only comparable within one result list.")


class SearchResult(_Strict):
    query: str
    mode: Literal["keyword", "semantic", "hybrid"]
    count: int
    results: list[SearchHit]


class Product(ProductSummary):
    main_category: str | None
    features: list[str] = Field(description="Bullet points from the listing.")
    description: str = Field(description="Listing description, truncated to max_description_chars.")
    categories: list[str] = Field(description="Category path from broad to narrow.")
    details: dict[str, str] = Field(description="Spec table from the listing (dimensions, model number, ...).")
    image_url: str | None


class SpecRow(_Strict):
    name: str
    values: list[str | None] = Field(description="One value per product, in the same order as `products`. null = not listed.")


class Comparison(_Strict):
    products: list[ProductSummary]
    specs: list[SpecRow] = Field(description="Core fields first, then spec keys listed by at least two of the products.")
    cheapest_id: str | None
    best_rated_id: str | None = Field(description="Highest rating after shrinking toward 4.0 by rating count, so a 5.0 from 2 ratings does not win.")
    most_reviewed_id: str


class DuplicateCandidate(ProductSummary):
    similarity: float = Field(description="Cosine similarity of the listing embeddings (0-1).")
    title_overlap: float = Field(description="Jaccard overlap of title words (0-1).")
    shared_model_numbers: list[str] = Field(description="Model or part numbers found in both listings.")
    same_store: bool
    platform_conflict: bool = Field(description="True when titles name different gaming platforms (e.g. PS5 vs Switch): a variant, not a duplicate.")
    verdict: Literal["likely_duplicate", "possible_duplicate", "variant"]


class Duplicates(_Strict):
    query_product_id: str | None
    query_title: str
    candidates: list[DuplicateCandidate]
