"""Paths and defaults. Everything can be overridden with environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_data_dir

DATASET_REPO = "McAuley-Lab/Amazon-Reviews-2023"
DEFAULT_CATEGORIES = ("Appliances", "Video_Games")
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# bge models expect this instruction in front of short retrieval queries (not documents).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Every category of the dataset's item metadata, for `catalog-mcp build --categories`.
ALL_CATEGORIES = (
    "All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing", "Automotive",
    "Baby_Products", "Beauty_and_Personal_Care", "Books", "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry", "Digital_Music", "Electronics",
    "Gift_Cards", "Grocery_and_Gourmet_Food", "Handmade_Products", "Health_and_Household",
    "Health_and_Personal_Care", "Home_and_Kitchen", "Industrial_and_Scientific", "Kindle_Store",
    "Magazine_Subscriptions", "Movies_and_TV", "Musical_Instruments", "Office_Products",
    "Patio_Lawn_and_Garden", "Pet_Supplies", "Software", "Sports_and_Outdoors",
    "Subscription_Boxes", "Tools_and_Home_Improvement", "Toys_and_Games", "Unknown", "Video_Games",
)


def default_data_dir() -> Path:
    env = os.environ.get("CATALOG_MCP_HOME")
    return Path(env).expanduser() if env else Path(user_data_dir("catalog-mcp", appauthor=False))


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def db(self) -> Path:
        return self.root / "catalog.duckdb"

    @property
    def index(self) -> Path:
        return self.root / "vectors.faiss"

    @property
    def build_info(self) -> Path:
        return self.root / "build_info.json"

    def exists(self) -> bool:
        return self.db.exists() and self.index.exists() and self.build_info.exists()


def paths(data_dir: str | Path | None = None) -> Paths:
    return Paths(Path(data_dir).expanduser() if data_dir else default_data_dir())
