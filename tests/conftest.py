"""A tiny hand-written catalog and a deterministic embedder, so tests run offline in seconds."""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pytest

from catalog_mcp.catalog import Catalog
from catalog_mcp.config import Paths
from catalog_mcp.ingest import build


class HashingEmbedder:
    """Signed feature hashing of word unigrams + bigrams. Similar titles -> similar vectors."""

    name = "hashing-test"

    def __init__(self, dim: int = 512):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _vec(self, text: str) -> np.ndarray:
        words = re.findall(r"[a-z0-9]+", text.lower())
        feats = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        v = np.zeros(self._dim, dtype=np.float32)
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            v[h % self._dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_docs(self, texts, batch_size=256, progress=False):
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)

    def encode_query(self, text):
        return self._vec(text).reshape(1, -1)


def _p(asin, title, store, price, rating, n, cats, features=(), details=None, description=()):
    return {
        "main_category": cats[0] if cats else None, "title": title, "average_rating": rating,
        "rating_number": n, "features": list(features), "description": list(description),
        "price": price, "images": [{"large": f"https://example.com/{asin}.jpg"}], "store": store,
        "categories": list(cats), "details": details or {}, "parent_asin": asin,
    }


APPLIANCES = [
    _p("B0APPL0001", "Whirlpool W10295370A Refrigerator Water Filter", "Whirlpool", 49.99, 4.7, 5200,
       ["Appliances", "Parts & Accessories", "Refrigerator Parts", "Water Filters"],
       ["Reduces chlorine taste and odor", "Replace every 6 months"],
       {"Brand": "Whirlpool", "Part Number": "W10295370A", "Item Weight": "0.5 pounds"}),
    _p("B0APPL0002", "Refrigerator Water Filter Replacement for W10295370A EDR1RXD1, 3 Pack", "PureWater", 24.99, 4.4, 1800,
       ["Appliances", "Parts & Accessories", "Refrigerator Parts", "Water Filters"],
       ["NSF certified", "3 filters, 18 months of clean water"],
       {"Brand": "PureWater", "Part Number": "PW-370A", "Item Weight": "1.4 pounds"}),
    _p("B0APPL0003", "W10295370A Water Filter for Whirlpool Refrigerator", "FilterKing", 19.99, 4.1, 300,
       ["Appliances", "Parts & Accessories", "Refrigerator Parts", "Water Filters"],
       ["Fits Whirlpool and KitchenAid fridges"], {"Brand": "FilterKing", "Item Weight": "0.5 pounds"}),
    _p("B0APPL0004", "Dishwasher Upper Rack Wheel Roller WP99003147", "PartsPro", 9.99, 4.6, 950,
       ["Appliances", "Parts & Accessories", "Dishwasher Parts"],
       ["Replaces 99003147"], {"Brand": "PartsPro", "Part Number": "WP99003147"}),
    _p("B0APPL0005", "Countertop Ice Maker Machine, 26 lbs in 24 Hours, Self-Cleaning", "FrostyCo", 119.0, 4.3, 2100,
       ["Appliances", "Refrigerators, Freezers & Ice Makers", "Ice Makers"],
       ["Makes 9 bullet ice cubes in 6 minutes", "Quiet compressor"],
       {"Brand": "FrostyCo", "Capacity": "26 Pounds", "Voltage": "120 Volts", "Item Weight": "18 pounds"}),
    _p("B0APPL0006", "Portable Ice Maker Countertop, 44 lbs per Day, Stainless Steel", "ChillMax", 189.0, 4.5, 640,
       ["Appliances", "Refrigerators, Freezers & Ice Makers", "Ice Makers"],
       ["Makes 24 ice cubes in 13 minutes"],
       {"Brand": "ChillMax", "Capacity": "44 Pounds", "Voltage": "115 Volts", "Item Weight": "30 pounds"}),
    _p("B0APPL0007", "Mini Ice Maker for Home Bar", "ChillMax", None, 3.2, 12,
       ["Appliances", "Refrigerators, Freezers & Ice Makers", "Ice Makers"],
       ["Small and light"], {"Brand": "ChillMax", "Capacity": "15 Pounds"}),
    _p("B0APPL0008", "Washing Machine Drain Pump for Samsung DC31-00054A", "Samsung", 39.5, 4.6, 410,
       ["Appliances", "Parts & Accessories", "Washer Parts"],
       ["Genuine OEM drain pump"], {"Brand": "Samsung", "Part Number": "DC31-00054A"}),
    _p("B0APPL0009", "Range Hood Grease Filter, Aluminum Mesh, 2 Pack", "HoodMate", 15.0, 4.0, 220,
       ["Appliances", "Parts & Accessories", "Range Hood Parts"],
       ["Dishwasher safe aluminum mesh"], {"Brand": "HoodMate", "Product Dimensions": "10 x 12 inches"}),
    _p("B0APPL0010", "Espresso Machine Portafilter Basket 58mm", "BrewParts", 12.5, 5.0, 2,
       ["Appliances", "Parts & Accessories", "Coffee Machine Parts"],
       ["Stainless steel"], {"Brand": "BrewParts"}),
]

GAMES = [
    _p("B0GAME0001", "Starfall Odyssey - PlayStation 5", "Nebula Games", 59.99, 4.6, 3100,
       ["Video Games", "PlayStation 5", "Games"], ["Open-world space adventure"],
       {"Rated": "Teen", "Release date": "March 3, 2023"}),
    _p("B0GAME0002", "Starfall Odyssey - Nintendo Switch", "Nebula Games", 49.99, 4.4, 1200,
       ["Video Games", "Nintendo Switch", "Games"], ["Open-world space adventure"],
       {"Rated": "Teen", "Release date": "March 3, 2023"}),
    _p("B0GAME0003", "Starfall Odyssey (PS5) Standard Edition", "GameStore", 57.0, 4.5, 90,
       ["Video Games", "PlayStation 5", "Games"], ["Explore 100 planets"], {"Rated": "Teen"}),
    _p("B0GAME0004", "DualSense Wireless Controller for PS5, Midnight Black", "Sony", 69.99, 4.8, 45000,
       ["Video Games", "PlayStation 5", "Accessories", "Controllers"],
       ["Haptic feedback", "Adaptive triggers"], {"Brand": "Sony", "Color": "Midnight Black"}),
    _p("B0GAME0005", "Controller Charging Station for PS5, Dual Charger Dock", "ChargeIt", 19.99, 4.5, 8700,
       ["Video Games", "PlayStation 5", "Accessories", "Chargers"],
       ["Charges two controllers in 3 hours"], {"Brand": "ChargeIt"}),
    _p("B0GAME0006", "Racing Wheel and Pedals for PS5 and PC, 900 Degree", "SpeedRig", 299.0, 4.2, 760,
       ["Video Games", "PlayStation 5", "Accessories", "Racing Wheels"],
       ["Force feedback", "Metal pedals"], {"Brand": "SpeedRig"}),
    _p("B0GAME0007", "Legend of the Lost Kingdom - Nintendo Switch", "Pixel Forge", 39.99, 4.9, 15000,
       ["Video Games", "Nintendo Switch", "Games"], ["Puzzle and adventure in a fantasy world"], {"Rated": "Everyone 10+"}),
    _p("B0GAME0008", "Carrying Case for Nintendo Switch with 20 Game Card Slots", "TravelPro", 14.99, 4.7, 22000,
       ["Video Games", "Nintendo Switch", "Accessories", "Cases"], ["Hard shell"], {"Brand": "TravelPro"}),
    _p("B0GAME0009", "Gaming Headset for Xbox Series X with Noise Cancelling Mic", "SoundWave", 49.0, 4.1, 3300,
       ["Video Games", "Xbox Series X & S", "Accessories", "Headsets"], ["50mm drivers"], {"Brand": "SoundWave"}),
    _p("B0GAME0010", "Retro Arcade Stick for PC", "ArcadeHub", None, 3.9, 40,
       ["Video Games", "PC", "Accessories", "Controllers"], ["Sanwa-style buttons"], {"Brand": "ArcadeHub"}),
]


@pytest.fixture(scope="session")
def catalog_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("catalog")
    files = {}
    for category, rows in (("Appliances", APPLIANCES), ("Video_Games", GAMES)):
        path = root / f"meta_{category}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        files[category] = path
    build(Paths(root / "data"), list(files), files=files, embedder=HashingEmbedder(), log=lambda m: None)
    return root / "data"


@pytest.fixture(scope="session")
def catalog(catalog_dir):
    return Catalog(catalog_dir, embedder=HashingEmbedder())
