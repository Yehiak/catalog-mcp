import pytest

from catalog_mcp.catalog import Filters, ProductNotFound, flatten_details, jaccard, model_numbers, platforms, title_tokens


def ids(ranked):
    return [pid for pid, _ in ranked]


def test_build_counts(catalog):
    assert len(catalog) == 20
    assert catalog.info["products_per_category"] == {"Appliances": 10, "Video_Games": 10}


def test_keyword_finds_exact_part_number(catalog):
    top = ids(catalog.keyword("DC31-00054A", 3))
    assert top[0] == "B0APPL0008"


def test_keyword_no_match_is_empty(catalog):
    assert catalog.keyword("zzzqqqxxx", 5) == []


def test_semantic_and_hybrid_rank_relevant_first(catalog):
    for mode in ("semantic", "hybrid"):
        top = ids(catalog.search("ice maker countertop", mode, 3))
        assert set(top[:2]) <= {"B0APPL0005", "B0APPL0006", "B0APPL0007"}, mode


def test_hybrid_contains_union_of_signals(catalog):
    top = ids(catalog.hybrid("water filter W10295370A", 5))
    assert {"B0APPL0001", "B0APPL0002", "B0APPL0003"} <= set(top)


@pytest.mark.parametrize("mode", ["keyword", "semantic", "hybrid"])
def test_filters_apply_in_every_mode(catalog, mode):
    f = Filters(category="Video_Games", max_price=60)
    hits = catalog.summaries(ids(catalog.search("controller for ps5", mode, 10, f)))
    assert hits, mode
    assert all(h["category"] == "Video_Games" and h["price"] is not None and h["price"] <= 60 for h in hits)


def test_brand_and_rating_filters(catalog):
    hits = catalog.summaries(ids(catalog.semantic("ice maker", 10, Filters(brand="chillmax", min_rating=4.0))))
    assert [h["product_id"] for h in hits] == ["B0APPL0006"]


def test_filter_matching_nothing(catalog):
    assert catalog.semantic("anything", 5, Filters(min_price=10_000)) == []


def test_get_and_missing(catalog):
    p = catalog.get("B0APPL0005")
    assert p["details"]["Capacity"] == "26 Pounds"
    assert p["features"][0].startswith("Makes 9")
    with pytest.raises(ProductNotFound):
        catalog.get("B000000000")


def test_compare_specs_and_winners(catalog):
    res = catalog.compare(["B0APPL0005", "B0APPL0006", "B0APPL0007"])
    names = [r["name"] for r in res["specs"]]
    assert names[:5] == ["Brand", "Store", "Price (USD)", "Average rating", "Number of ratings"]
    assert {"Capacity", "Voltage", "Item Weight"} <= set(names)
    assert res["cheapest_id"] == "B0APPL0005"  # B0APPL0007 has no price
    assert res["best_rated_id"] == "B0APPL0006"
    assert res["most_reviewed_id"] == "B0APPL0005"
    cap = next(r for r in res["specs"] if r["name"] == "Capacity")
    assert cap["values"] == ["26 Pounds", "44 Pounds", "15 Pounds"]


def test_best_rated_is_not_fooled_by_few_ratings(catalog):
    res = catalog.compare(["B0APPL0010", "B0APPL0004"])  # 5.0 from 2 ratings vs 4.6 from 950
    assert res["best_rated_id"] == "B0APPL0004"


def test_near_duplicates_by_shared_part_number(catalog):
    res = catalog.near_duplicates("B0APPL0001", min_similarity=0.3)
    by_id = {c["product_id"]: c for c in res["candidates"]}
    assert by_id["B0APPL0003"]["verdict"] == "likely_duplicate"
    assert "W10295370A" in by_id["B0APPL0003"]["shared_model_numbers"]
    assert all(c["category"] == "Appliances" for c in res["candidates"])


def test_near_duplicates_platform_variant(catalog):
    res = catalog.near_duplicates("B0GAME0001", min_similarity=0.3)
    by_id = {c["product_id"]: c for c in res["candidates"]}
    assert by_id["B0GAME0002"]["verdict"] == "variant"
    assert by_id["B0GAME0002"]["platform_conflict"] is True
    assert by_id["B0GAME0003"]["platform_conflict"] is False


def test_near_duplicates_by_title(catalog):
    res = catalog.near_duplicates(title="Whirlpool Refrigerator Water Filter W10295370A", min_similarity=0.3)
    assert res["query_product_id"] is None
    assert res["candidates"][0]["product_id"] in {"B0APPL0001", "B0APPL0003"}


def test_near_duplicates_needs_exactly_one_input(catalog):
    with pytest.raises(ValueError):
        catalog.near_duplicates()
    with pytest.raises(ValueError):
        catalog.near_duplicates("B0APPL0001", title="x")


def test_category_stats(catalog):
    s = catalog.category_stats("Appliances")
    assert s["products"] == 10 and s["with_price"] == 9
    assert s["top_subcategories"][0]["subcategory"] == "Parts & Accessories"
    assert sum(s["rating_histogram"].values()) == 10


def test_text_signals():
    assert model_numbers("Pump for DC31-00054A, 1500W, 99003147") == {"DC3100054A", "99003147"}
    assert platforms("Game - PlayStation 5") == {"ps5"}
    assert platforms("Xbox Series X headset") == {"xbox-series"}
    assert platforms("Nintendo Switch case") == {"switch"}
    assert jaccard(title_tokens("Red Ice Maker"), title_tokens("ice maker, red")) == 1.0
    assert flatten_details('{"Best Sellers Rank": {"Appliances": 12}, "Color": "Red"}') == {
        "Best Sellers Rank": "Appliances: 12", "Color": "Red"}
