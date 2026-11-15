"""The server through a real MCP client session (in-process transport)."""

import json

import pytest
from mcp import Client

from catalog_mcp.server import create_server

pytestmark = pytest.mark.anyio

TOOLS = {"search_products", "get_product", "compare_products", "find_near_duplicates"}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client(catalog):
    async with Client(create_server(catalog=catalog)) as c:
        yield c


async def test_tools_are_listed_with_schemas(client):
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == TOOLS
    for t in tools.values():
        assert t.description and len(t.description) > 40
        assert t.output_schema is not None
        assert t.annotations.read_only_hint is True
        for name, prop in t.input_schema["properties"].items():
            assert prop.get("description"), f"{t.name}.{name} has no description"
    assert tools["search_products"].input_schema["required"] == ["query"]


async def test_search_products(client):
    r = await client.call_tool("search_products", {"query": "ps5 controller", "limit": 3, "category": "Video_Games"})
    assert not r.is_error
    data = r.structured_content
    assert data["mode"] == "hybrid" and data["count"] == len(data["results"]) <= 3
    assert all(x["category"] == "Video_Games" for x in data["results"])
    assert json.loads(r.content[0].text) == data  # text copy for text-only clients


async def test_get_product(client):
    r = await client.call_tool("get_product", {"product_id": "B0APPL0005", "max_description_chars": 0})
    assert not r.is_error
    assert r.structured_content["details"]["Voltage"] == "120 Volts"
    assert r.structured_content["description"] == ""


async def test_compare_products(client):
    r = await client.call_tool("compare_products", {"product_ids": ["B0APPL0005", "B0APPL0006"]})
    assert not r.is_error
    assert r.structured_content["cheapest_id"] == "B0APPL0005"


async def test_find_near_duplicates(client):
    r = await client.call_tool("find_near_duplicates", {"product_id": "B0GAME0001", "min_similarity": 0.5})
    assert not r.is_error
    verdicts = {c["product_id"]: c["verdict"] for c in r.structured_content["candidates"]}
    assert verdicts.get("B0GAME0002") == "variant"


@pytest.mark.parametrize("tool,args,needle", [
    ("get_product", {"product_id": "nope"}, "at least 10 characters"),
    ("get_product", {"product_id": "B000000000"}, "No product with id"),
    ("search_products", {"query": ""}, "at least 1 character"),
    ("search_products", {"query": "x", "limit": 0}, "greater than or equal to 1"),
    ("search_products", {"query": "x", "mode": "fuzzy"}, "hybrid"),
    ("search_products", {"query": "x", "category": "Books"}, "Unknown category"),
    ("search_products", {"query": "x", "min_price": 50, "max_price": 10}, "min_price"),
    ("compare_products", {"product_ids": ["B0APPL0005"]}, "at least 2"),
    ("compare_products", {"product_ids": ["B0APPL0005", "B0APPL0005"]}, "duplicates"),
    ("compare_products", {"product_ids": ["B0APPL0005", "B000000000"]}, "Unknown product ids"),
    ("find_near_duplicates", {}, "exactly one"),
    ("find_near_duplicates", {"product_id": "B0APPL0005", "title": "ice maker"}, "exactly one"),
    ("search_products", {"query": "x", "unexpected": 1}, "unexpected"),
])
async def test_bad_input_is_a_clean_tool_error(client, tool, args, needle):
    r = await client.call_tool(tool, args)
    assert r.is_error
    assert needle in r.content[0].text


async def test_resources(client):
    listed = [str(r.uri) for r in (await client.list_resources()).resources]
    assert listed == ["catalog://stats"]
    templates = [t.uri_template for t in (await client.list_resource_templates()).resource_templates]
    assert templates == ["catalog://categories/{category}/stats"]

    stats = json.loads((await client.read_resource("catalog://stats")).contents[0].text)
    assert stats["products"] == 20
    cat = json.loads((await client.read_resource("catalog://categories/Video_Games/stats")).contents[0].text)
    assert cat["products"] == 10 and cat["top_brands"]


async def test_unknown_category_resource(client):
    with pytest.raises(Exception, match="Unknown category"):
        await client.read_resource("catalog://categories/Books/stats")


async def test_prompt(client):
    prompts = (await client.list_prompts()).prompts
    assert [p.name for p in prompts] == ["write_product_comparison"]
    assert {a.name for a in prompts[0].arguments} == {"product_ids", "audience", "length"}
    r = await client.get_prompt("write_product_comparison",
                                {"product_ids": "B0APPL0005, B0APPL0006", "audience": "a café owner", "length": "short"})
    text = r.messages[0].content.text
    assert '["B0APPL0005", "B0APPL0006"]' in text and "a café owner" in text and "150 words" in text
