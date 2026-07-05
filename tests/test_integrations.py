"""Integration adapters, tested against the real libraries (no network: litellm
mock_response; langchain-core fake generations)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yoro import HashEmbedder


def test_litellm_semantic_cache_hit_and_invalidation(tmp_path):
    import json

    import litellm
    from litellm.caching.caching import Cache
    from yoro.integrations.litellm_cache import YoroSemanticCache
    from yoro import deps as depsmod

    sidecar = tmp_path / "deps.json"
    sidecar.write_text(json.dumps({"rollup.csv": "v1"}))
    yc = YoroSemanticCache(embedder=HashEmbedder(), deps_file=str(sidecar))
    litellm.cache = Cache()
    litellm.cache.cache = yc

    kw = dict(model="gpt-3.5-turbo", caching=True)
    m = [{"role": "user", "content": "summarize the march rollup numbers"}]

    r1 = litellm.completion(messages=m, mock_response="FIRST", **kw)
    assert r1.choices[0].message.content == "FIRST"
    r2 = litellm.completion(messages=m, mock_response="SECOND", **kw)
    assert r2.choices[0].message.content == "FIRST"  # served from YORO, not the mock
    assert yc.hits == 1

    # the sidecar reports the dependency moved -> must NOT serve the stale entry
    sidecar.write_text(json.dumps({"rollup.csv": "v2"}))
    depsmod._CACHE.clear()
    r3 = litellm.completion(messages=m, mock_response="THIRD", **kw)
    assert r3.choices[0].message.content == "THIRD"  # fresh call, correct

    # unrelated question -> novelty gate refuses force-fit
    r4 = litellm.completion(messages=[{"role": "user", "content": "capital city of france please"}],
                            mock_response="PARIS", **kw)
    assert r4.choices[0].message.content == "PARIS"
    litellm.cache = None


def test_langchain_cache_hit_scoping_and_invalidation(tmp_path):
    import json

    from langchain_core.outputs import Generation
    from yoro.integrations.langchain_cache import YoroLangChainCache
    from yoro import deps as depsmod

    sidecar = tmp_path / "deps.json"
    sidecar.write_text(json.dumps({"kb": "v1"}))
    c = YoroLangChainCache(embedder=HashEmbedder(), deps_file=str(sidecar))

    val = [Generation(text="cached answer")]
    c.update("summarize the quarterly kb article", "model-A params", val)

    got = c.lookup("summarize the quarterly kb article", "model-A params")
    assert got is not None and got[0].text == "cached answer"

    # a different model must not be served this entry (llm_string is a dependency)
    assert c.lookup("summarize the quarterly kb article", "model-B params") is None

    # novelty gate: unrelated prompt refuses force-fit
    assert c.lookup("weather in lisbon tomorrow morning", "model-A params") is None

    # dependency moved -> stale entry refuses to serve
    sidecar.write_text(json.dumps({"kb": "v2"}))
    depsmod._CACHE.clear()
    assert c.lookup("summarize the quarterly kb article", "model-A params") is None


def test_mcp_bridge_fingerprints_resources(tmp_path):
    """In-process MCP server via the official SDK: the bridge snapshots resources into
    a deps-file, the fingerprint moves when the resource changes, and the proxy-side
    deps source picks it up."""
    import asyncio
    import json

    from mcp.server.fastmcp import FastMCP
    from mcp.shared.memory import create_connected_server_and_client_session
    from yoro.integrations.mcp_bridge import run_bridge
    from yoro import deps as depsmod

    state = {"doc": "march numbers v1"}
    server = FastMCP("kb")

    @server.resource("kb://rollup")
    def rollup() -> str:
        return state["doc"]

    deps_path = str(tmp_path / "mcp_deps.json")

    async def scenario():
        async with create_connected_server_and_client_session(server._mcp_server) as session:
            first = await run_bridge(session, deps_path, cycles=1)
            state["doc"] = "march numbers v2 (revised)"
            second = await run_bridge(session, deps_path, cycles=1)
            return first, second

    first, second = asyncio.run(scenario())
    key = "mcp:kb://rollup"
    assert key in first and key in second
    assert first[key] != second[key]  # content change moved the fingerprint

    on_disk = json.loads(open(deps_path).read())
    assert on_disk[key] == second[key]

    depsmod._CACHE.clear()
    assert depsmod.resolve_deps({}, deps_file=deps_path)[key] == second[key]
