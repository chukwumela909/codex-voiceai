import asyncio
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("VOICE_AGENT_MODE", "mock")

from app.conversation_context import agent_transcript_with_context
from app.config import Settings
from app.memory import build_manager, get_store, reset_store_cache
from app.memory.embedder import (
    DEFAULT_LOCAL_MODEL,
    LocalEmbedder,
    MockEmbedder,
    build_embedder,
    embedder_ready,
)
from app.memory.manager import MemoryManager
from app.memory.store import MemoryRecord, VectorMemoryStore, cosine
from app.main import app


client = TestClient(app)


def run(coro):
    return asyncio.run(coro)


class _Settings:
    """Minimal settings stand-in for manager/store unit tests."""

    normalized_mode = "mock"
    memory_top_k = 5
    groq_api_key = None
    groq_model = "x"
    memory_embedding_provider = "openai"
    openai_api_key = None


# ---- embedder ---------------------------------------------------------------


def test_mock_embedder_is_deterministic_and_overlaps_on_shared_tokens():
    emb = MockEmbedder()
    [a1] = run(emb.embed(["I love jazz music"]))
    [a2] = run(emb.embed(["I love jazz music"]))
    [b] = run(emb.embed(["jazz is my favourite music"]))
    [c] = run(emb.embed(["please reboot the server now"]))
    assert a1 == a2  # deterministic
    # Shared vocabulary should score higher than unrelated text.
    assert cosine(a1, b) > cosine(a1, c)


def test_build_embedder_picks_mock_without_live_key():
    s = Settings(_env_file=None)  # mock mode, no keys
    assert build_embedder(s).name == "mock"
    assert embedder_ready(s) is False


# ---- local embedder (model2vec) --------------------------------------------


class _FakeStaticModel:
    """Stand-in for model2vec.StaticModel — no download, numpy output."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts):
        import numpy as np

        self.calls.append(list(texts))
        return np.array([[float(len(t))] * self.dim for t in texts], dtype="float32")


def test_local_embedder_returns_plain_float_lists():
    fake = _FakeStaticModel(dim=4)
    emb = LocalEmbedder(model=fake)
    out = run(emb.embed(["ab", "cde"]))
    assert out == [[2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]]
    # Must be JSON-serializable Python floats (numpy floats break the store).
    assert all(isinstance(x, float) for row in out for x in row)
    assert fake.calls == [["ab", "cde"]]


def test_local_embedder_falls_back_to_mock_on_load_failure(monkeypatch):
    emb = LocalEmbedder(model_id="does/not-exist")

    def _boom():
        raise RuntimeError("no model")

    monkeypatch.setattr(emb, "_load", _boom)
    out = run(emb.embed(["hello world"]))
    assert len(out[0]) == 256  # mock embedder dimensionality
    # Second call keeps using the fallback without retrying the load.
    assert len(run(emb.embed(["again"]))[0]) == 256


def test_build_embedder_selects_local_in_any_mode(monkeypatch):
    # provider=local needs no key and is honored even in mock mode (unlike openai).
    monkeypatch.setenv("VOICE_AGENT_MODE", "mock")
    monkeypatch.setenv("VOICE_AGENT_MEMORY_EMBEDDING_PROVIDER", "local")
    s = Settings(_env_file=None)
    emb = build_embedder(s)
    assert emb.name == "local"
    assert emb.model_id == DEFAULT_LOCAL_MODEL
    assert embedder_ready(s) is True


def test_local_embedder_model_id_is_configurable(monkeypatch):
    monkeypatch.setenv("VOICE_AGENT_MEMORY_EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("VOICE_AGENT_MEMORY_LOCAL_MODEL", "minishlab/potion-retrieval-32M")
    s = Settings(_env_file=None)
    assert build_embedder(s).model_id == "minishlab/potion-retrieval-32M"


@pytest.mark.skipif(
    not os.environ.get("RUN_LOCAL_EMBEDDER_INTEGRATION"),
    reason="downloads the model2vec model; set RUN_LOCAL_EMBEDDER_INTEGRATION=1 to run",
)
def test_local_embedder_real_model_is_semantic():
    emb = LocalEmbedder()
    facts = run(emb.embed(["I play drums in a band", "the weather is nice today"]))
    [query] = run(emb.embed(["do you make music?"]))
    # The music query must be closer to the drums fact than the weather fact.
    assert cosine(query, facts[0]) > cosine(query, facts[1])


# ---- store ------------------------------------------------------------------


def test_store_query_orders_by_similarity(tmp_path: Path):
    store = VectorMemoryStore(tmp_path / "store.json")
    emb = MockEmbedder()
    for text in ["the user loves jazz", "the user lives in Lagos", "the user codes in python"]:
        [vec] = run(emb.embed([text]))
        store.add(MemoryRecord(text=text, kind="fact", embedding=vec))
    [q] = run(emb.embed(["what music does the user like"]))
    top = store.query(q, 1)
    assert top and top[0][0].text == "the user loves jazz"


def test_store_dedupes_near_duplicates(tmp_path: Path):
    store = VectorMemoryStore(tmp_path / "store.json", dedupe_threshold=0.9)
    emb = MockEmbedder()
    [vec] = run(emb.embed(["the user loves jazz"]))
    assert store.add(MemoryRecord(text="the user loves jazz", kind="fact", embedding=vec)) is True
    assert store.add(MemoryRecord(text="the user loves jazz", kind="fact", embedding=vec)) is False
    assert store.count() == 1


def test_store_persists_and_reloads(tmp_path: Path):
    path = tmp_path / "store.json"
    emb = MockEmbedder()
    [vec] = run(emb.embed(["remember this"]))
    VectorMemoryStore(path).add(MemoryRecord(text="remember this", kind="fact", embedding=vec))
    assert path.exists()
    reloaded = VectorMemoryStore(path)
    assert reloaded.count() == 1
    assert reloaded.all()[0].text == "remember this"


# ---- manager ----------------------------------------------------------------


def test_manager_distills_facts_and_retrieves(tmp_path: Path):
    store = VectorMemoryStore(tmp_path / "store.json")
    manager = MemoryManager(embedder=MockEmbedder(), store=store, settings=_Settings())
    transcript = [
        {"role": "user", "content": "Hi, my name is Sam and I love jazz."},
        {"role": "assistant", "content": "Great to meet you, Sam!"},
    ]
    out = run(manager.distill_transcript(transcript, source_session="sess_1"))
    assert any("Sam" in f for f in out["facts"])
    assert any("jazz" in f.lower() for f in out["facts"])

    recalled = run(manager.retrieve("what music do I like?"))
    assert any("jazz" in r.text.lower() for r in recalled)


def test_manager_build_memory_block_formats_sections():
    manager = MemoryManager(embedder=MockEmbedder(), store=VectorMemoryStore(Path("unused")), settings=_Settings())
    records = [MemoryRecord(text="The user's name is Sam.", kind="fact", embedding=[0.0])]
    block = manager.build_memory_block(records, running_summary="we discussed dinner")
    assert "The user's name is Sam." in block
    assert "Earlier in this call: we discussed dinner" in block


# ---- injection seam (pure function) ----------------------------------------


def test_agent_transcript_with_context_prepends_memory_block():
    transcript = [{"role": "user", "content": "hello"}]
    out = agent_transcript_with_context(transcript, memory_block="MEMORY", intent_enabled=False)
    assert out[0] == {"role": "system", "content": "MEMORY"}
    assert out[-1] == {"role": "user", "content": "hello"}


def test_agent_transcript_with_context_no_block_is_passthrough():
    transcript = [{"role": "user", "content": "hello"}]
    out = agent_transcript_with_context(transcript, memory_block="   ", intent_enabled=False)
    assert all(t["role"] != "system" for t in out)


# ---- config -----------------------------------------------------------------


def test_memory_effective_enabled_auto_is_on_in_mock_off_in_live(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VOICE_AGENT_MODE", "mock")
    assert Settings(_env_file=None).memory_effective_enabled is True
    monkeypatch.setenv("VOICE_AGENT_MODE", "live")
    assert Settings(_env_file=None).memory_effective_enabled is False


def test_memory_explicit_true_enables_in_live(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VOICE_AGENT_MODE", "live")
    monkeypatch.setenv("VOICE_AGENT_MEMORY_ENABLED", "true")
    assert Settings(_env_file=None).memory_effective_enabled is True


def test_public_config_status_reports_memory_section():
    status = Settings(_env_file=None).public_config_status()
    assert "memory" in status
    assert "enabled" in status["memory"]
    assert "embedder_ready" in status["memory"]


# ---- HTTP endpoints ---------------------------------------------------------


def test_memory_endpoints_list_and_clear(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import app.main as main_mod

    reset_store_cache()
    monkeypatch.setattr(main_mod.settings, "memory_dir", str(tmp_path), raising=False)

    # seed one memory directly into the store the endpoint will read
    store = get_store(main_mod.settings)
    [vec] = run(MockEmbedder().embed(["the user loves jazz"]))
    store.add(MemoryRecord(text="the user loves jazz", kind="fact", embedding=vec))

    listing = client.get("/memory")
    assert listing.status_code == 200
    body = listing.json()
    assert body["count"] == 1
    assert body["memories"][0]["text"] == "the user loves jazz"
    assert "embedding" not in body["memories"][0]  # embeddings not leaked

    cleared = client.delete("/memory")
    assert cleared.status_code == 200
    assert cleared.json()["cleared"] == 1
    assert client.get("/memory").json()["count"] == 0
    reset_store_cache()


# ---- session integration ----------------------------------------------------


def test_session_close_distills_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from app.config import get_settings
    from app.mock_conversation import MockConversationSession

    settings = get_settings()
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path), raising=False)
    monkeypatch.setattr(settings, "memory_enabled", "true", raising=False)
    reset_store_cache()

    async def send(_payload):
        return None

    session = MockConversationSession("sess_close", send, settings)
    session.transcript = [
        {"role": "user", "content": "Hey, my name is Dana and I love hiking."},
        {"role": "assistant", "content": "Nice, Dana!"},
    ]
    run(session.close())

    stored = get_store(settings).all()
    assert any("Dana" in r.text for r in stored)
    reset_store_cache()
