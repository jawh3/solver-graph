"""Test the FORGE champion-broadcast competitor (baseline BC, §14.2)."""

from config import SolverConfig
from knowledge_graph.store import KnowledgeGraphStore
from retrieval.embedder import MockEmbedder
from writeback.forge_competitor import ForgeChampionWriteBack


def _solve(tools, steps, solved=True):
    trace = [{"round": i + 1, "commands": [f"{t} target"]} for i, t in enumerate(tools)]
    return {"flag": "picoCTF{x}" if solved else None, "trace": trace,
            "rounds_used": steps, "total_commands": len(tools),
            "gave_up_reason": None if solved else "x"}


def test_forge_champion_is_trusted_immediately_and_replaced_by_faster():
    store = KnowledgeGraphStore()
    emb = MockEmbedder(dim=16, seed=1)
    forge = ForgeChampionWriteBack(SolverConfig())

    # first success -> champion, immediately TRUSTED (ungated; the FORGE property)
    recs = forge.write_back({"name": "c1", "category": "pwn"}, _solve(["ropper"], steps=9),
                            "r1", store, emb)
    assert len(recs) == 1
    assert recs[0].tier == "trusted"
    nid = recs[0].new_node_id
    assert store.get_node(nid).attributes["tier"] == "trusted"

    # a slower success does NOT replace the champion
    assert forge.write_back({"name": "c2", "category": "pwn"}, _solve(["ropper"], steps=12),
                            "r2", store, emb) == []

    # a faster success DOES replace it
    recs3 = forge.write_back({"name": "c3", "category": "pwn"}, _solve(["ropper", "pwntools"], steps=5),
                             "r3", store, emb)
    assert len(recs3) == 1


def test_forge_ignores_failures():
    store = KnowledgeGraphStore()
    emb = MockEmbedder(dim=16, seed=1)
    forge = ForgeChampionWriteBack(SolverConfig())
    assert forge.write_back({"name": "c", "category": "web"}, _solve(["sqlmap"], 8, solved=False),
                            "r", store, emb) == []
