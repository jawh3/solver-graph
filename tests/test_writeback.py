"""Tests for C7: gated write-back, provenance, promotion gate."""

from config import SolverConfig, WritebackMode
from knowledge_graph.store import KnowledgeGraphStore
from knowledge_graph.models import NodeType
from retrieval.embedder import MockEmbedder
from writeback import WriteBackManager, MockTrajectoryDecomposer, PromotionTracker


def _solve_result(tools, solved=True):
    trace = [{"round": i + 1, "commands": [f"{t} target_artifact"]} for i, t in enumerate(tools)]
    return {"flag": "picoCTF{x}" if solved else None, "trace": trace,
            "rounds_used": len(trace), "total_commands": len(trace),
            "gave_up_reason": None if solved else "rounds_exhausted"}


def _store():
    return KnowledgeGraphStore(), MockEmbedder(dim=32, seed=1)


def test_decomposer_extracts_tools_not_shell_noise():
    dec = MockTrajectoryDecomposer()
    res = _solve_result(["r2", "ropper", "echo"])  # echo is shell noise
    cands = dec.decompose({"category": "pwn"}, res)
    names = {c.node.attributes["canonical_name"] for c in cands}
    assert "r2" in names and "ropper" in names
    assert "echo" not in names


def test_writeback_inserts_provisional_with_provenance():
    store, emb = _store()
    wb = WriteBackManager(SolverConfig())
    recs = wb.write_back({"name": "ch1", "category": "rev"},
                         _solve_result(["r2"]), run_id="run1", store=store, embedder=emb)
    assert len(recs) == 1
    node = store.get_node(recs[0].new_node_id)
    assert node.attributes["tier"] == "provisional"
    assert node.attributes["source"] == "writeback"
    # derived_from provenance edge exists
    edge = store.get_edge(recs[0].new_node_id, "run_run1")
    assert edge is not None
    assert edge.attributes.get("edge_type") == "derived_from"


def test_writeback_mode_off_and_successful():
    store, emb = _store()
    wb_off = WriteBackManager(SolverConfig().with_overrides(writeback_mode="off"))
    assert wb_off.write_back({"name": "c", "category": "x"}, _solve_result(["r2"]),
                             "r", store, emb) == []
    wb_succ = WriteBackManager(SolverConfig().with_overrides(writeback_mode="successful"))
    # a failed run should not write back in 'successful' mode
    assert wb_succ.write_back({"name": "c", "category": "x"}, _solve_result(["r2"], solved=False),
                              "r", store, emb) == []


def test_failure_decomposition_tagged():
    store, emb = _store()
    wb = WriteBackManager(SolverConfig().with_overrides(writeback_mode="both"))
    recs = wb.write_back({"name": "c", "category": "web"}, _solve_result(["sqlmap"], solved=False),
                         "rf", store, emb)
    assert len(recs) == 1
    assert recs[0].failure_derived is True
    assert store.get_node(recs[0].new_node_id).attributes["failure_derived"] is True


def test_promotion_gate_requires_k_and_c():
    cfg = SolverConfig().with_overrides(k_promote=3, c_promote=2)
    store, emb = _store()
    wb = WriteBackManager(cfg)
    recs = wb.write_back({"name": "ch1", "category": "rev"}, _solve_result(["r2"]),
                         "run1", store, emb)
    nid = recs[0].new_node_id
    tracker = PromotionTracker(cfg)

    # cited in 2 successful runs but only 1 distinct challenge -> NOT promoted
    assert tracker.register_citation(store, nid, "ch1", success=True) is False
    assert tracker.register_citation(store, nid, "ch1", success=True) is False
    assert store.get_node(nid).attributes["tier"] == "provisional"
    # third cited success on a SECOND challenge -> meets k=3 and c=2 -> promoted
    assert tracker.register_citation(store, nid, "ch2", success=True) is True
    assert store.get_node(nid).attributes["tier"] == "trusted"


def test_n1_lesson_never_trusted():
    cfg = SolverConfig().with_overrides(k_promote=3, c_promote=2)
    store, emb = _store()
    wb = WriteBackManager(cfg)
    nid = wb.write_back({"name": "ch1", "category": "rev"}, _solve_result(["r2"]),
                        "run1", store, emb)[0].new_node_id
    tracker = PromotionTracker(cfg)
    tracker.register_citation(store, nid, "ch1", success=True)  # N=1
    assert store.get_node(nid).attributes["tier"] == "provisional"


def test_ungated_promotes_on_first_success():
    cfg = SolverConfig().with_overrides(k_promote=3, c_promote=2)
    store, emb = _store()
    wb = WriteBackManager(cfg)
    nid = wb.write_back({"name": "ch1", "category": "rev"}, _solve_result(["r2"]),
                        "run1", store, emb)[0].new_node_id
    tracker = PromotionTracker(cfg)
    assert tracker.register_citation(store, nid, "ch1", success=True, ungated=True) is True
    assert store.get_node(nid).attributes["tier"] == "trusted"
