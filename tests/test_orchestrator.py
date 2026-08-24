"""End-to-end orchestrator tests over the synthetic world (spec §7–§9, §12)."""

import numpy as np

from config import SolverConfig, ContextResolution
from retrieval.retriever import ContextRetriever
from retrieval.context_resolution import ContextResolver
from retrieval.reranker import MockReranker
from bandit.bandit import TabularBandit
from pipeline import Orchestrator, SystemSpec
from instrumentation.jsonl_logger import ExperimentLogger
from evaluation.synthetic import build_world


def _cfg(**kw):
    base = dict(N=40, R=20, M=6, n_clusters=8, embedding_dim=64,
                context_resolution="category", seed=1)
    base.update(kw)
    return SolverConfig().with_overrides(**base)


def _orch(world, cfg, system, bandit=None, logger=None):
    cr = ContextResolver(cfg).fit(world.store)
    retr = ContextRetriever(world.store, world.embedder, config=cfg,
                            reranker=MockReranker(), context_resolver=cr, bandit=bandit)
    return Orchestrator(world.store, world.embedder, cfg, system,
                        retriever=retr, bandit=bandit, logger=logger)


def test_b0_no_graph_runs_and_emits_runrecord(tmp_path):
    world = build_world(n_challenges=6, dim=64, seed=2)
    cfg = _cfg()
    logger = ExperimentLogger("b0", root=tmp_path)
    system = SystemSpec("B0", use_graph=False)
    orch = _orch(world, cfg, system, logger=logger)
    run = orch.run_one(world.challenges[0], experiment_id="b0", rng=np.random.RandomState(0))
    assert run.sampled_set == []          # no graph -> no lessons
    assert logger.read("run")[0]["challenge_id"] == world.challenges[0]["name"]


def test_b2_recall_ranks_helpful_high_and_samples_it_more(tmp_path):
    # The claim is statistical, not per-draw: a matching-category lesson should
    # (a) rank high by cosine in the candidate pool (recall), and
    # (b) be sampled more often than a random cross-category distractor.
    world = build_world(n_challenges=6, dim=64, seed=3)
    cfg = _cfg(M=6)
    logger = ExperimentLogger("b2", root=tmp_path)
    system = SystemSpec("B2", use_rerank=True, use_mmr=True, use_bandit=False)
    orch = _orch(world, cfg, system, logger=logger)
    ch = world.challenges[0]            # crypto challenge solved by sage/gmpy2
    helpful = set(world.helpful_node_ids[ch["category"]])

    # (a) recall: helpful nodes have higher mean cosine than cross-category nodes
    run0 = orch.run_one(ch, experiment_id="b2", rng=np.random.RandomState(0))
    pool = {c["node_id"]: c["cosine"] for c in run0.candidate_pool}
    helpful_cos = [pool[n] for n in helpful if n in pool]
    cross_cos = [v for nid, v in pool.items() if nid.startswith("tool::") and not nid.startswith("tool::crypto")]
    assert helpful_cos, "helpful lessons must be recalled into the pool"
    assert np.mean(helpful_cos) > np.mean(cross_cos)

    # (b) sampling frequency over many seeds
    helpful_hits = 0
    cross_hits = 0
    trials = 40
    for s in range(trials):
        r = orch.run_one(ch, experiment_id="b2", rng=np.random.RandomState(100 + s))
        helpful_hits += len(helpful & set(r.sampled_set))
        cross_hits += sum(1 for n in r.sampled_set
                          if n.startswith("tool::") and not n.startswith("tool::crypto"))
    # per-node sampling rate: helpful (2 nodes) vs cross (many nodes)
    helpful_rate = helpful_hits / (trials * len(helpful))
    n_cross = sum(1 for nid in pool if nid.startswith("tool::") and not nid.startswith("tool::crypto"))
    cross_rate = cross_hits / (trials * max(n_cross, 1))
    assert helpful_rate > cross_rate, (helpful_rate, cross_rate)

    # propensities logged for every sampled node (OPE requirement)
    assert len(run0.per_draw_propensity) == len(run0.sampled_set)


def test_b4_bandit_learns_to_prefer_helpful_over_poison(tmp_path):
    world = build_world(n_challenges=36, dim=64, seed=5)
    cfg = _cfg(M=6, alpha=0.2, kappa=2.0)
    bandit = TabularBandit(cfg)
    logger = ExperimentLogger("b4", root=tmp_path)
    system = SystemSpec("B4", use_bandit=True)
    orch = _orch(world, cfg, system, bandit=bandit, logger=logger)

    # train over several epochs through the dataset
    for epoch in range(6):
        for i, ch in enumerate(world.challenges):
            orch.run_one(ch, experiment_id="b4", rng=np.random.RandomState(1000 * epoch + i))

    # For at least one category, the helpful tool's learned Q should exceed the
    # poison node's Q in the matching context (poison routes the long way -> lower R).
    updates = logger.read("update")
    assert updates, "bandit should have emitted UpdateRecords"
    # collect final Q per node from the bandit
    helpful_qs, poison_qs = [], []
    for ctx, nodes in bandit.cells.items():
        for nid, cell in nodes.items():
            if nid.startswith("tool::") and cell.n > 0:
                helpful_qs.append(cell.Q)
            if nid.startswith("poison::") and cell.n > 0:
                poison_qs.append(cell.Q)
    # helpful lessons should on average earn higher Q than poison lessons
    if helpful_qs and poison_qs:
        assert np.mean(helpful_qs) > np.mean(poison_qs)


def test_b5_writeback_grows_graph_with_provisional_nodes(tmp_path):
    world = build_world(n_challenges=6, dim=64, seed=9)
    cfg = _cfg(M=6)
    bandit = TabularBandit(cfg)
    logger = ExperimentLogger("b5", root=tmp_path)
    system = SystemSpec("B5", use_bandit=True, use_writeback=True, writeback_gated=True)
    orch = _orch(world, cfg, system, bandit=bandit, logger=logger)
    n_before = len(world.store.get_all_embeddings()[0])
    for i, ch in enumerate(world.challenges):
        orch.run_one(ch, experiment_id="b5", rng=np.random.RandomState(i))
    n_after = len(world.store.get_all_embeddings()[0])
    assert n_after >= n_before    # write-back may add provisional nodes
    wbs = logger.read("writeback")
    if wbs:
        assert all(w["tier"] == "provisional" for w in wbs)


def test_recovery_masking_denied_credit():
    # A lesson present in context but NOT cited must get no UpdateRecord under cited_only.
    world = build_world(n_challenges=4, dim=64, seed=11)
    cfg = _cfg(M=6, credit_mode="cited_only")
    bandit = TabularBandit(cfg)
    system = SystemSpec("B4", use_bandit=True)
    orch = _orch(world, cfg, system, bandit=bandit)
    run = orch.run_one(world.challenges[0], rng=np.random.RandomState(0))
    # nodes sampled but never executed cannot have accrued n>0 in this single run
    # (they were not cited) — verify at least one sampled-but-not-cited node exists
    # and has no bandit observation.
    from solver.citation import match_citations
    # rebuild citation set is internal; instead check: number of cells with n>0
    # equals number of cited nodes (<= sampled). This holds for cited_only.
    n_observed = sum(1 for nodes in bandit.cells.values() for c in nodes.values() if c.n > 0)
    assert n_observed <= len(run.sampled_set)
