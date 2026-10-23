"""tests/test_fix_catalog.py - regression tests for docs/FIX_CATALOG_2026-09.md.

Each test names the catalog item it guards. All GPU-free; the solver harness is exercised
with a scripted fake OpenAI client and a fake command runner.
"""
from __future__ import annotations

import json
import re
import types
from pathlib import Path

import numpy as np
import pytest

import benchmark.hard_ctf as hard
from pipeline.orchestrator import challenge_query_text


# --------------------------------------------------------------------------- #
# F1 - the family name (== the method) must never reach the solver             #
# --------------------------------------------------------------------------- #

def _family_tokens(fam_name: str):
    """Tokens that would leak the method: the family name and each of its parts."""
    toks = {fam_name.lower()}
    toks |= {t for t in fam_name.lower().split("_") if len(t) >= 3}
    return toks


def test_f1_no_family_token_reaches_solver(tmp_path):
    fams = ["b64_xor5a", "hex_rev_rot13"]
    corpus = hard.generate_corpus(tmp_path / "corpus", n_per_family=2, seed=7, families=fams)
    assert len(corpus) == 4
    for ch in corpus:
        leak = _family_tokens(ch["_family"]) - {"rev"}   # 'rev' alone is a common word; the pair test below covers it
        shown = ch["_display_name"]
        assert shown.startswith("task-") and len(shown) == 13
        # (a) solver-visible id, (b) working directory (`pwd`), (c) description, (d) filename
        visible = " ".join([shown, str(Path(ch["local_files"][0]["local_path"]).parent.name),
                            ch["description"], ch["local_files"][0]["text"]]).lower()
        for t in leak:
            assert t not in visible, (t, visible)
        assert ch["_family"].lower() not in visible
        # (e) the retrieval query
        q = challenge_query_text(ch).lower()
        assert ch["_family"].lower() not in q and ch["name"].lower() not in q
        # the LOGGING id still encodes the family so analyses keep working
        assert ch["name"].startswith(ch["_family"] + "_")


def test_f1_display_name_is_deterministic_and_neutral():
    a = hard.neutral_display_name("b64_xor5a", 424242, 3)
    b = hard.neutral_display_name("b64_xor5a", 424242, 3)
    c = hard.neutral_display_name("b64_xor5a", 424242, 4)
    assert a == b != c
    assert re.fullmatch(r"task-[0-9a-f]{8}", a)


def test_f1_real_adapter_presents_display_name(monkeypatch):
    """The adapter must hand the solver a copy whose `name` is the neutral id."""
    import batch_test.real_adapters as ra
    seen = {}

    def fake_solve(challenge, **kw):
        seen["name"] = challenge["name"]; seen["kw"] = kw
        return {"flag": None, "trace": [], "rounds_used": 0, "gave_up_reason": "x", "total_commands": 0}

    import solver.solver as ss
    monkeypatch.setattr(ss, "solve_challenge", fake_solve)
    fn = ra.real_solve_fn(max_output_tokens=10, context_window=1000)
    ch = {"name": "b64_xor5a_03", "_display_name": "task-deadbeef", "local_files": []}
    fn(ch, [], 4, seed=1)
    assert seen["name"] == "task-deadbeef"
    assert ch["name"] == "b64_xor5a_03"                    # caller's record untouched
    assert seen["kw"]["advertise_memory_tools"] is False   # F4a on the campaign path


def test_f1_isolate_dirs_are_neutral(tmp_path):
    from experiments.constructed_loo import _isolate as iso_c
    from experiments.lab_probe import _isolate as iso_p
    f = tmp_path / "data.bin"; f.write_bytes(b"x")
    ch = {"name": "b64_xor5a_03", "local_files": [{"local_path": str(f), "text": "data.bin"}]}
    c1 = iso_c(ch, "b64_xor5a_0_1_helpful", tmp_path / "w")
    c2 = iso_p(ch, "helpful", 0, tmp_path / "w2")
    for c in (c1, c2):
        d = Path(c["local_files"][0]["local_path"]).parent.name
        assert d.startswith("iso-") and "xor" not in d and "b64" not in d


# --------------------------------------------------------------------------- #
# F4a - prompt: no phantom memory tool, real budget                            #
# --------------------------------------------------------------------------- #

def test_f4a_prompt_no_memory_tools_when_off():
    import solver.solver as ss
    legacy = ss.render_system_prompt("", "strings, grep", advertise_memory_tools=True, max_rounds=None)
    fixed = ss.render_system_prompt("", "strings, grep", advertise_memory_tools=False, max_rounds=4)
    # legacy text preserved verbatim (reproducibility anchor)
    assert "3. Query your memory of past CTF challenges" in legacy
    assert '"action": "query_memory"' in legacy
    assert "**Strategy - use memory tools wisely:**" in legacy
    assert "up to 6 commands per round, with up to 7 rounds total" in legacy
    # fixed prompt: no memory tool anywhere, real budget
    assert "query_memory" not in fixed and "memory_search" not in fixed and "Query your memory" not in fixed
    assert f"up to {ss.MAX_COMMANDS_PER_ROUND} commands per round, with up to 4 rounds total" in fixed
    # everything else identical
    for line in ("**Critical rules:**", "**Tips - Crypto:**", "Available shell tools"):
        assert line in legacy and line in fixed


# --------------------------------------------------------------------------- #
# Fake LLM + fake shell for harness tests (F4b / F4c)                          #
# --------------------------------------------------------------------------- #

class _FakeClient:
    """Scripted chat.completions.create: returns the next canned content each call."""
    def __init__(self, scripts, fail_on_call=None):
        self.scripts = list(scripts); self.calls = 0; self.fail_on_call = fail_on_call
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise ConnectionError("Connection error.")
        content = self.scripts.pop(0) if self.scripts else json.dumps({"action": "give_up", "reason": "out"})
        msg = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)],
                                     usage=types.SimpleNamespace(total_tokens=10))


def _fake_run_commands(outputs):
    """Return a run_commands stand-in whose stdout for each command is looked up in `outputs`."""
    def _run(commands, working_dir, log=None, **kw):
        return [{"command": c, "stdout": outputs.get(c, ""), "stderr": "", "returncode": 0, "error": None}
                for c in commands]
    return _run


def _challenge(tmp_path):
    f = tmp_path / "data.txt"; f.write_text("x")
    return {"name": "task-00000000", "category": "crypto", "points": 1, "difficulty": "easy",
            "description": "Recover the flag.", "hints": [], "code_blocks": [],
            "local_files": [{"local_path": str(f), "text": "data.txt"}]}


def test_f4b_llm_error_is_its_own_reason_and_orchestrator_raises(monkeypatch, tmp_path):
    import solver.solver as ss
    from pipeline.orchestrator import Orchestrator, SolverUnavailable, SystemSpec
    from config import SolverConfig
    from evaluation.synthetic import build_world

    monkeypatch.setattr(ss, "get_available_tools", lambda: [])
    monkeypatch.setattr(ss, "client", _FakeClient([], fail_on_call=1))
    res = ss.solve_challenge(_challenge(tmp_path), max_rounds=3, advertise_memory_tools=False)
    assert res["flag"] is None and res["gave_up_reason"] == "llm_error"

    # the orchestrator turns it into an exception whose message the campaign retry loop recognizes
    world = build_world(n_challenges=2, dim=16, seed=1)
    cfg = SolverConfig().with_overrides(N=10, R=5, M=2, embedding_dim=16, context_resolution="category", seed=1)
    bad = lambda ch, nodes, max_rounds, seed=None: {"flag": None, "trace": [], "rounds_used": 1,
                                                    "gave_up_reason": "llm_error", "total_commands": 0}
    orch = Orchestrator(world.store, world.embedder, cfg, SystemSpec("B0", use_graph=False), solve_fn=bad)
    with pytest.raises(SolverUnavailable) as ei:
        orch.run_one(world.challenges[0], rng=np.random.RandomState(0))
    assert "connection error" in str(ei.value).lower()
    # opt-out path records error=True on the run instead
    orch.raise_on_llm_error = False
    run = orch.run_one(world.challenges[0], rng=np.random.RandomState(0))
    assert run.error is True and run.gave_up_reason == "llm_error" and run.outcome == 0


def test_f4c_wrong_candidate_flag_does_not_end_run_when_checker_given(monkeypatch, tmp_path):
    import solver.solver as ss
    monkeypatch.setattr(ss, "get_available_tools", lambda: [])
    wrong, right = "picoCTF{wr0ng_c4nd1d4t3}", "picoCTF{r1ght_4nsw3r_9z}"
    outputs = {"echo one": wrong, "echo two": right}
    monkeypatch.setattr(ss, "run_commands", _fake_run_commands(outputs))
    scripts = [json.dumps({"action": "run_commands", "commands": ["echo one"]}),
               json.dumps({"action": "run_commands", "commands": ["echo two"]})]

    # legacy behaviour: the first candidate ends the run (wrong flag returned)
    monkeypatch.setattr(ss, "client", _FakeClient(list(scripts)))
    res = ss.solve_challenge(_challenge(tmp_path), max_rounds=4, advertise_memory_tools=False)
    assert res["flag"] == wrong and res["rounds_used"] == 1

    # with a checker: the wrong candidate is rejected, the loop continues, the right one wins
    fake = _FakeClient(list(scripts))
    monkeypatch.setattr(ss, "client", fake)
    res = ss.solve_challenge(_challenge(tmp_path), max_rounds=4, advertise_memory_tools=False,
                             flag_checker=lambda f: f == right)
    assert res["flag"] == right and res["rounds_used"] == 2
    assert fake.calls == 2


def test_f9_orchestrator_passes_llm_seed_and_order_seed_only_when_supported():
    from pipeline.orchestrator import Orchestrator, SystemSpec
    from config import SolverConfig
    from evaluation.synthetic import build_world
    world = build_world(n_challenges=2, dim=16, seed=1)
    cfg = SolverConfig().with_overrides(N=10, R=5, M=2, embedding_dim=16, context_resolution="category", seed=1)
    got = {}

    def wide(ch, nodes, max_rounds, seed=None, order_seed=None, flag_checker=None):
        got.update(seed=seed, order_seed=order_seed, checker=flag_checker)
        return {"flag": None, "trace": [], "rounds_used": 1, "gave_up_reason": "rounds_exhausted", "total_commands": 0}

    orch = Orchestrator(world.store, world.embedder, cfg, SystemSpec("B2", use_bandit=False), solve_fn=wide)
    orch.randomize_lesson_order = True
    orch.flag_checker_factory = lambda ch: (lambda f: True)
    orch.run_one(world.challenges[0], rng=np.random.RandomState(0), llm_seed=12345)
    assert got["seed"] == 12345 and got["order_seed"] is not None and callable(got["checker"])

    # a narrow solve_fn (mock-style) must still work
    narrow = lambda ch, nodes, max_rounds, seed=None: {"flag": None, "trace": [], "rounds_used": 1,
                                                       "gave_up_reason": "rounds_exhausted", "total_commands": 0}
    orch2 = Orchestrator(world.store, world.embedder, cfg, SystemSpec("B2", use_bandit=False), solve_fn=narrow)
    orch2.randomize_lesson_order = True
    orch2.run_one(world.challenges[0], rng=np.random.RandomState(0), llm_seed=1)


# --------------------------------------------------------------------------- #
# F2 - distinctive citation                                                    #
# --------------------------------------------------------------------------- #

def _crypto_graph(fams=("b64_rev_rot13", "hex_xor42"), **kw):
    from retrieval.embedder import MockEmbedder
    store, truth = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=list(fams), **kw)
    by = {}
    for nid, (lab, fam) in truth.items():
        by.setdefault(fam, {})[lab] = store.get_node(nid)
    return store, by


def _trace(*cmds):
    return [{"round": i + 1, "commands": [c]} for i, c in enumerate(cmds)]


def test_f2_outer_token_alone_cites_nothing_in_distinctive_mode():
    from solver.citation import match_citations, distinctive_keywords
    store, by = _crypto_graph()
    h, p = by["b64_rev_rot13"]["helpful"], by["b64_rev_rot13"]["poison"]
    assert "base64" in [k.lower() for k in h.attributes["invocation_signature"]]
    assert "base64" in [k.lower() for k in p.attributes["invocation_signature"]]
    # the shared outer token is not distinctive for either lesson
    assert "base64" not in [k.lower() for k in distinctive_keywords(h, store)]
    assert "base64" not in [k.lower() for k in distinctive_keywords(p, store)]
    tr = _trace("base64 -d data.txt > x.bin", "cat x.bin")
    legacy = {m.node_id: m.cited for m in match_citations([h, p], tr, mode="any", store=store)}
    strict = {m.node_id: m.cited for m in match_citations([h, p], tr, mode="distinctive", store=store)}
    assert legacy == {h.id: True, p.id: True}          # the artifact the review found
    assert strict == {h.id: False, p.id: False}


def test_f2_distinctive_step_cites_only_its_own_lesson():
    from solver.citation import match_citations, distinctive_keywords
    store, by = _crypto_graph()
    h, p = by["b64_rev_rot13"]["helpful"], by["b64_rev_rot13"]["poison"]
    hk = [k for k in distinctive_keywords(h, store) if k.isalnum()]
    pk = [k for k in distinctive_keywords(p, store) if k.isalnum()]
    assert hk and pk and not (set(k.lower() for k in hk) & set(k.lower() for k in pk))
    tr_h = _trace("base64 -d data.txt | " + hk[0] + " | tr A-Za-z N-ZA-Mn-za-m")
    tr_p = _trace("base64 -d data.txt | python3 -c 'x=1' " + pk[0])
    res_h = {m.node_id: m.cited for m in match_citations([h, p], tr_h, mode="distinctive", store=store)}
    res_p = {m.node_id: m.cited for m in match_citations([h, p], tr_p, mode="distinctive", store=store)}
    assert res_h[h.id] and not res_h[p.id]
    assert res_p[p.id] and not res_p[h.id]
    for m in match_citations([h, p], tr_h, mode="distinctive", store=store):
        assert m.mode == "distinctive" and m.n_distinctive > 0


def test_f2_word_boundary_and_empty_regex():
    from solver.citation import _matches
    assert _matches("rev", "cat x | reverse_it", word_boundary=True) is None
    assert _matches("rev", "cat x | rev", word_boundary=True) == "rev"
    assert _matches("^", "python3 -c 'print(1)'") is None       # empty regex match never cites


# --------------------------------------------------------------------------- #
# F3 - deployed retrieval: deterministic top-M, real temperature, ε-stream     #
# --------------------------------------------------------------------------- #

def _synthetic(cfg_over=None):
    from config import SolverConfig
    from evaluation.synthetic import build_world
    from retrieval.retriever import ContextRetriever
    from retrieval.context_resolution import ContextResolver
    from retrieval.reranker import MockReranker
    world = build_world(n_challenges=6, dim=64, seed=3)
    base = dict(N=40, R=20, M=3, n_clusters=8, embedding_dim=64, context_resolution="category", seed=1)
    base.update(cfg_over or {})
    cfg = SolverConfig().with_overrides(**base)
    cr = ContextResolver(cfg).fit(world.store)
    retr = ContextRetriever(world.store, world.embedder, config=cfg, reranker=MockReranker(), context_resolver=cr)
    return world, cfg, retr


def test_f3_deterministic_topM_logs_unit_propensity():
    from pipeline.orchestrator import challenge_query_text
    world, cfg, retr = _synthetic()
    ch = world.challenges[0]
    r1 = retr.retrieve(challenge_query_text(ch), category=ch["category"], rng=np.random.RandomState(0),
                       use_bandit=False, deterministic=True)
    r2 = retr.retrieve(challenge_query_text(ch), category=ch["category"], rng=np.random.RandomState(99),
                       use_bandit=False, deterministic=True)
    assert r1.sampled_ids == r2.sampled_ids == [n.id for n in r1.pool_d_nodes[:cfg.M]]
    assert all(d["p"] == 1.0 for d in r1.per_draw_propensity) and r1.temperature == 0.0
    assert r1.policy["deterministic"] is True


def test_f3_low_tau_concentrates_inclusion_where_tau1_does_not():
    from pipeline.orchestrator import challenge_query_text
    ch_idx = 0
    def incl_rate(tau, trials=60):
        world, cfg, retr = _synthetic({"tau": tau})
        ch = world.challenges[ch_idx]
        helpful = set(world.helpful_node_ids[ch["category"]])
        hits = 0
        for s in range(trials):
            r = retr.retrieve(challenge_query_text(ch), category=ch["category"],
                              rng=np.random.RandomState(1000 + s), use_bandit=False)
            hits += len(helpful & set(r.sampled_ids))
        return hits / (trials * len(helpful))
    hot, cold = incl_rate(1.0), incl_rate(0.02)
    assert cold > 0.8, cold                 # deployed: the top lessons are (almost) always retrieved
    assert cold > hot + 0.2, (hot, cold)    # tau=1 could not do this


def test_f3_epsilon_stream_is_logged_and_propensity_is_the_mixture():
    from pipeline.orchestrator import challenge_query_text
    from ope.causal_value import run_inclusion_propensities, marginal_inclusion_propensities
    world, cfg, retr = _synthetic({"tau": 0.02, "epsilon_explore": 0.5, "tau_explore": 1.0,
                                   "exploration_design": "softmax"})   # the legacy (adaptive) stream
    ch = world.challenges[0]
    flags = []
    for s in range(40):
        r = retr.retrieve(challenge_query_text(ch), category=ch["category"],
                          rng=np.random.RandomState(500 + s), use_bandit=False)
        flags.append(r.exploration)
        assert r.temperature == (1.0 if r.exploration else 0.02)
        assert {k: r.policy[k] for k in ("tau_deploy", "tau_explore", "epsilon", "deterministic", "design")} == \
            {"tau_deploy": 0.02, "tau_explore": 1.0, "epsilon": 0.5, "deterministic": False, "design": "softmax"}
    assert 0.2 < np.mean(flags) < 0.8
    run = {"candidate_pool": r.candidate_pool, "sampled_set": r.sampled_ids, "temperature": r.temperature,
           "policy": r.policy, "run_id": "x"}
    pi = run_inclusion_propensities(run, cfg.M, n_mc=2000, seed=0)
    pd = marginal_inclusion_propensities(r.candidate_pool, cfg.M, 0.02, n_mc=2000, seed=0)
    pe = marginal_inclusion_propensities(r.candidate_pool, cfg.M, 1.0, n_mc=2000, seed=1)
    for k in pi:
        assert abs(pi[k] - (0.5 * pd[k] + 0.5 * pe[k])) < 1e-9
    assert run_inclusion_propensities({"candidate_pool": r.candidate_pool, "sampled_set": [], "policy": {"deterministic": True}}, 3) is None


# --------------------------------------------------------------------------- #
# F5 - a poison with no correct step                                           #
# --------------------------------------------------------------------------- #

def test_f5_cross_outer_poison_shares_no_token_with_truth():
    fams = ["b64_rev_rot13", "hex_xor42", "b32_zlib", "b85_gzip_rev"]
    store, by = _crypto_graph(fams, poison_kind="cross_outer")
    for f in fams:
        h, p = by[f]["helpful"], by[f]["poison"]
        hs = {k.lower() for k in h.attributes["invocation_signature"]}
        ps = {k.lower() for k in p.attributes["invocation_signature"]}
        assert not (hs & ps), (f, hs & ps)
        assert p.attributes["_poison_kind"] == "cross_outer"
        assert f.split("_")[0] not in p.attributes["text"].lower().split("outer")[1][:12]   # different outer named
    # legacy kind still shares the outer token (documents the defect the review found)
    store2, by2 = _crypto_graph(["b64_rev_rot13"], poison_kind="sibling")
    h, p = by2["b64_rev_rot13"]["helpful"], by2["b64_rev_rot13"]["poison"]
    assert "base64" in {k.lower() for k in h.attributes["invocation_signature"]} & {k.lower() for k in p.attributes["invocation_signature"]}
    with pytest.raises(ValueError):
        _crypto_graph(["b64_rev_rot13"], poison_kind="bogus")


# --------------------------------------------------------------------------- #
# F6 - ratio-scale contrasts; F8 - node value columns; F9 - cluster bootstrap  #
# --------------------------------------------------------------------------- #

def test_f6_ratio_scale_reports_relative_harm_the_risk_difference_hides():
    from experiments.a1_contrast import _ratio_scale, _contrast
    rng = np.random.RandomState(0)
    runs = []
    # 10 families: base rate 0.10 (no recipe) vs poison-only 0.01 -> RD only -0.09, RR ~0.1, OR ~0.09
    for f in range(10):
        for _ in range(300):
            runs.append({"family": f"fam{f}", "HC": False, "PC": False, "HS": False, "PS": False,
                         "outcome": int(rng.rand() < 0.10)})
        for _ in range(100):
            runs.append({"family": f"fam{f}", "HC": False, "PC": True, "HS": False, "PS": True,
                         "outcome": int(rng.rand() < 0.01)})
    rd, _ = _contrast(runs, cited=True)
    ratio, rows = _ratio_scale(runs, cited=True)
    assert -0.12 < rd < -0.06
    assert ratio["risk_ratio_weighted"] < 0.3 and ratio["odds_ratio_weighted"] < 0.3
    assert len(rows) == 10


def test_f6_cells_2x2_and_empty_context_base():
    from experiments.ontarget_analysis import _cells_2x2
    truth = {"h::a": ("helpful", "famA"), "p::a": ("poison", "famA"),
             "h::b": ("helpful", "famB"), "p::b": ("poison", "famB")}
    rng = np.random.RandomState(1)
    runs = []
    for fam, h, p in (("famA", "h::a", "p::a"), ("famB", "h::b", "p::b")):
        for i in range(400):
            cell = i % 4
            ss = {"x"} if cell == 0 else {h} if cell == 1 else {p} if cell == 2 else {h, p}
            base = {0: 0.2, 1: 0.8, 2: 0.05, 3: 0.6}[cell]
            runs.append({"challenge_id": f"{fam}_{i:02d}", "sampled_set": list(ss),
                         "outcome": int(rng.rand() < base)})
    c = _cells_2x2(runs, truth, n_boot=200)
    assert c["n"] == {"--": 200, "H-": 200, "-P": 200, "HP": 200}
    assert c["solve_rate"]["H-"] > c["solve_rate"]["--"] > c["solve_rate"]["-P"]
    v = c["vs_neither"]
    assert v["H-"]["rr"]["point"] > 2 and v["-P"]["rr"]["point"] < 0.6
    assert v["-P"]["or"]["ci_hi"] < 1.0 and v["H-"]["or"]["ci_lo"] > 1.0


def test_f9_family_cluster_bootstrap_reports_clusters():
    from ope.causal_value import summarize_by_label_ci
    vals = {f"n{i}": {"ate": 0.1 + 0.05 * (i % 3), "label": "helpful"} for i in range(12)}
    fam = {f"n{i}": f"fam{i // 3}" for i in range(12)}     # 4 families x 3 lessons
    plain = summarize_by_label_ci(vals, n_boot=2000, seed=0)["helpful"]
    clus = summarize_by_label_ci(vals, n_boot=2000, seed=0, family_of=fam)["helpful"]
    assert abs(plain["mean"] - clus["mean"]) < 1e-12
    assert clus["n_clusters"] == 4 and clus["cluster_bootstrap"] is True and clus["n"] == 12
    assert "n_clusters" not in plain


def test_f8_signflip_path_runs_and_values_are_plain(tmp_path):
    """The per-lesson values dict must contain only lesson entries (review #1) and lab_analysis's
    signflip path must run on a tiny synthetic log."""
    import experiments.lab_analysis as la
    from ope.causal_value import observational_lesson_values
    rng = np.random.RandomState(0)
    pool = [{"node_id": f"n{i}", "score": 0.3 + 0.1 * (i % 3)} for i in range(8)]
    runs = []
    for i in range(120):
        ss = list(rng.choice([c["node_id"] for c in pool], size=3, replace=False))
        runs.append({"run_id": f"r{i}", "challenge_id": f"fam_{i % 4:02d}", "candidate_pool": pool,
                     "sampled_set": ss, "temperature": 1.0, "outcome": int("n0" in ss or rng.rand() < 0.2),
                     "R_final": float("n0" in ss)})
    truth = {f"n{i}": ("helpful" if i == 0 else "poison", "fam") for i in range(8)}
    vals = observational_lesson_values(runs, truth=truth, n_boot=50)
    assert set(vals) == {f"n{i}" for i in range(8)} and all("ate" in d for d in vals.values())
    sf = la.signflip(runs, truth, n_boot_lesson=50)
    assert "per_lesson_snipw" in sf and sf["snipw"]["label_level"]["helpful"]["mean"] > 0
    import inspect, experiments.base_rate_sweep as brs
    assert "causal_ATE_ontarget_o" in inspect.getsource(brs.main) and "causal_ATE_marginal_R" in inspect.getsource(la.run)


# --------------------------------------------------------------------------- #
# Review 2026-09-04 (methodology) blocking items                               #
# --------------------------------------------------------------------------- #

def test_r1_per_solve_isolation_protects_the_source_artifact(monkeypatch, tmp_path):
    import batch_test.real_adapters as ra
    import solver.solver as ss
    src = tmp_path / "corpus" / "task-abc" / "data.bin"; src.parent.mkdir(parents=True); src.write_bytes(b"ORIGINAL")
    seen = {}

    def fake_solve(challenge, **kw):
        p = Path(challenge["local_files"][0]["local_path"])
        seen["path"] = str(p)
        p.write_bytes(b"TRUNCATED")                 # the malignant `destroy` arm, in the copy
        (p.parent / "decoded.txt").write_text("leftover")
        return {"flag": None, "trace": [], "rounds_used": 1, "gave_up_reason": "rounds_exhausted", "total_commands": 1}

    monkeypatch.setattr(ss, "solve_challenge", fake_solve)
    fn = ra.real_solve_fn(max_output_tokens=10, context_window=1000, workroot=str(tmp_path / "work"))
    ch = {"name": "b64_xor5a_03", "_display_name": "task-abc", "local_files": [{"local_path": str(src), "text": "data.bin"}]}
    res = fn(ch, [], 4, seed=1)
    assert res["isolated"] is True
    assert seen["path"] != str(src) and "iso-" in seen["path"]
    assert src.read_bytes() == b"ORIGINAL"                       # source untouched
    assert not (src.parent / "decoded.txt").exists()             # no leftovers in the corpus dir
    assert not Path(seen["path"]).parent.exists()                # temp dir removed
    assert ch["local_files"][0]["local_path"] == str(src)        # caller's record untouched

    # a solver that somehow damages the SOURCE raises instead of logging a failure
    def bad_solve(challenge, **kw):
        src.write_bytes(b"DAMAGED")
        return {"flag": None, "trace": [], "rounds_used": 1, "gave_up_reason": "x", "total_commands": 0}
    monkeypatch.setattr(ss, "solve_challenge", bad_solve)
    fn2 = ra.real_solve_fn(max_output_tokens=10, context_window=1000, workroot=str(tmp_path / "work"))
    with pytest.raises(ra.ArtifactCorrupted):
        fn2(ch, [], 4, seed=1)


def test_r2_rendered_lessons_never_name_the_family():
    from solver.context_assembly import assemble_context, render_node
    fams = ["b64_rev_rot13", "hex_xor42", "b32_zlib"]
    store, by = _crypto_graph(fams, malignant_arms=list(hard.MALIGNANT_ARMS), distractor_arm=True)
    ids, _ = store.get_all_embeddings()
    rendered = assemble_context([store.get_node(i) for i in ids]).lower()
    for f in fams:
        assert f.lower() not in rendered, f
    for f in fams:
        for lab, node in by[f].items():
            assert re.fullmatch(r"lesson-[0-9a-f]{8}", node.attributes["canonical_name"]), (lab, node.attributes["canonical_name"])
            assert node.attributes["canonical_name"] in render_node(node)


def test_r3_stratified_exploration_gives_exact_half_inclusion_and_is_off_when_frozen():
    from pipeline.orchestrator import challenge_query_text
    from retrieval.embedder import MockEmbedder
    from config import SolverConfig
    from retrieval.retriever import ContextRetriever
    from retrieval.context_resolution import ContextResolver
    from retrieval.reranker import MockReranker
    fams = ["b64_rev_rot13", "hex_xor42", "b32_zlib", "b85_gzip_rev"]
    emb = MockEmbedder(dim=16, seed=1)
    store, truth = hard.build_seed_graph(emb, families=fams)
    hid = {f: n for n, (l, f) in truth.items() if l == "helpful"}; pid = {f: n for n, (l, f) in truth.items() if l == "poison"}
    cfg = SolverConfig().with_overrides(N=40, R=30, M=3, embedding_dim=16, context_resolution="category", seed=1,
                                        tau=0.05, epsilon_explore=1.0, exploration_design="stratified")
    cr = ContextResolver(cfg).fit(store)
    retr = ContextRetriever(store, emb, config=cfg, reranker=MockReranker(), context_resolver=cr)
    hits_h = hits_p = 0; n = 300
    for s in range(n):
        r = retr.retrieve("q", category="crypto", rng=np.random.RandomState(s), use_bandit=False,
                          explore=True, on_target_family="hex_xor42")
        assert r.exploration and r.policy["design"] == "stratified" and r.policy["epsilon"] == 1.0
        pool = {c["node_id"]: c for c in r.candidate_pool}
        assert pool[hid["hex_xor42"]]["pi_explore"] == 0.5 and pool[pid["hex_xor42"]]["on_target"] is True
        assert all(abs(c["pi_explore"] - (3 - 1.0) / (len(pool) - 2)) < 1e-9 for c in r.candidate_pool if not c["on_target"])
        hits_h += hid["hex_xor42"] in r.sampled_ids; hits_p += pid["hex_xor42"] in r.sampled_ids
        assert len(r.sampled_ids) == 3
    assert 0.4 < hits_h / n < 0.6 and 0.4 < hits_p / n < 0.6
    # frozen / eval: explore=False -> no exploration draw at all
    r = retr.retrieve("q", category="crypto", rng=np.random.RandomState(7), use_bandit=False, explore=False,
                      on_target_family="hex_xor42")
    assert r.exploration is False and r.policy["epsilon"] == 0.0 and r.policy["explore_enabled"] is False


def test_r3_mixture_uses_exact_design_propensity_and_exploration_only_slice():
    from ope.causal_value import run_inclusion_propensities, _collect_obs, observational_lesson_values
    pool = [{"node_id": "a", "score": 0.9, "on_target": True, "pi_explore": 0.5},
            {"node_id": "b", "score": 0.1, "on_target": True, "pi_explore": 0.5},
            {"node_id": "c", "score": 0.0, "on_target": False, "pi_explore": 0.25},
            {"node_id": "d", "score": 0.0, "on_target": False, "pi_explore": 0.25}]
    pol = {"tau_deploy": 0.05, "tau_explore": 1.0, "epsilon": 0.2, "deterministic": False, "design": "stratified"}
    run_e = {"candidate_pool": pool, "sampled_set": ["a", "c"], "temperature": 1.0, "policy": pol, "exploration": True,
             "run_id": "e", "outcome": 1, "R_final": 1.0}
    pi = run_inclusion_propensities(run_e, 2, n_mc=2000, seed=0)
    assert abs(pi["a"] - (0.8 * pi_deploy_a(pool) + 0.2 * 0.5)) < 0.05
    # deploy run under the same design: exploration term reconstructed from on_target flags
    pool_d = [{k: v for k, v in c.items() if k != "pi_explore"} for c in pool]
    run_d = {"candidate_pool": pool_d, "sampled_set": ["a", "c"], "temperature": 0.05, "policy": pol, "exploration": False,
             "run_id": "d", "outcome": 1, "R_final": 1.0}
    pi_d = run_inclusion_propensities(run_d, 2, n_mc=2000, seed=0)
    assert abs(pi_d["a"] - pi["a"]) < 0.05 and abs(pi_d["c"] - pi["c"]) < 0.05
    # exploration-only slice keeps only exploration runs and uses the exact design propensity
    audit = {}
    obs = _collect_obs([run_e, run_d], clip=0.05, exploration_only=True, audit=audit)
    assert set(obs) == {"a", "b", "c", "d"} and all(len(v) == 1 for v in obs.values())
    assert obs["a"][0][2] == 0.5 and obs["c"][0][2] == 0.25 and audit["obs"] == 4
    # a greedy log has no usable randomization: legacy (no policy) AND the deployed arm's frozen eval
    # runs (policy present, epsilon 0, tau 0.05) are both refused
    assert run_inclusion_propensities({"candidate_pool": pool_d, "sampled_set": ["a"], "temperature": 0.05}, 1) is None
    frozen = {"candidate_pool": pool_d, "sampled_set": ["a"], "temperature": 0.05,
              "policy": {**pol, "epsilon": 0.0, "explore_enabled": False}}
    assert run_inclusion_propensities(frozen, 1) is None
    audit = {}
    vals = observational_lesson_values([run_e] * 6 + [run_d] * 6, exploration_only=True, response_key="outcome", audit=audit)
    assert "_positivity_audit" not in vals and all("ate" in d for d in vals.values())   # review #1
    assert audit["obs"] == 24 and audit["runs_skipped_no_randomization"] == 0


def pi_deploy_a(pool):
    from ope.causal_value import marginal_inclusion_propensities
    return marginal_inclusion_propensities(pool, 2, 0.05, n_mc=2000, seed=0)["a"]


def test_r6_run_level_distinctiveness_generic_tokens_and_idioms():
    from solver.citation import match_citations, distinctive_keywords
    store, by = _crypto_graph(["b64_xor42", "hex_xor42", "b64_rev_rot13"])   # grid families (idiom sigs)
    h1, h2 = by["b64_xor42"]["helpful"], by["hex_xor42"]["helpful"]
    # two sampled lessons sharing the xor-key tokens: neither can be cited on them
    d1 = {k.lower() for k in distinctive_keywords(h1, store, others=[h1, h2])}
    assert not ({"xor", "0x42", "\\x42"} & d1)
    # alone, the key tokens ARE distinctive and the \x42 idiom is accepted (as a LITERAL, not a regex)
    d1_alone = {k.lower() for k in distinctive_keywords(h1, store)}
    assert "0x42" in d1_alone and "\\x42" in d1_alone
    from solver.citation import _matches
    assert _matches("\\x42", "base64 -d data.bin") is None            # regex \x42 == 'B' must NOT fire on 'b'
    assert _matches("\\x42", "python3 -c 'k=b\"\\x42\"'") is not None
    tr = _trace("base64 -d data.txt | python3 -c \"import sys; d=sys.stdin.buffer.read(); print(bytes(x ^ 0x42 for x in d))\"")
    m = {x.node_id: x.cited for x in match_citations([h1], tr, mode="distinctive", store=store)}
    assert m[h1.id]
    # the display name never grounds a citation
    tr_name = _trace(f"echo {h1.attributes['canonical_name']}")
    assert not any(x.cited for x in match_citations([h1], tr_name, mode="distinctive", store=store))
    # benchmark generic tokens (outer decoders) are removed even for off-target lessons
    tr_outer = _trace("base64 -d data.txt")
    assert not any(x.cited for x in match_citations([h1, by["b64_rev_rot13"]["helpful"]], tr_outer, mode="distinctive",
                                                    store=store, generic_tokens=hard.GENERIC_CITATION_TOKENS))
    # rot13 idioms
    h3 = by["b64_rev_rot13"]["helpful"]
    assert any(x.cited for x in match_citations([h3], _trace("python3 -c \"import codecs; print(codecs.decode(s,'rot_13'))\""),
                                                mode="distinctive", store=store, generic_tokens=hard.GENERIC_CITATION_TOKENS))


def test_r7_order_seed_is_a_hash_of_challenge_and_llm_seed():
    from pipeline.orchestrator import Orchestrator, SystemSpec
    from config import SolverConfig
    from evaluation.synthetic import build_world
    world = build_world(n_challenges=2, dim=16, seed=1)
    cfg = SolverConfig().with_overrides(N=10, R=5, M=2, embedding_dim=16, context_resolution="category", seed=1)
    got = []
    def wide(ch, nodes, max_rounds, seed=None, order_seed=None, flag_checker=None):
        got.append(order_seed)
        return {"flag": None, "trace": [], "rounds_used": 1, "gave_up_reason": "rounds_exhausted", "total_commands": 0}
    for spec in (SystemSpec("B2", use_bandit=False), SystemSpec("B0", use_graph=False)):
        o = Orchestrator(world.store, world.embedder, cfg, spec, solve_fn=wide); o.randomize_lesson_order = True
        o.run_one(world.challenges[0], rng=np.random.RandomState(0), llm_seed=4242)
    assert got[0] == got[1] is not None       # same challenge + llm_seed -> same order across conditions


# --------------------------------------------------------------------------- #
# Second code review (2026-09-04)                                              #
# --------------------------------------------------------------------------- #

def test_c2_tool_nodes_named_after_their_tool_stay_citable():
    from solver.citation import match_citations
    from knowledge_graph.models import Node, NodeType
    tool = Node(id="tool::r2", type=NodeType.TOOL, attributes={"canonical_name": "r2", "invocation_signature": "r2"})
    label_only = Node(id="tech::x", type=NodeType.TECHNIQUE,
                      attributes={"canonical_name": "lesson-deadbeef", "invocation_signature": ["gunzip"]})
    tr = _trace("r2 -q -c 'aaa' binary", "echo lesson-deadbeef")
    res = {m.node_id: m for m in match_citations([tool, label_only], tr, mode="distinctive")}
    assert res["tool::r2"].cited and res["tool::r2"].n_distinctive >= 1
    assert not res["tech::x"].cited                      # the display name never grounds a citation


def test_c3_stratified_marginals_exact_for_k_ge_m_and_small_pools():
    from retrieval.sampling import stratified_design_marginals
    from retrieval.retriever import fixed_design_draw
    for n, k, m in ((16, 2, 3), (16, 2, 1), (10, 4, 3), (4, 2, 3), (6, 6, 2)):
        pi_ot, pi_rest = stratified_design_marginals(n, k, m)
        cnt = np.zeros(n); N = 20000
        rng = np.random.RandomState(1)
        for _ in range(N):
            order, per, lp, pi = fixed_design_draw(n, m, rng, list(range(k)), stratified=True)
            assert len(set(order)) == len(order) <= m
            cnt[order] += 1
        emp = cnt / N
        if k:
            assert abs(emp[:k].mean() - pi_ot) < 0.012, (n, k, m, emp[:k].mean(), pi_ot)
        if n > k:
            assert abs(emp[k:].mean() - pi_rest) < 0.012, (n, k, m, emp[k:].mean(), pi_rest)
        assert abs(pi[:k].mean() - pi_ot) < 1e-12 if k else True


def test_c5_cross_outer_falls_back_for_families_without_a_candidate():
    fams = ["b64_rev_gzip_rot13", "hex_gzip_xor5a", "b32_zlib"]     # depth-3, depth-2, depth-1
    store, by = _crypto_graph(fams, poison_kind="cross_outer")
    kinds = {f: by[f]["poison"].attributes["_poison_kind"] for f in fams}
    assert set(kinds.values()) <= {"cross_outer", "sibling_fallback"}
    for f in fams:
        if kinds[f] == "cross_outer":
            hs = {k.lower() for k in by[f]["helpful"].attributes["invocation_signature"]}
            ps = {k.lower() for k in by[f]["poison"].attributes["invocation_signature"]}
            assert not (hs & ps), f


def test_c7_bad_request_is_logged_not_raised(monkeypatch, tmp_path):
    import solver.solver as ss
    from pipeline.orchestrator import Orchestrator, SystemSpec
    from config import SolverConfig
    from evaluation.synthetic import build_world
    monkeypatch.setattr(ss, "get_available_tools", lambda: [])

    class _Bad(Exception):
        status_code = 400
    class _BadClient(_FakeClient):
        def _create(self, **kw):
            raise _Bad("prompt too long")
    _BadClient.__name__ = "x"
    monkeypatch.setattr(ss, "client", _BadClient([]))
    _Bad.__name__ = "BadRequestError"
    res = ss.solve_challenge(_challenge(tmp_path), max_rounds=3, advertise_memory_tools=False)
    assert res["gave_up_reason"] == "llm_bad_request"

    world = build_world(n_challenges=2, dim=16, seed=1)
    cfg = SolverConfig().with_overrides(N=10, R=5, M=2, embedding_dim=16, context_resolution="category", seed=1)
    bad = lambda ch, nodes, max_rounds, seed=None: {"flag": None, "trace": [], "rounds_used": 1,
                                                    "gave_up_reason": "llm_bad_request", "total_commands": 0}
    orch = Orchestrator(world.store, world.embedder, cfg, SystemSpec("B0", use_graph=False), solve_fn=bad)
    run = orch.run_one(world.challenges[0], rng=np.random.RandomState(0))     # no raise
    assert run.error is True and run.gave_up_reason == "llm_bad_request"


def test_p2_forced_on_target_arms_bracket_the_deployed_arm():
    """oracle_static forces the family's helpful lesson into M; worst_static forces its poison;
    both are flagged deterministic/forced so no observational estimator uses them."""
    from retrieval.embedder import MockEmbedder
    from config import SolverConfig
    from retrieval.retriever import ContextRetriever
    from retrieval.context_resolution import ContextResolver
    from retrieval.reranker import MockReranker
    from ope.causal_value import run_inclusion_propensities
    fams = ["b64_rev_rot13", "hex_xor42", "b32_zlib", "b85_gzip_rev", "b64_xor42"]
    emb = MockEmbedder(dim=16, seed=1)
    store, truth = hard.build_seed_graph(emb, families=fams)
    hid = {f: n for n, (l, f) in truth.items() if l == "helpful"}; pid = {f: n for n, (l, f) in truth.items() if l == "poison"}
    cfg = SolverConfig().with_overrides(N=40, R=30, M=3, embedding_dim=16, context_resolution="category", seed=1)
    cr = ContextResolver(cfg).fit(store)
    retr = ContextRetriever(store, emb, config=cfg, reranker=MockReranker(), context_resolver=cr)
    for label, table in (("helpful", hid), ("poison", pid)):
        r = retr.retrieve("q", category="crypto", rng=np.random.RandomState(0), use_bandit=False,
                          deterministic=True, on_target_family="hex_xor42", force_on_target=label)
        assert table["hex_xor42"] in r.sampled_ids and len(r.sampled_ids) == 3
        assert r.policy["forced"] == label and r.policy["deterministic"] is True
        run = {"candidate_pool": r.candidate_pool, "sampled_set": r.sampled_ids, "temperature": r.temperature, "policy": r.policy}
        assert run_inclusion_propensities(run, 3) is None
