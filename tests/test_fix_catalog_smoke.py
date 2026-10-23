"""Regression tests for the post-smoke fixes (FIX_CATALOG 2026-09; Phase-1 smoke 2026-09-05 and the
independent review of the same day): F22 diagnostics + context ladder, F23/F-7 lure, F24 recall-all,
F25 gates, F4e finish_reason, F-1 neutral filename, F-2 rerank prior, F-3 B2det without MMR, F-5 balanced
selection, F-6 prompt hygiene (asserted on the REAL messages), F-12 charset robustness, F-13 clean brackets.
Mock mode, no GPU, no network."""
from __future__ import annotations

import json
import types

import numpy as np
import pytest

from benchmark import hard_ctf as hard
from retrieval.embedder import MockEmbedder

OUTERS = {"b64_rot13": "base64", "hex_xor42": "hex", "b32_zlib": "base32", "b85_gzip_rev": "base85"}
INNER_TOKENS = ("rot13", "xor", "zlib", "gzip", "bz2", "rev", "0x42", "0x5a", "length", "padding")


# ---------------------------------------------------------------- F22 / F-1 / F-12
def test_f22_charset_class_names_each_outer_alphabet_across_the_grid():
    rng = np.random.RandomState(3)
    bad = []
    for fam in hard._BY_NAME.values():
        for _ in range(3):
            flag = "picoCTF{" + "".join(rng.choice(list("abcdef0123456789_"), int(rng.randint(10, 22)))) + "}"
            key = hard.alphabet_key(fam.encode(flag, rng))
            if key != {"b64": "base64", "hex": "hex", "b32": "base32", "b85": "base85"}[fam.name.split("_")[0]]:
                bad.append((fam.name, key))
    assert not bad, bad[:5]
    assert hard.alphabet_key(b"\x00\xff\x10") == "other" and hard.alphabet_key(b"") == "other"


def test_f12_short_base85_is_not_mistaken_for_base64():
    import base64
    rng = np.random.RandomState(5)
    for _ in range(400):
        raw = bytes(rng.randint(32, 127, int(rng.randint(18, 26))).tolist())      # 25-32 chars of base85
        assert hard.alphabet_key(base64.b85encode(raw)) == "base85"
        assert hard.alphabet_key(base64.b64encode(raw)) == "base64"


def test_f1_every_family_uses_one_neutral_filename_and_charset_diagnostics_carry_only_the_alphabet(tmp_path):
    assert {f.filename for f in hard._BY_NAME.values()} == {hard.ARTIFACT_FILENAME}
    fams = list(OUTERS)
    legacy = hard.generate_corpus(tmp_path / "l", n_per_family=1, seed=9, families=fams)
    new = hard.generate_corpus(tmp_path / "n", n_per_family=1, seed=9, families=fams, diagnostics="charset")
    for c in legacy:
        assert "alphabet:" not in c["artifacts"][0]["static_metrics"] and "_retrieval_category" not in c
    descs = set()
    for c in new:
        sm = c["artifacts"][0]["static_metrics"].lower()
        assert "alphabet:" in sm and "length" not in sm and "padding" not in sm
        assert c["_retrieval_category"] == OUTERS[c["_family"]]
        assert c["_family"].lower() not in sm and not any(t in sm for t in INNER_TOKENS), sm
        descs.add(c["description"])
    assert len(descs) == 1, descs          # the description is identical for every family
    with pytest.raises(ValueError):
        hard.generate_corpus(tmp_path / "x", n_per_family=1, seed=9, families=fams[:1], diagnostics="bogus")


def test_f22_retrieval_category_drives_the_bandit_context_for_every_ladder_rung(tmp_path):
    from pipeline.orchestrator import Orchestrator, SystemSpec
    from config import SolverConfig
    from retrieval.retriever import ContextRetriever
    from retrieval.context_resolution import ContextResolver
    from retrieval.reranker import MockReranker
    fams = ["b64_rot13", "hex_xor42"]
    emb = MockEmbedder(dim=16, seed=1)
    store, truth = hard.build_seed_graph(emb, families=fams)
    cfg = SolverConfig().with_overrides(N=40, R=30, M=2, embedding_dim=16, context_resolution="category", seed=1)
    retr = ContextRetriever(store, emb, config=cfg, reranker=MockReranker(), context_resolver=ContextResolver(cfg).fit(store))
    corpus = hard.generate_corpus(tmp_path / "c", n_per_family=1, seed=4, families=fams, diagnostics="charset")
    fake = lambda ch, nodes, max_rounds, seed=None, **kw: {"flag": None, "trace": [], "rounds_used": 1,
                                                          "gave_up_reason": "rounds_exhausted", "total_commands": 0}
    orch = Orchestrator(store, emb, cfg, SystemSpec("B2", use_bandit=False), retriever=retr, solve_fn=fake)
    ch = dict(corpus[0]); assert ch["_retrieval_category"] == "base64"
    assert orch.run_one(ch, rng=np.random.RandomState(0)).context_node_id == "ctx::cat::base64"
    ch2 = dict(corpus[1]); ch2["_retrieval_category"] = ch2["_family"]
    assert orch.run_one(ch2, rng=np.random.RandomState(0)).context_node_id == f"ctx::cat::{ch2['_family']}"
    ch3 = dict(corpus[1]); ch3["_retrieval_category"] = "global"
    assert orch.run_one(ch3, rng=np.random.RandomState(0)).context_node_id == "ctx::cat::global"


# ---------------------------------------------------------------- F-6 / F-11: the REAL messages
class _CaptureClient:
    def __init__(self):
        self.messages = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.messages.append(kw["messages"])
        msg = types.SimpleNamespace(content=json.dumps({"action": "give_up", "reason": "test"}))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")],
                                     usage=types.SimpleNamespace(total_tokens=10))


def test_f6_solver_messages_carry_no_family_inner_alphabet_or_private_field(monkeypatch, tmp_path):
    import solver.solver as ss
    from batch_test.real_adapters import real_solve_fn
    monkeypatch.setattr(ss, "get_available_tools", lambda: [])
    fams = list(OUTERS)
    corpus = hard.generate_corpus(tmp_path / "c", n_per_family=1, seed=11, families=fams, diagnostics="charset")
    texts = {}
    for c in corpus:
        cap = _CaptureClient(); monkeypatch.setattr(ss, "client", cap)
        c = dict(c); c["_retrieval_category"] = c["_family"]          # the oracle rung's private key
        fn = real_solve_fn(isolate=False, max_output_tokens=64, context_window=4096)
        fn(c, [], 2, seed=0)
        assert cap.messages, "no LLM call captured"
        blob = json.dumps(cap.messages).lower()
        fam = c["_family"].lower()
        assert fam not in blob and c["name"].lower() not in blob
        # private VALUES must not leak (key names such as `submit_flag` legitimately occur in the prompt)
        assert c["_flag"].lower() not in blob and "_retrieval_category" not in blob and "static_metrics" not in blob
        assert "alphabet" not in blob
        # inner-transform tokens: checked on the challenge-derived (user) messages; the SYSTEM prompt's
        # generic tool list legitimately names gzip/rev/xxd for every challenge alike
        user_blob = json.dumps([m for conv in cap.messages for m in conv if m.get("role") == "user"]).lower()
        for tok in ("rot13", "xor", "zlib", "gzip", "bz2", " rev", "0x42", "0x5a"):
            assert tok not in user_blob, tok
        assert "difficulty: unknown" in blob and "difficulty: easy" not in blob and "difficulty: hard" not in blob
        import re
        texts[c["_family"]] = re.sub(r"\(\d+ bytes\)", "(N bytes)", json.dumps(cap.messages[0]).replace(c["_display_name"], "TASK"))
    # up to the display hash and the artifact byte count (what `ls -l` shows any agent; identical across
    # arms), every family gets the IDENTICAL prompt (review F-1/F-6)
    assert len(set(texts.values())) == 1


# ---------------------------------------------------------------- F-7 lure
def test_f7_one_lure_per_outer_with_unique_tokens_in_the_recipe_register():
    store, truth = hard.build_seed_graph(MockEmbedder(dim=8, seed=1),
                                         families=["b64_rot13", "hex_xor42", "b32_zlib", "b85_gzip_rev", "b64_rev_xor5a"],
                                         distractor_arm=True)
    lures = [(n, f) for n, (l, f) in truth.items() if l == "distractor_ontarget"]
    assert sorted(f for _, f in lures) == ["outer::b32", "outer::b64", "outer::b85", "outer::hex"]
    other_tokens = {t.lower() for n, (l, f) in truth.items() if l != "distractor_ontarget"
                    for t in (store.get_node(n).attributes.get("invocation_signature") or [])}
    texts = set()
    for n, f in lures:
        a = store.get_node(n).attributes; text = a["text"].lower(); texts.add(text)
        assert text.startswith("2 steps"), text
        for bad in ("data.txt", "flag", "ctf", "pico", "strings", "binwalk", "exiftool", "scalpel", "recover"):
            assert bad not in text, (bad, text)
        assert set(a["invocation_signature"]) == set(hard._LURE_TOKENS) == {"uniq", "column"}
        assert not (set(a["invocation_signature"]) & other_tokens)
        assert a["_outer"] in text or hard._OUTER_LABEL[a["_outer"]] in text
    assert len(texts) == 4


def test_audit_selection_excludes_core_families_and_cross_outer_conflicts():
    """Lesson audit L-3/L-4: no hand-written core family in a selection; the cross-outer poison of a selected
    family is never the true recipe of another selected family, and the store never hands one out."""
    names = sorted(hard._BY_NAME)
    core = {f.name for f in hard.FAMILIES}
    sel = hard.select_balanced_families(names, per_outer=6, seed=424242, poison_kinds=("sibling", "cross_outer"))
    assert len(sel) == 24 and not (set(sel) & core)
    recipes = {hard._BY_NAME[n].true_recipe for n in sel}
    for n in sel:
        co = hard.cross_outer_poison(hard._BY_NAME[n], exclude=set(sel))
        assert co is None or co[0] not in recipes, n
    store, truth = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=sel, poison_kind="cross_outer")
    for nid, (lab, fam) in truth.items():
        if lab == "poison":
            a = store.get_node(nid).attributes
            assert a["text"] not in recipes and a["_poison_kind"] in ("cross_outer", "sibling_fallback")
    # dose bookkeeping on sibling poisons
    store, truth = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=sel[:4])
    for nid, (lab, fam) in truth.items():
        if lab == "poison":
            d = store.get_node(nid).attributes["_dose"]
            assert d["wrong_class"] in ("key_swap", "codec_swap", "class_change") and d["first_wrong_decode_step"] >= 1
    assert hard.poison_dose("hex_zlib") is None      # core family: no sibling


def test_audit_clean_malignant_arms_keep_the_harm_and_drop_the_hygiene_defects():
    for n in ("b32_xorc3", "hex_gzip_xor5a"):
        fam = hard._BY_NAME[n]
        for arm in ("destroy_clean", "exhaust_clean", "wrongflag_clean"):
            text, sig = hard.malignant_recipe(fam, arm)
            low = text.lower()
            assert "data.txt" not in low and "challenge" not in low, (arm, text)
            if arm != "wrongflag_clean":
                assert "flag" not in low, (arm, text)
            assert fam.true_recipe not in text                      # L-5: no embedded truth
            assert "xor" not in low or arm == "wrongflag_clean" or "additive" in low   # L-7: no true transform class
            assert sig and all(isinstance(t, str) for t in sig)
        assert "truncate -s -16" in hard.malignant_recipe(fam, "destroy_clean")[0]
        assert "256 shifts" in hard.malignant_recipe(fam, "exhaust_clean")[0]
        assert hard._MALIGNANT_FLAG in hard.malignant_recipe(fam, "wrongflag_clean")[0]
    assert set(hard.MALIGNANT_ARMS) >= {"destroy", "exhaust", "wrongflag", "destroy_clean", "exhaust_clean", "wrongflag_clean"}


# ---------------------------------------------------------------- F-5 selection rule
def test_f5_balanced_selection_is_outer_balanced_and_free_of_poison_equals_helpful_pairs():
    names = sorted(hard._BY_NAME)
    chosen = hard.select_balanced_families(names, per_outer=3, seed=424242)
    from collections import Counter
    assert Counter(n.split("_")[0] for n in chosen) == {"b64": 3, "hex": 3, "b32": 3, "b85": 3}
    rec = {n: hard._BY_NAME[n].true_recipe for n in chosen}
    for a in chosen:
        for b in chosen:
            if a != b:
                assert hard._BY_NAME[a].poison_recipe != rec[b], (a, b)
                if a.split("_")[0] == b.split("_")[0]:
                    sib = hard._BY_NAME[a].poison_sibling
                    assert not (set(hard._inner_tags_of(sib)) <= set(hard._inner_tags_of(b))), (a, b, sib)
    # the review's failing case: the default 12-family draw contained poison(b85_rot13_zlib) == true(b85_gzip_rev)
    assert not ({"b85_rot13_zlib", "b85_gzip_rev"} <= set(hard.select_balanced_families(names, 3, seed=1)))


# ---------------------------------------------------------------- F24 / F-2 / F-3 / F-13 retrieval arms
def _retr(fams, cfg_over, reranker=None):
    from config import SolverConfig
    from retrieval.retriever import ContextRetriever
    from retrieval.context_resolution import ContextResolver
    from retrieval.reranker import MockReranker
    emb = MockEmbedder(dim=16, seed=1)
    store, truth = hard.build_seed_graph(emb, families=fams, distractor_arm=True)
    cfg = SolverConfig().with_overrides(embedding_dim=16, context_resolution="category", seed=1, **cfg_over)
    return store, truth, cfg, ContextRetriever(store, emb, config=cfg, reranker=reranker or MockReranker(),
                                               context_resolver=ContextResolver(cfg).fit(store))


def test_f24_recall_all_overrides_pool_to_store_size():
    from experiments.run_campaign import recall_all_overrides, base_config
    store, _ = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=["b64_rot13", "hex_xor42", "b32_zlib"], distractor_arm=True)
    n = len(store.get_all_embeddings()[0])
    cfg = base_config(**recall_all_overrides(store, {"M": 3}))
    assert cfg.N == n and cfg.R == n and n > 6


def test_f2_rerank_prior_is_sigmoid_of_the_cross_encoder_logit_and_is_logged():
    store, truth, cfg, retr = _retr(["b64_rot13", "hex_xor42", "b32_zlib"], {"N": 40, "R": 40, "M": 3, "prior_source": "rerank"})
    r = retr.retrieve("base64 alphabet artifact", category="base64", rng=np.random.RandomState(0), use_bandit=False)
    assert r.policy["prior_source"] == "rerank" and r.policy["pool_order"] == "mmr"
    for c in r.candidate_pool:
        assert abs(c["prior"] - 1.0 / (1.0 + np.exp(-c["rerank_score"]))) < 1e-9
    store, truth, cfg, retr = _retr(["b64_rot13", "hex_xor42", "b32_zlib"], {"N": 40, "R": 40, "M": 3})
    r = retr.retrieve("base64 alphabet artifact", category="base64", rng=np.random.RandomState(0), use_bandit=False)
    assert r.policy["prior_source"] == "cosine"
    for c in r.candidate_pool:
        assert abs(c["prior"] - max(0.0, c["cosine"])) < 1e-9


def test_f3_deterministic_arm_without_mmr_is_the_cross_encoder_top_m():
    from experiments.run_campaign import _conditions
    conds = dict((n, (spec, over)) for n, spec, over in _conditions(["B2det", "B2det_mmr", "oracle_static", "worst_static"]))
    assert conds["B2det"][0].use_mmr is False and conds["oracle_static"][0].use_mmr is False and conds["worst_static"][0].use_mmr is False
    assert conds["B2det_mmr"][0].use_mmr is True
    store, truth, cfg, retr = _retr(["b64_rot13", "hex_xor42", "b32_zlib"], {"N": 40, "R": 40, "M": 3})
    r = retr.retrieve("q", category="base64", rng=np.random.RandomState(0), use_bandit=False, use_mmr=False, deterministic=True)
    by_rr = sorted(r.candidate_pool, key=lambda c: -c["rerank_score"])
    assert r.sampled_ids == [c["node_id"] for c in by_rr[:3]] and r.policy["pool_order"] == "rerank"


def test_f13_forced_arms_are_clean_brackets():
    store, truth, cfg, retr = _retr(["b64_rot13", "hex_xor42", "b32_zlib"], {"N": 40, "R": 40, "M": 3})
    hid = {f: n for n, (l, f) in truth.items() if l == "helpful"}; pid = {f: n for n, (l, f) in truth.items() if l == "poison"}
    for label, inn, out in (("helpful", hid, pid), ("poison", pid, hid)):
        r = retr.retrieve("q", category="hex", rng=np.random.RandomState(0), use_bandit=False, use_mmr=False,
                          deterministic=True, on_target_family="hex_xor42", force_on_target=label)
        assert inn["hex_xor42"] in r.sampled_ids and out["hex_xor42"] not in r.sampled_ids and len(r.sampled_ids) == 3
        assert r.policy["forced"] == label and all(d["p"] == 1.0 for d in r.per_draw_propensity)


# ---------------------------------------------------------------- F25 gates
def test_f25_gates_infer_the_context_key_and_only_gate_learning_when_powered():
    from experiments import smoke_checks as sc
    from collections import Counter
    assert sc._context_key_of(["ctx::cat::base64", "ctx::cat::hex"]) == "alphabet"
    assert sc._context_key_of(["ctx::cat::global"]) == "global"
    assert sc._context_key_of(["ctx::cat::b64_rot13", "ctx::cat::hex_xor42"]) == "family"
    assert sc._context_key_of(["ctx::cluster::0", "ctx::cluster::2"]) == "cluster"
    assert sc._context_key_of(["ctx::cat::base64", "ctx::cluster::0"]) is None
    sc.GLOBAL.clear()
    sc.GLOBAL["deployed"] = {"context_key": None, "contexts": Counter({"ctx::cat::base64": 3, "ctx::cluster::0": 2}),
                             "n_train_deploy": 50, "learning": (-0.01, -0.05, 0.02), "incl_deploy_helpful": 0.3,
                             "pi_helpful_minus_chance": (0.01, -0.02, 0.04)}
    report, fails = [], []
    sc.check_f3_gates(report, fails)
    tags = [t for t, _ in fails]
    assert "F3-mechanics" in tags and "F3-learning" not in tags        # underpowered: learning not gated
    sc.GLOBAL["deployed"].update({"context_key": "alphabet", "n_train_deploy": 400, "learning": (-0.01, -0.05, 0.02)})
    report, fails = [], []
    sc.check_f3_gates(report, fails)
    assert [t for t, _ in fails] == ["F3-learning"]
    sc.GLOBAL["deployed"].update({"learning": (0.08, 0.02, 0.14)})
    report, fails = [], []
    sc.check_f3_gates(report, fails)
    assert not fails


def test_f4e_step_record_carries_finish_reason_and_parse_error_head():
    from instrumentation.records import StepRecord
    r = StepRecord(run_id="r", step_idx=1, action_type="parse_error", finish_reason="length",
                   llm_response_head="{\"action\": \"run_commands\", \"commands\": [\"base64 -d")
    d = r.to_dict()
    assert d["finish_reason"] == "length" and d["llm_response_head"].startswith("{\"action\"")
    assert StepRecord(run_id="r", step_idx=2).to_dict()["finish_reason"] == ""


# ---------------------------------------------------------------- F40 exposure update
def test_f40_exposure_update_decays_uncited_lessons_without_crediting_them(tmp_path):
    """cited_only_decay: the cited on-target helpful lesson gains positive Q; every other SAMPLED lesson
    gets an exposure update (n grows, Q relaxes toward zero advantage) instead of keeping its cold prior;
    under legacy cited_only the uncited lessons have no cell at all."""
    from pipeline.orchestrator import Orchestrator, SystemSpec
    from config import SolverConfig
    from bandit.bandit import TabularBandit
    from retrieval.retriever import ContextRetriever
    from retrieval.context_resolution import ContextResolver
    from retrieval.reranker import MockReranker
    fams = ["b64_rot13", "hex_xor42"]
    emb = MockEmbedder(dim=16, seed=1)
    store, truth = hard.build_seed_graph(emb, families=fams)
    hid = {f: n for n, (l, f) in truth.items() if l == "helpful"}

    def fake(ch, nodes, max_rounds, seed=None, **kw):
        h = [n for n in nodes if n.attributes.get("_truth") == "helpful" and n.attributes.get("_family") == ch["_family"]]
        cmds = [" ".join(h[0].attributes["invocation_signature"][:3]) + " data.txt"] if h else ["file data.txt"]
        return {"flag": ch["_flag"] if h else None, "trace": [{"round": 1, "action": "run_commands", "commands": cmds}],
                "rounds_used": 1, "gave_up_reason": None if h else "rounds_exhausted", "total_commands": 1}

    corpus = hard.generate_corpus(tmp_path / "c", n_per_family=8, seed=3, families=fams, diagnostics="charset")
    cells = {}
    for mode in ("cited_only", "cited_only_decay"):
        cfg = SolverConfig().with_overrides(N=40, R=30, M=3, embedding_dim=16, context_resolution="category", seed=1,
                                            credit_mode=mode, credit_signal="outcome", tau=1.0, epsilon_explore=0.0)
        bandit = TabularBandit(cfg)
        retr = ContextRetriever(store, emb, config=cfg, reranker=MockReranker(),
                                context_resolver=ContextResolver(cfg).fit(store), bandit=bandit)
        orch = Orchestrator(store, emb, cfg, SystemSpec("B4d", use_bandit=True), retriever=retr, bandit=bandit, solve_fn=fake)
        for k, ch in enumerate(corpus):
            orch.run_one(dict(ch), rng=np.random.RandomState(k), llm_seed=k)
        cells[mode] = {(c, n): (cell.Q, cell.n) for c, d in bandit.cells.items() for n, cell in d.items() if cell.n > 0}
    on = set(hid.values())
    legacy, decay = cells["cited_only"], cells["cited_only_decay"]
    assert all(n in on for (_, n) in legacy), "legacy cited_only updated an uncited lesson"
    exposed = {k: v for k, v in decay.items() if k[1] not in on}
    assert exposed, "decay mode exposed no uncited lesson"
    q0 = SolverConfig().optimistic_q0
    assert all(n > 0 and q < q0 for (q, n) in exposed.values()), exposed      # prior weight decays, Q relaxes toward 0
    credited = [q for (c, n), (q, _) in decay.items() if n in on]
    assert credited and max(credited) > max(q for (q, _) in exposed.values())  # cited helpful sits above exposed lessons


# ---------------------------------------------------------------- F41 learner options
def test_f41_running_mean_no_scaling_and_no_prior_are_exact_and_legacy_defaults_unchanged():
    from config import SolverConfig
    from bandit.bandit import TabularBandit
    cfg = SolverConfig().with_overrides(init_mode="zero", alpha_schedule="mean", distance_scaling="none", use_baseline_b=False)
    b = TabularBandit(cfg)
    for r in (1.0, 0.0, 0.5, 1.0):
        b.update("c", "v", r, similarity=0.1, success=r > 0)
    assert abs(b.cells["c"]["v"].Q - 0.625) < 1e-9 and b.cells["c"]["v"].n == 4       # exact running mean
    legacy = SolverConfig()
    assert legacy.alpha_schedule == "ema" and legacy.prior_source == "cosine" and legacy.distance_scaling.value == "asymmetric"
    # prior none -> every candidate's prior is 0 and the policy says so
    from retrieval.retriever import ContextRetriever
    from retrieval.context_resolution import ContextResolver
    from retrieval.reranker import MockReranker
    emb = MockEmbedder(dim=16, seed=1)
    store, truth = hard.build_seed_graph(emb, families=["b64_rot13", "hex_xor42"])
    cfg2 = SolverConfig().with_overrides(N=40, R=40, M=3, embedding_dim=16, context_resolution="category", seed=1, prior_source="none")
    retr = ContextRetriever(store, emb, config=cfg2, reranker=MockReranker(), context_resolver=ContextResolver(cfg2).fit(store))
    r = retr.retrieve("q", category="base64", rng=np.random.RandomState(0), use_bandit=False)
    assert r.policy["prior_source"] == "none" and all(c["prior"] == 0.0 for c in r.candidate_pool)


# ---------------------------------------------------------------- flag format (user request 2026-09-05)
def test_flag_format_is_configurable_and_legacy_is_byte_identical(monkeypatch, tmp_path):
    import solver.solver as ss
    # legacy: default pattern and prompt unchanged
    assert ss.compile_flag_pattern(None) is ss.FLAG_PATTERN and ss.compile_flag_pattern("picoCTF") is ss.FLAG_PATTERN
    legacy = ss.render_system_prompt("", [], advertise_memory_tools=False, max_rounds=3)
    assert legacy == ss.render_system_prompt("", [], advertise_memory_tools=False, max_rounds=3, flag_prefix="picoCTF")
    assert "picoCTF{" in legacy
    # generic and prefixed formats
    assert ss.extract_flag("got flag{s0me_r4nd0m_4b1c}", ss.compile_flag_pattern("flag")) == "flag{s0me_r4nd0m_4b1c}"
    assert ss.extract_flag("got picoCTF{s0me_r4nd0m_4b1c}", ss.compile_flag_pattern("flag")) is None
    anyp = ss.compile_flag_pattern("any")
    for f in ("HTB{d3lulu_1s_th3_s0lulu}", "gctf{c0ld_st0r4ge_9f}", "SEKAI{x_7a9b}", "flag{a_b_9x}"):
        assert ss.extract_flag("out: " + f, anyp) == f
    assert ss.extract_flag("out: flag{flag_here}", anyp) is None          # placeholders rejected for any prefix
    assert ss.flag_prefix_text("any") == "flag" and ss.flag_prefix_text("HTB") == "HTB"
    p = ss.render_system_prompt("", [], advertise_memory_tools=False, max_rounds=3, flag_prefix="flag")
    assert "picoCTF{" not in p and "flag{" in p
    # end to end: a solver that prints and submits flag{...} succeeds when the challenge declares flag_format
    monkeypatch.setattr(ss, "get_available_tools", lambda: [])
    f = tmp_path / "data.txt"; f.write_text("x")
    ch = {"name": "task-00000000", "category": "crypto", "points": 0, "difficulty": "unknown", "description": "Recover the flag.",
          "hints": [], "code_blocks": [], "local_files": [{"local_path": str(f), "text": "data.txt"}], "flag_format": "flag"}
    scripts = [json.dumps({"action": "run_commands", "commands": ["echo flag{r34l_0ne_77}"]}),
               json.dumps({"action": "submit_flag", "flag": "flag{r34l_0ne_77}"})]
    class _C:
        def __init__(s): s.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=s._c))
        def _c(s, **kw):
            m = types.SimpleNamespace(content=scripts.pop(0) if scripts else json.dumps({"action": "give_up", "reason": "x"}))
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=m, finish_reason="stop")], usage=types.SimpleNamespace(total_tokens=1))
    monkeypatch.setattr(ss, "client", _C())
    monkeypatch.setattr(ss, "run_commands", lambda commands, working_dir, log=None, **kw: [
        {"command": c, "stdout": "flag{r34l_0ne_77}\n", "stderr": "", "returncode": 0, "error": None} for c in commands])
    res = ss.solve_challenge(ch, max_rounds=3, advertise_memory_tools=False)
    assert res["flag"] == "flag{r34l_0ne_77}", res


def test_non_object_json_reply_is_a_parse_error_not_a_crash(monkeypatch, tmp_path):
    """Phase-2 launch 2026-09-06: the exhaust lesson made the model answer with a bare number, which
    _parse_llm_json accepted and `.get` then crashed the arm. Must be treated as a parse error."""
    import solver.solver as ss
    monkeypatch.setattr(ss, "get_available_tools", lambda: [])
    f = tmp_path / "data.txt"; f.write_text("x")
    ch = {"name": "task-00000000", "category": "crypto", "points": 0, "difficulty": "unknown", "description": "Recover the flag.",
          "hints": [], "code_blocks": [], "local_files": [{"local_path": str(f), "text": "data.txt"}]}
    scripts = ["5", "[1, 2]", "\"just a string\"", json.dumps({"action": "give_up", "reason": "done"})]
    class _C:
        def __init__(s): s.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=s._c))
        def _c(s, **kw):
            m = types.SimpleNamespace(content=scripts.pop(0))
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=m, finish_reason="stop")], usage=types.SimpleNamespace(total_tokens=1))
    monkeypatch.setattr(ss, "client", _C())
    res = ss.solve_challenge(ch, max_rounds=6, advertise_memory_tools=False)
    assert res["flag"] is None and res["gave_up_reason"] == "done"
    assert sum(1 for r in res["trace"] if r.get("action") == "parse_error") == 3


# ---------------------------------------------------------------- causal arm (P2-5)
def test_deployed_causal_condition_inherits_the_campaign_sampler():
    from experiments.run_campaign import _conditions
    name, spec, over = _conditions(["deployed_causal"])[0]
    assert spec.use_bandit and over["credit_signal"] == "causal" and over["causal_reroll_seeds"] == 3
    assert "tau" not in over and "epsilon_explore" not in over        # unlike the legacy `causal` arm


def test_causal_rerolls_parallel_equal_sequential_and_are_accounted(tmp_path):
    """Pre-drawn seeds make the parallel re-rolls reproduce the sequential Q updates exactly; the run
    record carries the re-roll compute (solves, tokens, wall)."""
    import threading
    from pipeline.orchestrator import Orchestrator, SystemSpec
    from config import SolverConfig
    from bandit.bandit import TabularBandit
    from retrieval.retriever import ContextRetriever
    from retrieval.context_resolution import ContextResolver
    from retrieval.reranker import MockReranker
    fams = ["b64_rot13", "hex_xor42"]
    emb = MockEmbedder(dim=16, seed=1)
    store, truth = hard.build_seed_graph(emb, families=fams)
    corpus = hard.generate_corpus(tmp_path / "c", n_per_family=3, seed=5, families=fams, diagnostics="charset")
    calls = []; lock = threading.Lock()

    def fake(ch, nodes, max_rounds, seed=None, order_seed=None, flag_checker=None):
        # deterministic in (context, seed): solve iff the on-target helpful is present and seed is even
        h = [n for n in nodes if n.attributes.get("_truth") == "helpful" and n.attributes.get("_family") == ch["_family"]]
        with lock:
            calls.append((tuple(sorted(n.id for n in nodes)), seed))
        cmds = [" ".join(h[0].attributes["invocation_signature"][:3]) + " data.txt"] if h else ["file data.txt"]
        solved = bool(h) and (seed or 0) % 2 == 0
        return {"flag": ch["_flag"] if solved else None, "trace": [{"round": 1, "action": "run_commands", "commands": cmds}],
                "rounds_used": 2, "gave_up_reason": None if solved else "rounds_exhausted", "total_commands": 1, "tokens": 1000}

    def run(workers):
        cfg = SolverConfig().with_overrides(N=40, R=30, M=3, embedding_dim=16, context_resolution="category", seed=1,
                                            credit_signal="causal", causal_reroll_seeds=3, causal_parallel_workers=workers,
                                            tau=1.0, epsilon_explore=0.0)
        bandit = TabularBandit(cfg)
        retr = ContextRetriever(store, emb, config=cfg, reranker=MockReranker(),
                                context_resolver=ContextResolver(cfg).fit(store), bandit=bandit)
        orch = Orchestrator(store, emb, cfg, SystemSpec("B4dc", use_bandit=True), retriever=retr, bandit=bandit, solve_fn=fake)
        recs = [orch.run_one(dict(ch), rng=np.random.RandomState(k), llm_seed=k) for k, ch in enumerate(corpus)]
        cells = {(c, n): round(cell.Q, 9) for c, d in bandit.cells.items() for n, cell in d.items()}
        return recs, cells

    recs_seq, cells_seq = run(1)
    recs_par, cells_par = run(6)
    assert cells_seq == cells_par and cells_seq, "parallel re-rolls must reproduce the sequential learner exactly"
    cited_runs = [r for r in recs_par if r.reroll_solves]
    assert cited_runs, "no causal re-roll happened"
    for r in cited_runs:
        assert r.reroll_solves % 3 == 0 and r.reroll_tokens == 1000 * r.reroll_solves and r.reroll_wall_s >= 0
    assert all(r.reroll_solves == 0 for r in recs_par if not r.reroll_solves)


def test_review0908_f6_batch_credit_n_is_observation_count():
    """Review 2026-09-08 F6: the observational arm wrote every batch cell with n = 1, so the sampling blend
    (n*Q + kappa*prior)/(n + kappa) with kappa = 4 and prior 0 sampled on Q/5. With the observation counts the
    cell carries the support the estimate has."""
    from bandit.bandit import TabularBandit
    from config import SolverConfig
    b = TabularBandit(SolverConfig(kappa=4.0))
    b.set_q_table({"c": {"h": 0.5, "p": -0.2}}, n_table={"c": {"h": 40, "p": 3}})
    assert b.cells["c"]["h"].n == 40 and b.cells["c"]["p"].n == 3
    b2 = TabularBandit(SolverConfig(kappa=4.0)); b2.set_q_table({"c": {"h": 0.5}})
    assert b2.cells["c"]["h"].n == 1.0        # legacy behaviour when no counts are given


def test_review0908_f6_observational_q_table_returns_counts():
    from ope.causal_value import observational_q_table
    import random
    rng = random.Random(0); runs = []
    pool = [{"node_id": f"v{i}", "score": 0.0, "cosine": 0.5, "prior": 0.0} for i in range(6)]
    for i in range(400):
        sampled = rng.sample([c["node_id"] for c in pool], 3)
        runs.append({"run_id": f"r{i}", "context_node_id": "c", "candidate_pool": [dict(c, pi_explore=0.5) for c in pool],
                     "sampled_set": sampled, "R_final": 1.0 if "v0" in sampled else 0.0, "temperature": 1.0,
                     "policy": {"epsilon": 1.0, "tau_deploy": 1.0, "design": "uniform", "deterministic": False}})
    table, counts = observational_q_table(runs, return_counts=True)
    assert set(table) == {"c"} and counts["c"]["v0"] > 100 and abs(table["c"]["v0"] - 1.0) < 0.2


def test_review0908_f3_draw_seed_offset(tmp_path):
    """Review 2026-09-08 F3: arms share the train draws (seed = hash(epoch, item)); a replication arm with a non-zero
    draw_seed_offset gets independent train seeds, offset 0 reproduces the campaign's byte-identical formula."""
    import hashlib
    from experiments.run_campaign import Campaign
    c0 = Campaign.__new__(Campaign); c0.draw_seed_offset = 0
    c1 = Campaign.__new__(Campaign); c1.draw_seed_offset = 1
    legacy = int(hashlib.sha1("train|0|b32_gzip_007".encode()).hexdigest()[:8], 16) % (2**31 - 1)
    assert c0.train_seed(0, "b32_gzip_007") == legacy
    assert c1.train_seed(0, "b32_gzip_007") != legacy
    assert c1.train_seed(0, "b32_gzip_007") != c1.train_seed(1, "b32_gzip_007")


def test_solver_coerces_malformed_commands_field():
    """2026-09-08: a run_commands action with a NESTED list (`[["a","b"]]`) crashed the loop detector's
    `sorted(commands)` (LOO re-run, one job); bare strings and non-string items are normalised too."""
    from solver.solver import _coerce_commands
    assert _coerce_commands([["base64 -d data.txt", "xxd -p"]]) == ["base64 -d data.txt", "xxd -p"]
    assert _coerce_commands("base64 -d data.txt") == ["base64 -d data.txt"]
    assert _coerce_commands(["a", 3, "", None, ["b"]]) == ["a", "3", "b"]
    assert _coerce_commands({"cmd": "x"}) == [] and _coerce_commands(None) == []


def test_context_overflow_is_retried_by_dropping_old_exchanges(monkeypatch, tmp_path):
    """2026-09-15: a 400 'maximum context length' rejection no longer ends the run; the oldest exchanges are dropped
    using the server's token count and the call is retried (byte-identical behaviour when no overflow occurs)."""
    import types, json as _json
    import solver.solver as ss
    calls = {"n": 0, "lens": []}
    class BadRequestError(Exception):
        status_code = 400
    def _create(**kw):
        calls["n"] += 1; calls["lens"].append(len(kw["messages"]))
        if len(kw["messages"]) >= 6 and calls["n"] == 3:      # round 3, first attempt: overflow
            raise BadRequestError("Error code: 400 - {'error': {'message': \"This model's maximum context length is 16384 tokens. "
                                  "However, you requested 4096 output tokens and your prompt contains at least 12289 input tokens, "
                                  "for a total of at least 16385 tokens. Please reduce the length of the input prompt or the number of output tokens.\"}}")
        if calls["n"] <= 2:
            body = {"action": "run_commands", "commands": [f"echo round{calls['n']}"]}
        elif calls["n"] == 4:
            body = {"action": "run_commands", "commands": ["echo picoCTF{ctx_retry_ok}"]}
        else:
            body = {"action": "submit_flag", "flag": "picoCTF{ctx_retry_ok}"}
        msg = types.SimpleNamespace(content=_json.dumps(body))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")], usage=types.SimpleNamespace(total_tokens=10))
    fake = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=_create)))
    monkeypatch.setattr(ss, "client", fake); monkeypatch.setattr(ss, "get_available_tools", lambda: [])
    ch = {"name": "t", "category": "misc", "points": 0, "difficulty": "unknown", "description": "x", "hints": [], "code_blocks": [], "local_files": []}
    res = ss.solve_challenge(ch, use_tools=True, use_memory=False, max_rounds=6, max_output_tokens=64, context_window=16384,
                             working_dir=str(tmp_path), flag_checker=lambda f: f == "picoCTF{ctx_retry_ok}", advertise_memory_tools=False)
    assert res["flag"] == "picoCTF{ctx_retry_ok}" and res["gave_up_reason"] is None
    assert calls["lens"][2] >= 6 and calls["lens"][3] < calls["lens"][2]          # the retry went out with fewer messages
    assert any(r.get("context_retrims") == 1 for r in res["trace"])


def test_phase3_parser_tolerates_qwen3_reply_shapes():
    """2026-09-15: Qwen3.x replies carry an orphan </think> (the template prefills the opening tag), sometimes drop the
    "commands" key, and sometimes put literal newlines inside JSON strings. Canonical replies parse as before."""
    from solver.solver import _parse_llm_json, _extract_thinking
    canon = '{"action": "run_commands", "commands": ["ls -la"]}'
    assert _parse_llm_json(canon) == {"action": "run_commands", "commands": ["ls -la"]}
    orphan = 'Let me look at the files first.\nI will list them.\n</think>\n\n' + canon
    assert _parse_llm_json(orphan) == {"action": "run_commands", "commands": ["ls -la"]}
    th, body = _extract_thinking(orphan); assert "list them" in th and body == canon
    bare = 'reasoning…</think> {"action": "run_commands", ["cat source.py", "cat output.txt"]}'
    assert _parse_llm_json(bare) == {"action": "run_commands", "commands": ["cat source.py", "cat output.txt"]}
    newline = '{"action": "run_commands", "commands": ["python3 -c \'print(1)\nprint(2)\'"]}'
    assert _parse_llm_json(newline)["commands"][0].startswith("python3 -c")
    assert _parse_llm_json("no json here at all") is None


def test_none_content_from_a_reasoning_parser_does_not_crash_the_run(monkeypatch, tmp_path):
    """2026-09-15: gpt-oss-20b served with vLLM's reasoning parser returned content=None on every call; the run
    crashed in round 1 with 'expected string or bytes-like object' (51/51). Now it is an empty reply."""
    import types, json as _json
    import solver.solver as ss
    calls = {"n": 0}
    def _create(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            msg = types.SimpleNamespace(content=None, reasoning_content="thinking only")
        else:
            msg = types.SimpleNamespace(content=_json.dumps({"action": "give_up", "reason": "done"}))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")], usage=types.SimpleNamespace(total_tokens=5))
    monkeypatch.setattr(ss, "client", types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=_create))))
    monkeypatch.setattr(ss, "get_available_tools", lambda: [])
    ch = {"name": "t", "category": "misc", "points": 0, "difficulty": "unknown", "description": "x", "hints": [], "code_blocks": [], "local_files": []}
    res = ss.solve_challenge(ch, use_tools=True, use_memory=False, max_rounds=4, max_output_tokens=64, working_dir=str(tmp_path), advertise_memory_tools=False)
    assert res["gave_up_reason"] != "llm_error" and calls["n"] >= 2
    assert res["trace"][0]["action"] == "parse_error" and res["trace"][0]["thinking"] == "thinking only"


def test_tool_calling_accepts_a_function_call_as_the_action(monkeypatch, tmp_path):
    """2026-09-15: with tool_calling=True the actions are offered as OpenAI tools and a returned tool call is executed
    exactly like the JSON-in-text action; with tool_calling=False no tools are sent (byte-identical request)."""
    import types, json as _json
    import solver.solver as ss
    seen = []
    def _create(**kw):
        seen.append(kw)
        if kw.get("tools"):
            n = sum(1 for k in seen if k.get("tools"))
            if n == 1:
                tc = types.SimpleNamespace(function=types.SimpleNamespace(name="run_commands", arguments=_json.dumps({"commands": ["echo picoCTF{tool_ok}"]})))
                msg = types.SimpleNamespace(content=None, tool_calls=[tc], reasoning_content="use the tool")
            else:
                tc = types.SimpleNamespace(function=types.SimpleNamespace(name="submit_flag", arguments=_json.dumps({"flag": "picoCTF{tool_ok}"})))
                msg = types.SimpleNamespace(content=None, tool_calls=[tc])
        else:
            msg = types.SimpleNamespace(content=_json.dumps({"action": "give_up", "reason": "x"}))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")], usage=types.SimpleNamespace(total_tokens=5))
    monkeypatch.setattr(ss, "client", types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=_create))))
    monkeypatch.setattr(ss, "get_available_tools", lambda: [])
    ch = {"name": "t", "category": "misc", "points": 0, "difficulty": "unknown", "description": "x", "hints": [], "code_blocks": [], "local_files": []}
    res = ss.solve_challenge(ch, use_tools=True, use_memory=False, max_rounds=4, max_output_tokens=64, working_dir=str(tmp_path),
                             flag_checker=lambda f: f == "picoCTF{tool_ok}", advertise_memory_tools=False, tool_calling=True)
    assert res["flag"] == "picoCTF{tool_ok}" and res["trace"][0].get("tool_call") is True and res["trace"][0]["thinking"] == "use the tool"
    assert all("tools" in k and k["tool_choice"] == "auto" for k in seen)
    seen.clear()
    ss.solve_challenge(ch, use_tools=True, use_memory=False, max_rounds=2, max_output_tokens=64, working_dir=str(tmp_path), advertise_memory_tools=False)
    assert seen and all("tools" not in k for k in seen)


def test_seed_oss_think_tags_are_stripped_like_qwen_ones():
    from solver.solver import _parse_llm_json, _extract_thinking
    canon = '{"action": "run_commands", "commands": ["ls"]}'
    s = "<seed:think>plan first</seed:think>\n" + canon
    th, body = _extract_thinking(s); assert th == "plan first" and body == canon
    assert _parse_llm_json(s) == {"action": "run_commands", "commands": ["ls"]}
    assert _parse_llm_json("thinking…</seed:think>" + canon) == {"action": "run_commands", "commands": ["ls"]}


def test_chat_template_kwargs_can_be_omitted_for_mistral_tokenizer_mode(monkeypatch, tmp_path):
    import types, json as _json
    import solver.solver as ss
    seen = []
    def _create(**kw):
        seen.append(kw); msg = types.SimpleNamespace(content=_json.dumps({"action": "give_up", "reason": "x"}))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")], usage=types.SimpleNamespace(total_tokens=5))
    monkeypatch.setattr(ss, "client", types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=_create))))
    monkeypatch.setattr(ss, "get_available_tools", lambda: [])
    ch = {"name": "t", "category": "misc", "points": 0, "difficulty": "unknown", "description": "x", "hints": [], "code_blocks": [], "local_files": []}
    ss.solve_challenge(ch, use_tools=True, use_memory=False, max_rounds=1, max_output_tokens=32, working_dir=str(tmp_path), advertise_memory_tools=False)
    assert seen[-1]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}      # default unchanged
    monkeypatch.setenv("CTF_NO_CHAT_TEMPLATE_KWARGS", "1")
    ss.solve_challenge(ch, use_tools=True, use_memory=False, max_rounds=1, max_output_tokens=32, working_dir=str(tmp_path), advertise_memory_tools=False)
    assert "extra_body" not in seen[-1]
