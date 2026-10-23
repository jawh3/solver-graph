#!/usr/bin/env python3
"""experiments/phase2_channels.py - review 2026-09-08 F8/F9: per arm, (a) the frozen-eval propensity YARDSTICK
(mean deploy propensity of the on-target helpful, the on-target poison and the mean off-target lesson, plus the
poison's rank among the non-helpful lessons), (b) the recovery-MASKING channel from the update log (share of the
on-target poison's credited updates that occur in solved runs / with a positive Q step, same for the helpful),
(c) the final Q and n of on-target vs off-target lessons per context, and the number of contexts in which the
on-target poison's Q ended positive.
Usage: python -m experiments.phase2_channels --root results/lab/phase2_14b --arms a,b,... --out docs/...md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import importlib as _il
hard = _il.import_module(os.environ.get("CTF_DOMAIN", "benchmark.hard_ctf"))
from ope.causal_value import marginal_inclusion_propensities
from retrieval.embedder import MockEmbedder

_fam_of = lambda cid: cid.rsplit("_", 1)[0]


def load(root, arm, kind):
    out = []
    for f in glob.glob(f"{root}/arms/{arm}/shard_00/logs/campaign_*/{kind}.jsonl"):
        out += [json.loads(l) for l in open(f) if l.strip()]
    return out


def is_train(r):
    pol = r.get("policy") or {}
    return (r.get("phase") == "train") if r.get("phase") else (bool(pol.get("explore_enabled", True)) and not pol.get("deterministic"))


def analyse(root, arm, hid, pid, truth, n_mc=300):
    runs = load(root, arm, "runs"); ups = load(root, arm, "updates")
    by_id = {r["run_id"]: r for r in runs}
    train = [r for r in runs if is_train(r)]; ev = [r for r in runs if not is_train(r) and not (r.get("policy") or {}).get("deterministic")]
    # membership: family -> its majority context in train
    ctx_of = defaultdict(Counter)
    for r in train:
        ctx_of[_fam_of(r["challenge_id"])][r.get("context_node_id")] += 1
    member = {f: c.most_common(1)[0][0] for f, c in ctx_of.items()}
    # (a) frozen-eval yardstick
    yh, yp, yoff, rank = [], [], [], []
    for k, r in enumerate(ev):
        f = _fam_of(r["challenge_id"]); h, p = hid.get(f), pid.get(f); pool = r["candidate_pool"]
        if h is None or p is None or not pool:
            continue
        tau = float((r.get("policy") or {}).get("tau_deploy", r.get("temperature", 1.0)) or 1.0)
        pis = marginal_inclusion_propensities(pool, len(r["sampled_set"]), max(tau, 1e-3), n_mc=n_mc, seed=k)
        off = [v for n, v in pis.items() if n not in (h, p) and not n.startswith("distract::")]
        yh.append(pis.get(h, 0.0)); yp.append(pis.get(p, 0.0)); yoff.append(float(np.mean(off)) if off else np.nan)
        nonh = sorted([v for n, v in pis.items() if n != h], reverse=True)
        rank.append((1 + sum(1 for v in nonh if v > pis.get(p, 0.0))) / max(len(nonh), 1))
    yard = {"n_eval": len(yh), "pi_helpful": float(np.mean(yh)), "pi_poison": float(np.mean(yp)), "pi_off_target_mean": float(np.nanmean(yoff)),
            "poison_rank_frac_median": float(np.median(rank)), "chance": float(np.mean([len(r["sampled_set"]) / len(r["candidate_pool"]) for r in ev if r.get("candidate_pool")]))}
    # (b) masking channel from credited updates on train runs
    ch = {k: {"n": 0, "solved": 0, "positive": 0, "marg": []} for k in ("helpful_on", "poison_on", "off_target")}
    for u in ups:
        if not u.get("credit_eligible"):
            continue
        r = by_id.get(u["run_id"])
        if r is None or not is_train(r):
            continue
        f = _fam_of(r["challenge_id"]); nid = u["node_id"]
        lab = "helpful_on" if nid == hid.get(f) else "poison_on" if nid == pid.get(f) else "off_target"
        c = ch[lab]; c["n"] += 1; c["solved"] += int(bool(r["outcome"])); c["positive"] += int(u["post_Q"] > u["pre_Q"] + 1e-12)
        rs = str(u.get("reason", ""))
        if rs.startswith("causal:margR="):          # causal credit: the logged marginal reward R(with) - mean R(re-rolls without)
            try:
                c["marg"].append(float(rs.split("margR=")[1].split("(")[0]))
            except ValueError:
                pass
    for c in ch.values():
        c["share_solved"] = c["solved"] / c["n"] if c["n"] else None; c["share_positive"] = c["positive"] / c["n"] if c["n"] else None
        m = np.array(c.pop("marg"))
        if len(m):
            c["margR"] = {"n": int(len(m)), "mean": float(m.mean()), "share_pos": float((m > 1e-9).mean()), "share_zero": float((np.abs(m) <= 1e-9).mean()),
                          "share_neg": float((m < -1e-9).mean())}
    # (c) final Q by (label, on/off target of the context)
    state_files = glob.glob(f"{root}/arms/{arm}/shard_00/state/*/bandit.json")
    q = defaultdict(list); pos_ctx = 0; n_ctx_with_poison = 0
    if state_files:
        st = json.load(open(state_files[0]))
        for ctx, cells in st.get("cells", {}).items():
            for nid, cell in cells.items():
                lab, fam = truth.get(nid, ("other", None))
                if lab not in ("helpful", "poison"):
                    continue
                on = member.get(fam) == ctx
                q[(lab, "on" if on else "off")].append((float(cell["Q"]), float(cell["n"])))
                if lab == "poison" and on:
                    n_ctx_with_poison += 1; pos_ctx += int(cell["Q"] > 0)
    finalq = {f"{lab}_{side}": {"cells": len(v), "mean_Q": float(np.mean([x[0] for x in v])), "mean_n": float(np.mean([x[1] for x in v]))}
              for (lab, side), v in q.items()}
    return {"arm": arm, "n_train": len(train), "contexts": sorted(set(member.values())), "yardstick": yard, "masking": ch, "final_Q": finalq,
            "on_target_poison_cells": n_ctx_with_poison, "on_target_poison_Q_positive": pos_ctx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--arms", required=True); ap.add_argument("--out", default=None)
    ap.add_argument("--n-mc", type=int, default=300)
    args = ap.parse_args()
    fams = open(f"{args.root}/families.txt").read().strip().split(",")
    _, truth = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=fams)
    hid = {f: n for n, (l, f) in truth.items() if l == "helpful"}; pid = {f: n for n, (l, f) in truth.items() if l == "poison"}
    res = {a: analyse(args.root, a, hid, pid, truth, args.n_mc) for a in args.arms.split(",")}
    L = ["# Phase-2 channels (review 2026-09-08 F8/F9): eval yardstick, recovery masking, final Q", "",
         "## (a) Frozen-eval deploy propensity per lesson (chance = M / pool)", "",
         "| arm | π helpful (on-target) | π poison (on-target) | mean π off-target lesson | poison / off-target | poison rank among non-helpful (median frac) | chance |",
         "|---|---|---|---|---|---|---|"]
    for a, r in res.items():
        y = r["yardstick"]
        L.append(f"| {a} | {y['pi_helpful']:.3f} | {y['pi_poison']:.3f} | {y['pi_off_target_mean']:.3f} | {y['pi_poison'] / y['pi_off_target_mean']:.2f}× | {y['poison_rank_frac_median']:.2f} | {y['chance']:.3f} |")
    L += ["", "## (b) Recovery-masking channel: credited updates on train runs", "",
          "| arm | on-target poison: updates, in solved runs, positive steps | on-target helpful: updates, solved, positive | off-target: updates, solved, positive |",
          "|---|---|---|---|"]
    for a, r in res.items():
        m = r["masking"]; f = lambda k: (f"{m[k]['n']}, {m[k]['share_solved']:.1%}, {m[k]['share_positive']:.1%}" if m[k]["n"] else "-")
        L.append(f"| {a} | {f('poison_on')} | {f('helpful_on')} | {f('off_target')} |")
    if any("margR" in r["masking"][k] for r in res.values() for k in r["masking"]):
        L += ["", "Causal-credit arms - logged marginal reward margR = R(run) - mean R(re-rolls without the lesson), per credited update:", "",
              "| arm | on-target poison: mean margR, share > 0 / = 0 / < 0 | on-target helpful | off-target |", "|---|---|---|---|"]
        for a, r in res.items():
            m = r["masking"]
            if not any("margR" in m[k] for k in m):
                continue
            g = lambda k: (f"{m[k]['margR']['mean']:+.3f}, {m[k]['margR']['share_pos']:.1%} / {m[k]['margR']['share_zero']:.1%} / {m[k]['margR']['share_neg']:.1%} (n={m[k]['margR']['n']})" if "margR" in m[k] else "-")
            L.append(f"| {a} | {g('poison_on')} | {g('helpful_on')} | {g('off_target')} |")
    L += ["", "## (c) Final Q (mean over cells; n = mean credited count) by label and whether the lesson's family belongs to the context", "",
          "| arm | helpful on-target Q (n) | poison on-target Q (n) | helpful off-target Q (n) | poison off-target Q (n) | on-target poison cells with Q > 0 |",
          "|---|---|---|---|---|---|"]
    for a, r in res.items():
        q = r["final_Q"]; f = lambda k: (f"{q[k]['mean_Q']:+.3f} ({q[k]['mean_n']:.1f}; {q[k]['cells']} cells)" if k in q else "-")
        L.append(f"| {a} | {f('helpful_on')} | {f('poison_on')} | {f('helpful_off')} | {f('poison_off')} | {r['on_target_poison_Q_positive']}/{r['on_target_poison_cells']} |")
    out = args.out or f"{args.root}/CHANNELS.md"
    Path(out).write_text("\n".join(L) + "\n"); Path(out).with_suffix(".json").write_text(json.dumps(res, indent=1)); print("\n".join(L))


if __name__ == "__main__":
    main()
