"""
ope/causal_value.py - observational causal credit from the sampling randomization.

The key idea (the cheap counterpart to active LOO re-rollout): the stochastic
retrieval already RANDOMIZES which lessons enter context, with a propensity we log.
So whether lesson v is in context is a (stochastically) assigned treatment, and the
causal effect of v on the run reward is identifiable from the logged runs by
inverse-propensity weighting - with NO extra inference.

For lesson v in context c, with marginal inclusion propensity π_i(v) on run i:

    V_treat(v,c)  = Σ_{i: v∈M_i, c_i=c}  R_i / π_i(v)        / Σ 1/π_i(v)     (SNIPW)
    V_control(v,c)= Σ_{i: v∉M_i, c_i=c}  R_i / (1-π_i(v))    / Σ 1/(1-π_i(v))
    ATE(v,c)      = V_treat - V_control            # v's average marginal value

A poison lesson the agent recovers from contributes the same reward whether present
or not -> ATE ≈ 0 (and ≤ 0 if it costs steps). A genuinely useful lesson raises the
reward when present -> ATE > 0. This recovers the causal ranking the outcome credit
could not -- from the SAME logs the outcome bandit was fooled by.

π_i(v) (the MARGINAL probability v is among the M sampled, sampling-without-
replacement) is estimated by Monte-Carlo from the logged per-candidate scores +
temperature (cheap, no GPU); we already log the score components per candidate.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from retrieval.sampling import softmax_temperature, stratified_design_marginals


def _stable_seed(s: str, salt: int = 0) -> int:
    """Deterministic 31-bit seed from a string (CPython's str hash() is per-process
    randomized via PYTHONHASHSEED - using it for the propensity MC made the headline
    promotion rate non-reproducible across runs; sha1 fixes that). `salt` lets an
    ablation vary the MC seed deterministically (the ≥10-seed sign-stability check)."""
    h = hashlib.sha1(f"{salt}:{s}".encode()).hexdigest()
    return int(h[:8], 16)            # 32-bit; np.random.RandomState accepts < 2**32


def marginal_inclusion_propensities(candidate_pool: List[Dict], m: int, tau: float,
                                    n_mc: int = 400, seed: int = 0) -> Dict[str, float]:
    """Monte-Carlo estimate of P(node ∈ sampled M) for each candidate, w/o replacement.

    Uses the logged per-candidate `score` and the run temperature - the exact inputs
    to the sequential softmax that produced the realized draw.
    """
    ids = [c["node_id"] for c in candidate_pool]
    scores = np.array([c["score"] for c in candidate_pool], dtype=np.float64)
    n = len(ids)
    m = min(m, n)
    if n == 0:
        return {}
    if m >= n:
        return {ids[i]: 1.0 for i in range(n)}     # everything is always included
    # Sequential softmax-without-replacement sampling is Plackett-Luce; its marginal
    # top-m inclusion probability equals the Gumbel-top-m law on utilities = score/τ.
    # Vectorizing over the n_mc trials (no Python inner loop) is ~100x faster - required
    # at scale (the pilot's per-step loop made re-scoring infeasible for ~150 lessons).
    tau = max(float(tau), 1e-6)
    rng = np.random.RandomState(seed)
    util = scores / tau
    keys = util[None, :] + rng.gumbel(size=(n_mc, n))      # (n_mc, n)
    topm = np.argpartition(keys, n - m, axis=1)[:, n - m:]  # indices of the m largest per trial
    counts = np.bincount(topm.reshape(-1), minlength=n)
    incl = counts / n_mc
    return {ids[i]: float(incl[i]) for i in range(n)}


def run_inclusion_propensities(run: Dict, m: int, n_mc: int = 400, seed: int = 0):
    """F3: marginal inclusion propensity under the run's ACTUAL logging policy.

    - deterministic runs (policy.deterministic) carry no randomization -> None (skipped by
      every observational estimator; they are the deployed arm, evaluated by paired solve rate).
    - epsilon-mixture runs: pi = (1-eps)*pi(tau_deploy) + eps*pi(tau_explore), the marginal
      inclusion probability of the mixture policy that actually generated the log.
    - otherwise: the logged temperature (legacy behaviour, byte-identical numbers).
    """
    pool = run.get("candidate_pool", [])
    pol = run.get("policy") or {}
    if pol.get("deterministic"):
        return None
    eps = float(pol.get("epsilon", 0.0) or 0.0)
    if eps > 0.0:
        pd = marginal_inclusion_propensities(pool, m, float(pol["tau_deploy"]), n_mc=n_mc, seed=seed)
        pe = exploration_propensities(run, m, n_mc=n_mc, seed=seed + 1)
        return {k: (1.0 - eps) * pd[k] + eps * pe.get(k, 0.0) for k in pd}
    tau = float(run.get("temperature", 1.0))
    if tau < 0.2:
        # review 2026-09-04 (#4): a greedy log without an exploration stream (legacy low-tau logs AND
        # the deployed arm's frozen eval runs, where eps=0) has no usable randomization; refuse it
        # rather than emit clipped garbage. Eval arms are compared by paired solve rate instead.
        return None
    return marginal_inclusion_propensities(pool, m, tau, n_mc=n_mc, seed=seed)


def exploration_propensities(run: Dict, m: int, n_mc: int = 400, seed: int = 1) -> Dict[str, float]:
    """The exploration STREAM's own inclusion propensity for every candidate of a run:
    exact (`pi_explore` logged, or reconstructed from `on_target` flags with the exact stratified /
    uniform marginals) for fixed designs, softmax at tau_explore for the legacy design."""
    pool = run.get("candidate_pool", [])
    pol = run.get("policy") or {}
    design = pol.get("design", "softmax")
    if design in ("stratified", "uniform") and pool and all("pi_explore" in c for c in pool):
        return {c["node_id"]: float(c["pi_explore"]) for c in pool}
    if design in ("stratified", "uniform"):
        n = len(pool); k = sum(1 for c in pool if c.get("on_target")) if design == "stratified" else 0
        pi_ot, pi_rest = stratified_design_marginals(n, k, m)
        return {c["node_id"]: (pi_ot if (design == "stratified" and c.get("on_target")) else pi_rest) for c in pool}
    return marginal_inclusion_propensities(pool, m, float(pol.get("tau_explore", run.get("temperature", 1.0))),
                                           n_mc=n_mc, seed=seed)


def _run_covariates(run: Dict) -> np.ndarray:
    """Observable per-run covariates for the per-lesson DR outcome model (must NOT
    include the focal lesson's own treatment indicator). Used by AIPW only."""
    pool = {c["node_id"]: c for c in run.get("candidate_pool", [])}
    sampled = run.get("sampled_set", [])
    cos = [float(pool[n]["cosine"]) for n in sampled if n in pool and pool[n].get("cosine") is not None]
    pri = [float(pool[n].get("prior", 0.0)) for n in sampled if n in pool]
    ctx_code = (_stable_seed(run.get("context_node_id", "")) % 997) / 997.0   # nuisance level
    return np.array([1.0, float(len(sampled)),
                     float(np.mean(cos)) if cos else 0.0,
                     float(np.mean(pri)) if pri else 0.0,
                     ctx_code], dtype=np.float64)


def _collect_obs(runs: List[Dict], clip: float = 0.05, mc_seed: int = 0,
                 n_mc: int = 400, with_cov: bool = False,
                 response_key: str = "R_final", exploration_only: bool = False,
                 audit: Optional[Dict] = None) -> Dict[str, List]:
    """Per-lesson observations from logged runs (single propensity-MC pass, shared by
    SNIPW and AIPW). Returns {node_id: [(R, T, pi, x_or_None), ...]} where T=1[v∈M_i],
    pi is the clipped marginal inclusion propensity, x is the run covariate vector
    (only when with_cov). DETERMINISTIC: the MC seed is sha1(run_id) salted by mc_seed.
    response_key selects the response: "R_final" (full reward o·(1+λe)) or "outcome"
    (binary solve - budget/step-rescaling INVARIANT, the panel's headline estimand)."""
    obs: Dict[str, List] = defaultdict(list)
    n_skipped = 0; n_clipped = 0; n_obs = 0
    for run in runs:
        pool = run.get("candidate_pool", [])
        if not pool:
            continue
        if exploration_only and not run.get("exploration"):
            continue
        sampled = set(run.get("sampled_set", []))
        m = len(sampled)
        R = float(run.get(response_key, 0.0))
        if exploration_only:
            pi = exploration_propensities(run, m, n_mc=n_mc,
                                          seed=_stable_seed(run.get("run_id", ""), salt=mc_seed) + 1)
        else:
            pi = run_inclusion_propensities(run, m, n_mc=n_mc,
                                            seed=_stable_seed(run.get("run_id", ""), salt=mc_seed))
        if pi is None:            # F3: deterministic / degenerate run -> no randomization to exploit
            n_skipped += 1
            continue
        x = _run_covariates(run) if with_cov else None
        for c in pool:
            v = c["node_id"]
            raw = pi.get(v, 0.0)
            p = min(max(raw, clip), 1.0 - clip)
            n_obs += 1; n_clipped += int(raw < clip or raw > 1.0 - clip)
            obs[v].append((R, 1 if v in sampled else 0, p, x))
    if audit is not None:
        audit.update({"runs_skipped_no_randomization": n_skipped, "obs": n_obs,
                      "clipped_frac": (n_clipped / n_obs if n_obs else None)})
    return obs


def _snipw_value(obs):
    """Self-normalized IPW value of a set of (R, weight) obs + effective sample size."""
    if not obs:
        return None, 0.0
    w = np.array([x[1] for x in obs]); r = np.array([x[0] for x in obs])
    sw = w.sum()
    if sw <= 0:
        return None, 0.0
    val = float((w * r).sum() / sw)
    ess = float(sw * sw / np.sum(w * w))    # Kish effective sample size
    return val, ess


def _hardened_ate(treat, control, min_obs=5, ess_frac=0.3):
    """ATE = SNIPW(treat) - SNIPW(control), rejected (None) when unreliable.

    Reject if either arm has < min_obs observations or effective sample size below
    ess_frac of its count (positivity/overlap guard against exploded weights).
    """
    vt, et = _snipw_value(treat)
    vc, ec = _snipw_value(control)
    nt, nc = len(treat), len(control)
    if nt < min_obs or nc < min_obs:
        return None
    if et < ess_frac * nt or ec < ess_frac * nc:
        return None
    if vt is None or vc is None:
        return None
    return vt - vc


def _snipw_diff_ci(treat, control, n_boot: int, seed: int, alpha: float = 0.05):
    """Per-lesson ATE bootstrap CI: stratified resample of the lesson's treated and
    control (R, w) observations independently (the unit resampled here is the RUN that
    contributes to this single lesson - the well-powered arm), recompute SNIPW(treat) -
    SNIPW(control). Returns (ci_lo, ci_hi, p_two_sided)."""
    if n_boot <= 0 or not treat or not control:
        return None, None, None
    rng = np.random.RandomState(seed)
    tr = np.array(treat, dtype=np.float64)        # columns: R, w
    co = np.array(control, dtype=np.float64)
    nt, nc = len(tr), len(co)
    ests = np.empty(n_boot)
    for b in range(n_boot):
        ti = rng.randint(0, nt, size=nt); ci = rng.randint(0, nc, size=nc)
        wt = tr[ti, 1]; rt = tr[ti, 0]; wc = co[ci, 1]; rc = co[ci, 0]
        vt = (wt * rt).sum() / wt.sum() if wt.sum() > 0 else 0.0
        vc = (wc * rc).sum() / wc.sum() if wc.sum() > 0 else 0.0
        ests[b] = vt - vc
    lo = float(np.percentile(ests, 100 * alpha / 2))
    hi = float(np.percentile(ests, 100 * (1 - alpha / 2)))
    p = min(1.0, 2 * min(float(np.mean(ests <= 0)), float(np.mean(ests >= 0))))
    return lo, hi, p


def observational_lesson_values(
    runs: List[Dict],
    truth: Optional[Dict[str, Tuple[str, str]]] = None,
    clip: float = 0.05,
    per_context: bool = False,
    min_obs: int = 5,
    ess_frac: float = 0.3,
    mc_seed: int = 0,
    n_mc: int = 400,
    n_boot: int = 0,
    boot_seed: int = 0,
    response_key: str = "R_final",
    exploration_only: bool = False,
    audit: Optional[Dict] = None,
) -> Dict:
    """Estimate each lesson's causal value (ATE on reward) from logged runs via SNIPW.
    exploration_only (review 2026-09-04): use ONLY exploration-stream runs with their exact design
    propensities - the clean randomized slice; the mixture estimate is the secondary one.

    runs: RunRecords (need candidate_pool with scores, sampled_set, R_final, temperature,
          context_node_id, M = len(sampled_set)).
    truth: optional node_id -> (label, family) for evaluation (helpful/poison/distractor).
    mc_seed: salts the (deterministic, sha1-based) propensity Monte-Carlo seed - vary it
             for the ≥10-seed sign-stability ablation.
    n_boot>0: also compute a per-lesson bootstrap CI (resampling the lesson's own runs).
    Returns {node_id: {ate, v_treat, v_control, n_treat, n_control, ess_treat, ess_control,
             ate_ci_lo, ate_ci_hi, ate_p, label}}.
    """
    # review 2026-09-04 (#1): the positivity audit is returned through the optional `audit` dict,
    # NEVER inside the per-lesson values mapping (every consumer indexes d["ate"]).
    obs = _collect_obs(runs, clip=clip, mc_seed=mc_seed, n_mc=n_mc, with_cov=False,
                       response_key=response_key, exploration_only=exploration_only,
                       audit=(audit if audit is not None else {}))
    out = {}
    for v, rows in obs.items():
        t = [(R, 1.0 / p) for (R, T, p, _) in rows if T == 1]
        k = [(R, 1.0 / (1.0 - p)) for (R, T, p, _) in rows if T == 0]
        vt, et = _snipw_value(t); vc, ec = _snipw_value(k)
        ate = _hardened_ate(t, k, min_obs=min_obs, ess_frac=ess_frac)   # None if unreliable
        ci_lo = ci_hi = ate_p = None
        if ate is not None and n_boot > 0:
            ci_lo, ci_hi, ate_p = _snipw_diff_ci(
                t, k, n_boot, seed=_stable_seed(v, salt=boot_seed))
        out[v] = {"ate": ate, "v_treat": vt, "v_control": vc,
                  "n_treat": len(t), "n_control": len(k),
                  "ess_treat": et, "ess_control": ec,
                  "ate_ci_lo": ci_lo, "ate_ci_hi": ci_hi, "ate_p": ate_p,
                  "ate_excl0": (None if ci_lo is None else (ci_lo > 0 or ci_hi < 0)),
                  "label": (truth.get(v, ("?", "?"))[0] if truth else None)}
    return out


def _aipw_lesson(rows, min_obs: int, ess_frac: float, ridge: float = 1e-2):
    """Per-lesson doubly-robust (AIPW) ATE from rows=[(R, T, pi, x)...].

    Outcome model: ridge regression of R on [x, T, x·T] (separate treated/control
    surfaces). AIPW influence: (μ1-μ0) + T(R-μ1)/π - (1-T)(R-μ0)/(1-π). Consistent if
    EITHER the propensity OR the outcome model is right (the robustness anchor for the
    SNIPW headline). Returns (ate, influence_vector) or (None, None) if under-powered /
    positivity-violating (same gates as SNIPW)."""
    R = np.array([r[0] for r in rows], dtype=np.float64)
    T = np.array([r[1] for r in rows], dtype=np.float64)
    pi = np.array([r[2] for r in rows], dtype=np.float64)
    X = np.array([r[3] for r in rows], dtype=np.float64)
    nt = int(T.sum()); nc = int(len(T) - nt)
    if nt < min_obs or nc < min_obs:
        return None, None
    wt = T / pi; wc = (1.0 - T) / (1.0 - pi)
    esst = wt.sum() ** 2 / max(np.sum(wt ** 2), 1e-12)
    essc = wc.sum() ** 2 / max(np.sum(wc ** 2), 1e-12)
    if esst < ess_frac * nt or essc < ess_frac * nc:
        return None, None
    Z = np.hstack([X, T[:, None], X * T[:, None]])
    A = Z.T @ Z + ridge * np.eye(Z.shape[1])
    theta = np.linalg.lstsq(A, Z.T @ R, rcond=None)[0]
    ones = np.ones((len(X), 1)); zeros = np.zeros((len(X), 1))
    mu1 = np.hstack([X, ones, X * 1.0]) @ theta
    mu0 = np.hstack([X, zeros, X * 0.0]) @ theta
    infl = (mu1 - mu0) + T * (R - mu1) / pi - (1.0 - T) * (R - mu0) / (1.0 - pi)
    return float(np.mean(infl)), infl


def observational_lesson_values_aipw(
    runs: List[Dict],
    truth: Optional[Dict[str, Tuple[str, str]]] = None,
    clip: float = 0.05, min_obs: int = 5, ess_frac: float = 0.3,
    mc_seed: int = 0, n_mc: int = 400, n_boot: int = 0, boot_seed: int = 0,
    response_key: str = "R_final",
) -> Dict:
    """Per-lesson DOUBLY-ROBUST (AIPW) ATE - same interface/return shape as
    observational_lesson_values, with `ate` from AIPW. The robustness anchor for the
    A.7/the plan 'sign-flip stable across {SNIPW, DR}' ablation."""
    obs = _collect_obs(runs, clip=clip, mc_seed=mc_seed, n_mc=n_mc, with_cov=True,
                       response_key=response_key)
    out = {}
    for v, rows in obs.items():
        ate, infl = _aipw_lesson(rows, min_obs, ess_frac)
        ci_lo = ci_hi = ate_p = None
        if ate is not None and n_boot > 0 and infl is not None:
            rng = np.random.RandomState(_stable_seed(v, salt=boot_seed))
            n = len(infl); ests = np.empty(n_boot)
            for b in range(n_boot):
                ests[b] = infl[rng.randint(0, n, size=n)].mean()
            ci_lo = float(np.percentile(ests, 2.5)); ci_hi = float(np.percentile(ests, 97.5))
            ate_p = min(1.0, 2 * min(float(np.mean(ests <= 0)), float(np.mean(ests >= 0))))
        nt = sum(1 for r in rows if r[1] == 1)
        out[v] = {"ate": ate, "v_treat": None, "v_control": None,
                  "n_treat": nt, "n_control": len(rows) - nt,
                  "ate_ci_lo": ci_lo, "ate_ci_hi": ci_hi, "ate_p": ate_p,
                  "ate_excl0": (None if ci_lo is None else (ci_lo > 0 or ci_hi < 0)),
                  "label": (truth.get(v, ("?", "?"))[0] if truth else None)}
    return out


def observational_q_table(runs: List[Dict], tau0: float = 4.0, clip: float = 0.05,
                          mc_seed: int = 0, min_obs: int = 5, ess_frac: float = 0.3,
                          return_counts: bool = False):
    """Per-(context, lesson) causal value (ATE) for the ONLINE observational bandit.

    Estimates ATE(v | context c) by IPW within each context, shrunk toward the
    node's global ATE for sparse cells (partial pooling, pseudo-count tau0). Returns
    {context_id: {node_id: q}} that can be written straight into the bandit's cells -
    a fully self-contained, no-re-roll credit signal driven by the sampling
    randomization the system already performs.
    """
    # global per-node treat/control sums (for shrinkage target): [Σ R*w, Σ w, Σ w², n]
    g_treat = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    g_ctrl = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    c_treat = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0, 0]))
    c_ctrl = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0, 0]))
    c_count = defaultdict(lambda: defaultdict(int))

    for run in runs:
        pool = run.get("candidate_pool", [])
        if not pool:
            continue
        ctx = run.get("context_node_id", "")
        sampled = set(run.get("sampled_set", []))
        m = len(sampled)
        R = float(run.get("R_final", 0.0))
        pi = run_inclusion_propensities(run, m, seed=_stable_seed(run.get("run_id", ""), salt=mc_seed))
        if pi is None:
            continue
        for c in pool:
            v = c["node_id"]
            p = min(max(pi.get(v, 0.0), clip), 1 - clip)
            if v in sampled:
                w = 1.0 / p
                t = c_treat[ctx][v]; t[0] += R * w; t[1] += w; t[2] += w * w; t[3] += 1
                g = g_treat[v]; g[0] += R * w; g[1] += w; g[2] += w * w; g[3] += 1
                c_count[ctx][v] += 1
            else:
                w = 1.0 / (1.0 - p)
                k = c_ctrl[ctx][v]; k[0] += R * w; k[1] += w; k[2] += w * w; k[3] += 1
                g = g_ctrl[v]; g[0] += R * w; g[1] += w; g[2] += w * w; g[3] += 1
                c_count[ctx][v] += 1

    def _val_ess(acc):
        sw = acc[1]
        if sw <= 0 or acc[3] == 0:
            return None, 0.0, 0
        return acc[0] / sw, (sw * sw / acc[2] if acc[2] > 0 else 0.0), acc[3]

    def _ate(treat, ctrl):
        vt, et, nt = _val_ess(treat); vc, ec, nc = _val_ess(ctrl)
        if vt is None or vc is None:
            return None
        if nt < min_obs or nc < min_obs:                 # too few observations
            return None
        if et < ess_frac * nt or ec < ess_frac * nc:     # weights too concentrated (positivity)
            return None
        return vt - vc

    g_ate = {v: _ate(g_treat[v], g_ctrl[v]) for v in g_treat}

    table = defaultdict(dict)
    for ctx in c_count:
        for v in c_count[ctx]:
            local = _ate(c_treat[ctx][v], c_ctrl[ctx][v])
            glob = g_ate.get(v)
            n = c_count[ctx][v]
            if local is None and glob is None:
                continue
            if local is None:
                q = glob
            elif glob is None:
                q = local
            else:
                q = (n * local + tau0 * glob) / (n + tau0)   # partial pooling
            table[ctx][v] = float(q)
    out = {k: dict(v) for k, v in table.items()}
    if return_counts:
        # review 2026-09-08 F6: the number of runs in which the lesson was SAMPLED in this context - the batch
        # analogue of the online learner's n, so the sampling blend (n*Q + kappa*prior)/(n + kappa) weights the
        # batch estimate as it would weight an online estimate with the same support (n = 1 shrank every cell to Q/5).
        counts = {ctx: {v: int(c_treat[ctx][v][3]) for v in out[ctx]} for ctx in out}
        return out, counts
    return out


def outcome_credit_values(runs: List[Dict], truth: Optional[Dict] = None,
                          citations: Optional[List[Dict]] = None) -> Dict:
    """The OUTCOME-credit value per lesson, computed directly from logged runs (the value
    the online outcome bandit converges toward, so H1 can be checked from a fixed-policy log
    WITHOUT a separate online-bandit condition): mean R over runs where the lesson was in
    the credit-eligible set. If `citations` (citation records) are given, eligibility is
    sampled-AND-cited (the spec's cited-only credit); else sampled. A poison the agent
    recovers from is credited the success reward -> value > 0 (the boundary outcome credit
    cannot escape). Returns {node_id: {value, n, label}}."""
    cited_by_run = defaultdict(set)
    if citations:
        for c in citations:
            if c.get("cited"):
                cited_by_run[c.get("run_id")].add(c.get("node_id"))
    by_node = defaultdict(list)
    for run in runs:
        R = float(run.get("R_final", 0.0))
        rid = run.get("run_id")
        sampled = set(run.get("sampled_set", []))
        elig = (sampled & cited_by_run.get(rid, set())) if citations else sampled
        for v in elig:
            by_node[v].append(R)
    return {v: {"value": float(np.mean(rs)), "n": len(rs),
                "label": (truth.get(v, ("?", "?"))[0] if truth else None)}
            for v, rs in by_node.items()}


def summarize_by_label(values: Dict) -> Dict:
    """Mean ATE grouped by ground-truth label (helpful/poison/distractor) for eval."""
    agg = defaultdict(list)
    for v, d in values.items():
        if d["ate"] is not None and d["label"]:
            agg[d["label"]].append(d["ate"])
    return {lab: {"mean_ate": float(np.mean(xs)), "n": len(xs),
                  "std": float(np.std(xs))} for lab, xs in agg.items()}


# --------------------------------------------------------------------------- #
# label-level (resample-over-LESSONS) CIs - the generalization claim  #
# --------------------------------------------------------------------------- #

def _boot_mean_ci(xs, n_boot: int, seed: int, alpha: float = 0.05):
    """Bootstrap CI + two-sided p(mean==0) for the mean of xs, resampling the ELEMENTS
    of xs (here each element is one lesson's point estimate → resampling over lessons)."""
    xs = np.asarray(xs, dtype=np.float64)
    n = len(xs)
    if n == 0:
        return None
    rng = np.random.RandomState(seed)
    means = xs[rng.randint(0, n, size=(n_boot, n))].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    p = min(1.0, 2 * min(float(np.mean(means <= 0)), float(np.mean(means >= 0))))
    return {"mean": float(xs.mean()), "ci_lo": lo, "ci_hi": hi, "p": p, "n": n,
            "excl0": (lo > 0 or hi < 0)}


def _boot_mean_ci_clustered(xs, clusters, n_boot: int, seed: int, alpha: float = 0.05):
    """F9: cluster bootstrap for the mean of xs - resample CLUSTERS (families) with
    replacement and carry every element of each drawn cluster. Correct when several lessons
    share a family (they share the family's control runs and instances)."""
    xs = np.asarray(xs, dtype=np.float64)
    by = defaultdict(list)
    for x, c in zip(xs, clusters):
        by[c].append(x)
    keys = sorted(by); k = len(keys)
    if k == 0:
        return None
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.randint(0, k, size=k)
        pooled = np.concatenate([np.asarray(by[keys[i]]) for i in draw])
        means[b] = pooled.mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    p = min(1.0, 2 * min(float(np.mean(means <= 0)), float(np.mean(means >= 0))))
    return {"mean": float(xs.mean()), "ci_lo": lo, "ci_hi": hi, "p": p, "n": int(len(xs)),
            "n_clusters": int(k), "excl0": (lo > 0 or hi < 0), "cluster_bootstrap": True}


def summarize_by_label_ci(values: Dict, n_boot: int = 10000, seed: int = 0,
                          family_of: Optional[Dict[str, str]] = None) -> Dict:
    """Label-level mean ATE with a bootstrap CI obtained by resampling over LESSONS. `values` = observational_lesson_values()
    output (uses each lesson's point `ate`). Returns {label: {mean, ci_lo, ci_hi, p, n,
    excl0}}.

    F9: pass `family_of` = {node_id: family} to resample FAMILIES instead of lessons
    (cluster bootstrap). With one lesson per label per family the two coincide; with
    several lessons per family, or when lessons of a family share control runs, the
    clustered interval is the honest one."""
    agg = defaultdict(list); cl = defaultdict(list)
    for v, d in values.items():
        if d.get("ate") is not None and d.get("label"):
            agg[d["label"]].append(d["ate"])
            cl[d["label"]].append((family_of or {}).get(v, v))
    if family_of:
        return {lab: _boot_mean_ci_clustered(xs, cl[lab], n_boot, seed) for lab, xs in agg.items() if xs}
    return {lab: _boot_mean_ci(xs, n_boot, seed) for lab, xs in agg.items() if xs}


def _values_by_label(value_by_node: Dict[str, Optional[float]],
                     label_by_node: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    out = defaultdict(dict)
    for nid, val in value_by_node.items():
        if val is None or nid not in label_by_node:
            continue
        out[label_by_node[nid]][nid] = float(val)
    return out


def promotion_rate_ci(value_by_node: Dict[str, Optional[float]],
                      label_by_node: Dict[str, str], theta: float = 0.0,
                      n_boot: int = 10000, seed: int = 0) -> Dict:
    """Per-label promote-if-(value>θ) rate with a bootstrap CI over LESSONS.
    `value_by_node` may be a {node: Q} (outcome credit) or {node: ATE} (causal credit).
    Returns {label: {rate, ci_lo, ci_hi, n}}."""
    by_lab = _values_by_label(value_by_node, label_by_node)
    out = {}
    for lab, d in by_lab.items():
        ind = np.array([1.0 if v > theta else 0.0 for v in d.values()])
        n = len(ind)
        if n == 0:
            continue
        rng = np.random.RandomState(seed)
        rates = ind[rng.randint(0, n, size=(n_boot, n))].mean(axis=1)
        out[lab] = {"rate": float(ind.mean()),
                    "ci_lo": float(np.percentile(rates, 2.5)),
                    "ci_hi": float(np.percentile(rates, 97.5)), "n": n}
    return out


def promotion_gap_ci(value_a: Dict[str, Optional[float]],
                     value_b: Dict[str, Optional[float]],
                     label_by_node: Dict[str, str], theta: float = 0.0,
                     n_boot: int = 10000, seed: int = 0) -> Dict:
    """Per-label PAIRED promotion-rate gap (signal A - signal B) on the SAME lessons,
    bootstrap CI over lessons. Used for H3 (outcome-Q vs causal-ATE poison promotion).
    Returns {label: {gap, rate_a, rate_b, ci_lo, ci_hi, p, n}}."""
    out = {}
    by_lab_nodes = defaultdict(list)
    for nid, lab in label_by_node.items():
        if value_a.get(nid) is not None and value_b.get(nid) is not None:
            by_lab_nodes[lab].append(nid)
    for lab, nodes in by_lab_nodes.items():
        a = np.array([1.0 if value_a[nid] > theta else 0.0 for nid in nodes])
        b = np.array([1.0 if value_b[nid] > theta else 0.0 for nid in nodes])
        diff = a - b
        n = len(diff)
        if n == 0:
            continue
        rng = np.random.RandomState(seed)
        boot = diff[rng.randint(0, n, size=(n_boot, n))].mean(axis=1)
        p = min(1.0, 2 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0))))
        out[lab] = {"gap": float(diff.mean()), "rate_a": float(a.mean()),
                    "rate_b": float(b.mean()), "ci_lo": float(np.percentile(boot, 2.5)),
                    "ci_hi": float(np.percentile(boot, 97.5)), "p": p, "n": n}
    return out
