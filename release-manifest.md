# Release manifest — curated public tree

Built 2026-08-24 by `release/build_release.py` from the private working repo.

**535 files, 25M.** Nothing is copied that is not on the include list below, so contaminated material is excluded by construction rather than scrubbed after the fact.

## Included

| path | why |
|---|---|
| `.env.example` | placeholder env template — verified to contain no real values |
| `.gitignore` | ignore rules [EXTENDED] |
| `CTF_SOLVER_ARCHITECTURE_v2.md` | the v2 spec — source of truth for the design |
| `LICENSE` | MIT for code/data; paper text all-rights-reserved |
| `OPEN_DECISIONS.md` | design decisions with reasoning and reversibility |
| `README.md` | repo entry point, written for a public audience |
| `appendix_c/` | the four artifact sets Appendix C cites, plus the scripts that produce them and the decisions log |
| `bandit` | contextual bandit + hygiene (C6, C9) |
| `benchmark` | the contamination-free generator — cited as benchmark/hard_ctf.py |
| `config` | config files |
| `config.py` | single-source swept/ablated parameters (§15) |
| `docs/CLAIMS_v2.md` | working claim ledger [SUPERSEDED BANNER ADDED] |
| `docs/DATA_PIPELINE.md` | data pipeline description |
| `docs/PREREG.md` | the pre-registration; every hypothesis is dispositioned |
| `docs/PREREG_INTERFERENCE.md` | the interference pre-registration |
| `docs/RELATED_WORK.md` | related-work notes |
| `docs/architecture.html` | visual model of the memory-management pipeline |
| `evaluation` | baselines B0-B6, paired bootstrap, datasets (C4) |
| `experiments` | campaign drivers and emitters cited throughout Appendix B [SANITIZED] |
| `infrastructure` | vLLM serving scripts |
| `instrumentation` | the §12 record schema and JSONL logger |
| `knowledge_graph` | typed-node lesson graph (C1) |
| `local_llm` | OpenAI-compatible server shim |
| `ope` | IPS/SNIPS/DR estimators — cited in Appendix B as ope/causal_value.py |
| `paper` | the manuscript sources, figures and compiled PDF |
| `pipeline` | orchestrator, system specs, causal credit (C8) |
| `requirements.txt` | dependencies |
| `results/campaign_v2` | the 4B causal-credit study (§8) — cited by the README, Appendix B and docs/PREREG.md |
| `results/lab` | lab-scale ladder artifacts — the load-bearing campaigns |
| `results/pilot` | 4B pilot artifacts + figures (§7) |
| `retrieval` | recall/rerank/MMR/propensity-logged sampling (C2) — the free-experiment machinery |
| `reward` | R = o(1+lambda*e) and grading (C5) |
| `run.sh` | convenience runner [SANITIZED] |
| `solver` | the solver and sandboxed tool executor |
| `tests` | test suite; runs in mock mode with no GPU |
| `writeback` | gated promotion + provenance (C7) |

## Excluded

The public tree omits internal working notes, session logs, legacy scratch directories, unrelated material, and machine-specific configuration. Nothing required to reproduce a reported number was withheld.

## Sanitized files

These were copied with machine-specific absolute paths replaced by environment variables, and with a third party's account name replaced by a generic current-user check. The GPU courtesy logic is preserved, only de-identified.

- `appendix_c/results/g1_verifier_acceptance_crypto_depth1.provenance.json`
- `appendix_c/results/g1_verifier_acceptance_sql.provenance.json`
- `appendix_c/results/g2_mechanism.provenance.json`
- `appendix_c/results/g2_step_contrast.provenance.json`
- `appendix_c/results/inband_selection_crypto.provenance.json`
- `experiments/lab_autorun.sh`
- `experiments/power_14b.sh`
- `experiments/power_14b_v2.sh`
- `experiments/power_32b.sh`
- `experiments/power_32b_tp2.sh`
- `experiments/power_72b.sh`
- `experiments/propensity_validate.py`
- `experiments/resume_32b.sh`
- `experiments/run_14b_queue.sh`
- `experiments/run_active_loo.sh`
- `experiments/run_boundary_loo.sh`
- `experiments/run_direct_loo.sh`
- `experiments/run_distractor.sh`
- `experiments/run_gemma27b.sh`
- `experiments/run_h5.sh`
- `experiments/run_internlm.sh`
- `experiments/run_llama.sh`
- `experiments/run_malignant.sh`
- `experiments/run_malignant_expand.sh`
- `experiments/run_mistral24b.sh`
- `experiments/run_nameleak_ab.sh`
- `experiments/run_sql_expand.sh`
- `experiments/run_struct_calib.sh`
- `experiments/run_yi34b.sh`
- `infrastructure/after_campaign_h5.sh`
- `infrastructure/serve_32b.sh`
- `infrastructure/serve_7b_dual.sh`
- `results/lab/campaign_14b_M3_powered/merged/crt_test.json`
- `results/lab/campaign_14b_M3_powered/merged/crt_test_outcome.json`
- `results/lab/campaign_32b_M3_tp2/merged/crt_test.json`
- `results/lab/campaign_32b_M3_tp2/merged/crt_test_outcome.json`
- `results/lab/campaign_3b_M3/merged/crt_test.json`
- `results/lab/campaign_3b_M3/merged/crt_test_outcome.json`
- `results/lab/campaign_7b35_M3_rep/merged/crt_test.json`
- `results/lab/campaign_7b35_M3_rep/merged/crt_test_outcome.json`
- `results/lab/campaign_llama8b_M3/merged/crt_test.json`
- `results/lab/campaign_llama8b_M3/merged/crt_test_outcome.json`
- `run.sh`

## Known gaps

- `appendix_c/select_families.py` defaults to `appendix_c/results/calib_tok768`, the calibration probe used to choose in-band families. That probe's raw output is not part of the artifact set Appendix C cites and is not shipped; the selection it produced is, at `appendix_c/results/inband_selection_crypto.json`.

