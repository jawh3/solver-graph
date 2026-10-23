"""
evaluation/datasets.py - dataset partitions controlling contamination (spec §14.1).

Three partitions, each tagged on every challenge so the eval can separate them:

  SEED-PUBLIC        contamination_flag=True  - picoCTF + scraped write-ups.
                     Diagnostic ONLY; never the headline (Qwen likely memorized).
  HELD-OUT-NOVEL     contamination_flag=False - challenges not public / not in
                     pretraining. The headline solve-rate number.
  TRANSFORMED-FAMILIES  semantics-preserving variants of public challenges
                     (Evolve-CTF style; CaptureTheFlags 2602.05523). Separates
                     *method learned* from *write-up memorized* - the headline
                     robustness number.

For the synthetic world, TRANSFORMED-FAMILIES is produced by `transform_family`:
it rewords/renames a challenge and the SURFACE form of its lessons while keeping
the underlying solution tools fixed - so a system that merely memorized the
write-up text fails, but one that learned the method still solves.
"""

from __future__ import annotations

import copy
from typing import Dict, List


def tag_partition(challenges: List[Dict], partition: str, contamination: bool) -> List[Dict]:
    out = []
    for ch in challenges:
        c = copy.deepcopy(ch)
        c["dataset"] = partition
        c["contamination_flag"] = contamination
        out.append(c)
    return out


def transform_family(challenge: Dict, variant_seed: int = 1) -> Dict:
    """Produce a semantics-preserving variant (renamed/reworded, same solution).

    The key property: `_world.solution_tools` is UNCHANGED (same method), but the
    name and description surface forms differ, so write-up-memorization does not
    transfer while method-learning does.
    """
    c = copy.deepcopy(challenge)
    cat = c.get("category", "misc")
    c["name"] = f"{c['name']}__var{variant_seed}"
    # reword description without naming the tools (defeat surface memorization)
    c["description"] = (f"A reworded {cat} task (variant {variant_seed}); "
                        f"the underlying method is unchanged but the wording differs.")
    c["dataset"] = "families"
    c["contamination_flag"] = False
    return c


def make_synthetic_partitions(world) -> Dict[str, List[Dict]]:
    """Split a synthetic world's challenges into the three eval partitions.

    Half the challenges become SEED-PUBLIC (contaminated); the other half become
    HELD-OUT-NOVEL; TRANSFORMED-FAMILIES are variants of the seed-public ones.
    """
    chals = world.challenges
    half = len(chals) // 2
    seed_public = tag_partition(chals[:half], "seed", contamination=True)
    held_out = tag_partition(chals[half:], "heldout", contamination=False)
    families = [transform_family(ch, variant_seed=1) for ch in seed_public]
    return {"seed": seed_public, "heldout": held_out, "families": families}
