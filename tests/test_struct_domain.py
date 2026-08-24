"""struct_domain (PREREG_INTERFERENCE 2nd domain) must be NON-degenerate: own method recovers the flag,
sibling method gives a clean-but-WRONG non-empty string, and the flag is NOT readable without the
tag-filter (no printable-column shortcut -> avoids the SQL ceiling). Pure CPU."""
import numpy as np
import benchmark.struct_domain as sd


def test_grid_size_and_widths():
    assert len(sd.GRID_FAMILIES) >= 75            # PREREG needs 75 in-band (50 floor) -> ample pool
    assert {f.width for f in sd.GRID_FAMILIES} == {20, 24, 28, 32}


def test_roundtrip_sibling_and_shortcut():
    v = sd.verify_corpus(n=24, seed=3)
    for fam, d in v.items():
        assert d["own_recovers"], fam
        assert d["sibling_fails"], fam
        assert d["no_unfiltered_shortcut"], fam   # flag NOT readable by reading a column unfiltered


def test_poison_is_clean_nonempty_wrong():
    rng = np.random.RandomState(7)
    for fam in sd.GRID_FAMILIES[:8]:
        flag = sd._rand_flag(rng); raw = fam.encode(flag, rng)
        out = sd._sibling(fam).decode(raw)
        assert out is not None and out != flag and len(out) == len(flag)   # plausible-but-wrong, executable
