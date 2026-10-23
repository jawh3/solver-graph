"""realbench - adapters that put REAL public CTF benchmarks behind the solver-graph harness.

First adapter: Cybench (Zhang et al. 2024, arXiv:2408.08926), 40 professional-CTF tasks.
See realbench/README.md for the field mapping, isolation model and what is not built yet.
"""
from realbench.cybench import (  # noqa: F401
    CybenchTask,
    DockerOps,
    FlagMatch,
    Provisioned,
    ProvisionError,
    RealDockerOps,
    check_flag,
    enumerate_tasks,
    flag_match,
    neutral_display_name,
    provision,
    to_challenge_dict,
)
