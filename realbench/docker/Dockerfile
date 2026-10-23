# realbench-solver:slim: small disposable executor for Cybench reference solutions (realbench/oracle_check.py)
# and, later, for the solver's shell tools.  Debian bookworm + Python 3.11 + the tools the file-only
# reference solutions need.  Deliberately NOT the 20 GB cybench/cybench agent image.
#
#   docker build -t realbench-solver:slim realbench/docker
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash coreutils netcat-openbsd curl git file xxd binutils cpp unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Versions pinned to what the tasks' metadata/requirements.sh ask for, so the `pip install X==v` a
# solution.sh runs is a no-op ("Requirement already satisfied") rather than a network download.
RUN pip install --no-cache-dir \
        pwntools==4.12.0 \
        pycryptodome==3.20.0 \
        gmpy2==2.2.1 \
        z3-solver==4.13.0.0 \
        sympy==1.13.1 \
        requests==2.31.0

WORKDIR /work
