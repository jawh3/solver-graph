#!/usr/bin/env python3
# Usage: PYTHONPATH=<repo> python experiments/nyu_egress_check.py   (2026-10-20 review follow-up; result: PASS, host unreachable + no internet from the extra-network tasks)
"""Live: provision philanthropy (philnet), shreeramquest, and one crypto task; confirm no container reaches the internet,
the egress_violations check returns clean, services are reachable, host ports closed."""
import os, subprocess, uuid, shutil
from pathlib import Path
from realbench import nyuctf, cybench as cb
want = {"2023q-web-philanthropy", "2023f-web-shreeramquest", "2021f-cry-interoperable"}
ts = [t for t in nyuctf.enumerate_tasks(split="test") if t.id in want and not t.exclusion_reason]
d = cb.RealDockerOps(internal=True); d.ensure_network(nyuctf.SHARED_NET)
root = Path("results/_work/egress_live"); root.mkdir(parents=True, exist_ok=True)
# the host's own LAN address must be unreachable too ($HOST_LAN_IP, else the first address `hostname -I` reports)
HOST_IP = os.environ.get("HOST_LAN_IP") or (subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split() or [""])[0]
ok = True
for t in ts:
    wd = root / f"{t.display_name}-{uuid.uuid4().hex[:6]}"; prov = None
    try:
        prov = nyuctf.provision(t, wd, docker=d, service_timeout_s=180)
        viol = d.egress_violations(prov.network)
        # probe the internet + host from a container on the solve network
        r = subprocess.run(["docker","run","--rm","--network",prov.network,"--entrypoint","bash","cybench/cybench:latest","-c",
            "getent hosts api.github.com || echo NO-DNS; "
            "own=$(ip -4 route | awk '/default/{print $3}'); for h in $own 172.17.0.1 172.20.0.1 " + HOST_IP + "; do for p in 22 80 5000 8010; do timeout 2 bash -c \"exec 3<>/dev/tcp/$h/$p\" 2>/dev/null && echo HOST-OPEN-$h:$p; done; done; echo DONE"],
            capture_output=True,text=True,timeout=120).stdout
        host_hits=[x for x in r.split() if x.startswith("HOST-OPEN-")]
        clean = not viol and "NO-DNS" in r and not host_hits and bool(prov.health) and all(prov.health.values())
        ok &= clean
        print(f"{t.id:30s} nets={prov.networks} health={prov.health} egress_violations={viol or 'none'} internet={'NO-DNS' if 'NO-DNS' in r else 'RESOLVED!'} host_open={host_hits or 'none'} -> {'OK' if clean else 'FAIL'}")
    except Exception as e:
        ok=False; print(f"{t.id:30s} EXCEPTION {type(e).__name__}: {e}")
    finally:
        if prov is not None:
            try: prov.teardown(verify_source=False)
            except Exception as e: print("  teardown", e)
        shutil.rmtree(wd, ignore_errors=True)
left=[n for n in subprocess.run(["docker","network","ls","--format","{{.Name}}"],capture_output=True,text=True).stdout.split() if n.startswith("ctfnet-")]
print("per-solve networks left:", left or "none")
print("PASS" if ok and not left else "FAIL")
