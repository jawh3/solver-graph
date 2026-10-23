#!/usr/bin/env python3
"""experiments/nyu_lesson_sources.py - what each NYU lesson writer saw.

For every lesson in the pool, re-collects the reference material the writer was given (realbench.nyuctf.gather_sources:
the task description, README / hints / write-ups and reference solver scripts in the development task's folder, never
its handouts) and records which of it was present.

"Saw a solution" was judged by reading each writer's sources (2026-10-22): a solver script counts, and so does a README or
write-up that describes the solution; a README that gives only the title, description and flag does not. The 18 writers
judged to have no solution are listed in NO_SOLUTION; every one of them has no solver script among its sources, so the
file lists in the output let a reader check the judgment. Needs a NYU CTF Bench checkout ($NYUCTF_ROOT).

    python experiments/nyu_lesson_sources.py [--lessons <lessons.jsonl>] [--out <tsv>]
"""
import argparse
import json
import re
from pathlib import Path

from realbench import nyuctf

REPO = Path(__file__).resolve().parents[1]
LESSONS = REPO / "results/realbench_nyu_mem/qwen_qwen3.8-27b-fp8/lessons.jsonl"
NO_SOLUTION = {
    "2013q-msc-life", "2013q-msc-networking_1", "2013q-msc-networking_2", "2014f-rev-return_of_the_weiner",
    "2014f-pwn-kernel", "2014q-rev-aerosol_can", "2016f-web-seizure_cipher", "2016f-cry-killer_cipher",
    "2016f-pwn-detective", "2016q-msc-regexpire", "2016q-msc-coinslot", "2016q-pwn-warmup", "2016q-pwn-tutorial",
    "2016q-pwn-hungman", "2015f-rev-wyvern2", "2015q-for-sharpturn", "2015q-cry-bricks_of_gold", "2015q-for-flash",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lessons", default=str(LESSONS))
    ap.add_argument("--out", default=str(Path(LESSONS).with_name("lesson_sources.tsv")))
    args = ap.parse_args()
    ids = [json.loads(l)["task_id"] for l in open(args.lessons) if l.strip()]
    tasks = {t.id: t for t in nyuctf.enumerate_tasks(split="development", task_ids=ids)}
    assert NO_SOLUTION <= set(ids), sorted(NO_SOLUTION - set(ids))
    rows = ["task_id\tsaw_solution\twriteup_files\tsolver_scripts"]
    n_saw = 0
    for tid in ids:
        src = nyuctf.gather_sources(tasks[tid])
        d = Path(tasks[tid].task_dir)
        is_file = lambda name: (d / name).is_file() or any(p.is_file() for p in d.rglob(Path(name).name))
        # gather_sources heads each file with "### <name>"; headings inside a README look the same, so keep real files
        docs = [x for x in re.findall(r"^### (.+)$", src.get("writeups", ""), re.M) if is_file(x)]
        scripts = [x for x in re.findall(r"^### (.+)$", src.get("solver_scripts", ""), re.M) if is_file(x)]
        saw = tid not in NO_SOLUTION
        assert saw or not scripts, f"{tid}: judged to have no solution but a solver script was among its sources"
        n_saw += saw
        rows.append(f"{tid}\t{int(saw)}\t{','.join(docs) or '-'}\t{','.join(scripts) or '-'}")
    Path(args.out).write_text("\n".join(rows) + "\n")
    print(f"{len(ids)} lessons: {n_saw} writers saw a solution, {len(ids) - n_saw} did not -> {args.out}")


if __name__ == "__main__":
    main()
