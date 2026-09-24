"""Fetch tournament datasets straight from the Gradients API on the machine that
will use them — never through a slow intermediary.

Task JSONs (`/v1/tasks/{id}`) carry signed S3 URLs for every image/caption pair
(valid ~7 days from task creation). For each task id this writes
`<out>/<id8>_<family>/NNN.png|.txt` plus `task.json`, exactly the layout
`prepare_cache.py` and `bootstrap_box.sh` expect.

    python tools/fetch_datasets.py --out /workspace/sn56/research/datasets --tournament tourn_30736259a9429dc0_20260921
    python tools/fetch_datasets.py --out ... --task <uuid> [--task <uuid> ...]
"""

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import re
import urllib.request
from urllib.parse import parse_qs, urlparse

API = "https://api.gradients.io"


def get(u, timeout=60):
    return json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "curl/8"}), timeout=timeout))


def fetch_bytes(u, timeout=120):
    return urllib.request.urlopen(u, timeout=timeout).read()


def url_alive(u):
    q = parse_qs(urlparse(u).query)
    try:
        t0 = dt.datetime.strptime(q["X-Amz-Date"][0], "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
        return t0 + dt.timedelta(seconds=int(q["X-Amz-Expires"][0])) > dt.datetime.now(dt.timezone.utc)
    except Exception:
        return True


def task_ids_for_tournament(tid):
    d = get(f"{API}/tournament/{tid}/details")
    ids = set()
    for r in d.get("rounds") or []:
        for t in r.get("tasks") or []:
            ids.add(t["task_id"])
    for b in d.get("boss_round_performance") or []:
        ids.add(b["task_id"])
    return sorted(ids)


def fetch_task(tid, out, workers=8):
    d = get(f"{API}/v1/tasks/{tid}")
    pairs = d.get("image_text_pairs") or []
    if not pairs:
        return tid, 0, 0, "no pairs"
    if not url_alive(pairs[0]["image_url"]):
        return tid, 0, len(pairs), "urls expired"
    dest = os.path.join(out, f"{tid[:8]}_{d.get('model_type')}")
    os.makedirs(dest, exist_ok=True)

    def one(i_p):
        i, p = i_p
        ext = os.path.splitext(urlparse(p["image_url"]).path)[1] or ".png"
        fi, ft = os.path.join(dest, f"{i:03d}{ext}"), os.path.join(dest, f"{i:03d}.txt")
        if os.path.exists(fi) and os.path.exists(ft) and os.path.getsize(fi) > 0:
            return True
        for attempt in range(3):
            try:
                img = fetch_bytes(p["image_url"]); txt = fetch_bytes(p["text_url"])
                open(fi, "wb").write(img); open(ft, "wb").write(txt)
                return True
            except Exception:
                continue
        return False

    with cf.ThreadPoolExecutor(workers) as ex:
        ok = sum(1 for r in ex.map(one, enumerate(pairs)) if r)
    json.dump({k: v for k, v in d.items() if k != "image_text_pairs"}, open(os.path.join(dest, "task.json"), "w"), indent=1)
    return tid, ok, len(pairs), dest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--tournament", action="append", default=[], help="tournament id, e.g. tourn_30736259a9429dc0_20260921")
    p.add_argument("--task", action="append", default=[], help="task uuid")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    ids = set(a.task)
    for t in a.tournament:
        ids.update(task_ids_for_tournament(t))
    os.makedirs(a.out, exist_ok=True)
    total = 0
    for tid in sorted(ids):
        try:
            tid, ok, n, note = fetch_task(tid, a.out, a.workers)
        except Exception as e:
            print(f"  {tid[:8]}  FAIL {type(e).__name__}: {e}")
            continue
        total += ok
        print(f"  {tid[:8]}  {ok:3d}/{n:<3d}  {note}")
    print(f"total pairs on disk: {total}")


if __name__ == "__main__":
    main()
