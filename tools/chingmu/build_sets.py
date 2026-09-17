#!/usr/bin/env python3
"""Build the chingmu training set, eval subset and manifests from validation output + policy.

Inputs
  --per-clip   validation per_clip.tsv (full run)
  --stage      staging dir (stage_manifest.tsv maps stems -> session incl. renamed conflicts)
  --pkl        converted PKL root (<session>/<stem>.pkl)
  --out        sets dir
Policy (user decisions 2026-09-14/15)
  keep everything except: validator FAIL (any gate), leaf categories in --drop-leaf,
  stems containing any --drop-substr (case-insensitive), and duplicate-identical clips.
Outputs
  <out>/<name>/<session>/<stem>.pkl        symlinks (training set)
  <out>/<name>_eval512/<stem>.pkl          flat symlinks, seeded 512 with dur < --eval-max-s
  <out>/<name>_clip_durations.csv          path,dur_s (from PKLs, post-trim)
  <out>/<name>_manifest.json, <out>/<name>_drops.tsv
"""
import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

import joblib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-clip", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="chingmu_train_v1")
    ap.add_argument("--drop-leaf", nargs="*", default=["1.7.3.9.Fall_and_Fall_Recovery"])
    ap.add_argument("--drop-substr", nargs="*", default=["climb"])
    ap.add_argument("--eval-n", type=int, default=512)
    ap.add_argument("--eval-max-s", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true", help="remove previously built set dirs of this name first")
    args = ap.parse_args()

    # stage manifest: src_rel -> (link stem, session)
    stage = {}
    with open(os.path.join(args.stage, "stage_manifest.tsv"), newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            stage[r["src_rel"]] = (r["stem"], r["session"], r["action"])

    rows = list(csv.DictReader(open(args.per_clip, newline=""), delimiter="\t"))
    rows.sort(key=lambda r: r["rel"])
    drops, keep = [], []
    seen_md5 = {}
    for r in rows:
        # byte-identical content under a different name (validator dup_content_n > 1): keep the first path only
        md5 = r.get("md5", "")
        if md5:
            if md5 in seen_md5:
                drops.append((r["rel"], "dup_content:" + seen_md5[md5])); continue
            seen_md5[md5] = r["rel"]
        st = stage.get(r["rel"])
        if st is None:
            drops.append((r["rel"], "not_staged")); continue
        stem, session, action = st
        if action == "skip_dup_identical":
            drops.append((r["rel"], "dup_identical")); continue
        if r["verdict"] != "PASS":
            drops.append((r["rel"], "gate:" + r["gate_failures"])); continue
        if r["leaf"] in args.drop_leaf:
            drops.append((r["rel"], "leaf:" + r["leaf"])); continue
        low = r["stem"].lower()
        hit = [s for s in args.drop_substr if s.lower() in low]
        if hit:
            drops.append((r["rel"], "substr:" + ",".join(hit))); continue
        pkl = os.path.join(args.pkl, session, stem + ".pkl")
        if not os.path.exists(pkl):
            drops.append((r["rel"], "pkl_missing")); continue
        keep.append((stem, session, pkl, r))

    train_dir = os.path.join(args.out, args.name)
    eval_dir_pre = os.path.join(args.out, f"{args.name}_eval{args.eval_n}")
    if args.clean:
        import shutil
        for d in (train_dir, eval_dir_pre):
            if os.path.isdir(d):
                shutil.rmtree(d)  # symlink dirs built by this script only
    os.makedirs(train_dir, exist_ok=True)
    durations = {}
    for stem, session, pkl, r in keep:
        d = os.path.join(train_dir, session)
        os.makedirs(d, exist_ok=True)
        link = os.path.join(d, stem + ".pkl")
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(pkl), link)
        e = next(iter(joblib.load(pkl).values()))
        durations[link] = e["dof"].shape[0] / float(e["fps"])

    dur_csv = os.path.join(args.out, f"{args.name}_clip_durations.csv")
    with open(dur_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "dur_s"])
        for p in sorted(durations):
            w.writerow([p, round(durations[p], 3)])

    # eval subset: seeded random sample of stems shorter than --eval-max-s
    eligible = sorted(os.path.splitext(os.path.basename(p))[0] for p, dsec in durations.items() if dsec < args.eval_max_s)
    random.seed(args.seed)
    chosen = set(random.sample(eligible, min(args.eval_n, len(eligible))))
    eval_dir = os.path.join(args.out, f"{args.name}_eval{args.eval_n}")
    os.makedirs(eval_dir, exist_ok=True)
    by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in durations}
    for s in sorted(chosen):
        link = os.path.join(eval_dir, s + ".pkl")
        if not os.path.lexists(link):
            os.symlink(os.path.realpath(by_stem[s]), link)
    with open(os.path.join(args.out, f"{args.name}_eval{args.eval_n}_stems.txt"), "w") as f:
        f.write("\n".join(sorted(chosen)) + "\n")

    with open(os.path.join(args.out, f"{args.name}_drops.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["src_rel", "reason"]); w.writerows(drops)
    per_leaf = defaultdict(lambda: [0, 0.0])
    for stem, session, pkl, r in keep:
        per_leaf[session][0] += 1; per_leaf[session][1] += durations[os.path.join(train_dir, session, stem + ".pkl")]
    man = {
        "name": args.name, "validated_clips": len(rows), "kept": len(keep), "dropped": len(drops),
        "kept_hours": round(sum(durations.values()) / 3600, 2),
        "drop_reasons": dict(Counter(reason.split(":")[0] if not reason.startswith("gate") else reason for _, reason in drops)),
        "drop_reason_groups": dict(Counter(reason.split(":")[0] for _, reason in drops)),
        "policy": {"drop_leaf": args.drop_leaf, "drop_substr": args.drop_substr, "gates": "validator PASS only"},
        "eval": {"n": len(chosen), "max_s": args.eval_max_s, "eligible": len(eligible), "seed": args.seed},
        "per_session": {k: {"clips": v[0], "hours": round(v[1] / 3600, 2)} for k, v in sorted(per_leaf.items())},
    }
    with open(os.path.join(args.out, f"{args.name}_manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    print(json.dumps({k: v for k, v in man.items() if k != "per_session"}, indent=2))
    n_links = sum(len(fs) for _, _, fs in os.walk(train_dir))
    print(f"train links: {n_links}  eval links: {len(os.listdir(eval_dir))}")


if __name__ == "__main__":
    main()
