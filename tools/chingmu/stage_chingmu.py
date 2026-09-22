#!/usr/bin/env python3
"""Build a flat, deduplicated staging tree for convert_soma_csv_to_motion_lib.py --individual.

MotionDecode is nested 2-3 levels deep and contains a byte-identical duplicate
top-level folder plus a few repeated stems across categories. The converter's
--individual mode expects <input>/<session>/*.csv and the motion library keys
clips by basename, so this script:
  * walks the source tree (skipping --exclude-dirs),
  * maps every leaf category to a sanitized session dir name,
  * symlinks each CSV as <stage>/<leaf>/<file>.csv,
  * keeps only the FIRST occurrence (sorted path) of a repeated stem and logs the
    rest (byte-identical or conflicting) in stage_manifest.tsv.
Nothing is copied or modified; the stage dir holds symlinks only.
"""
import argparse
import csv
import hashlib
import os
import re
from collections import defaultdict


def sanitize(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--exclude-dirs", nargs="*", default=["Standing_High_Jump"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = []
    for dirpath, dirnames, filenames in os.walk(args.root):
        rel_dir = os.path.relpath(dirpath, args.root)
        if rel_dir.split(os.sep)[0] in args.exclude_dirs:
            dirnames[:] = []
            continue
        for fn in filenames:
            if fn.endswith(".csv") and "joint_report" not in fn:
                files.append((os.path.relpath(os.path.join(dirpath, fn), args.root)))
    files.sort()

    leaf_map, used = {}, {}
    for rel in files:
        leaf = os.path.basename(os.path.dirname(rel))
        if leaf not in leaf_map:
            s = sanitize(leaf)
            if s in used and used[s] != leaf:
                raise SystemExit(f"sanitized leaf collision: {leaf} vs {used[s]} -> {s}")
            used[s] = leaf
            leaf_map[leaf] = s

    seen = {}
    rows = []
    n_link = n_dup_same = n_dup_diff = 0
    os.makedirs(args.stage, exist_ok=True)
    for rel in files:
        src = os.path.join(args.root, rel)
        stem = os.path.splitext(os.path.basename(rel))[0]
        leaf = leaf_map[os.path.basename(os.path.dirname(rel))]
        link_name = os.path.basename(rel)
        action = "link"
        if stem in seen:
            same = md5(src) == md5(os.path.join(args.root, seen[stem]))
            if same:
                rows.append([stem, leaf, rel, "skip_dup_identical", seen[stem]])
                n_dup_same += 1
                continue
            # different content under a repeated stem (e.g. Sadness clips misnamed *_Intense_Joy_*):
            # keep it under a category-prefixed unique name so the motion library key stays unique.
            link_name = f"{leaf}__{os.path.basename(rel)}"
            action = "link_renamed_CONFLICT"
            n_dup_diff += 1
        else:
            seen[stem] = rel
        dst_dir = os.path.join(args.stage, leaf)
        dst = os.path.join(dst_dir, link_name)
        rows.append([os.path.splitext(link_name)[0], leaf, rel, action, seen.get(stem, "") if action != "link" else ""])
        n_link += 1
        if not args.dry_run:
            os.makedirs(dst_dir, exist_ok=True)
            if not os.path.lexists(dst):
                os.symlink(os.path.abspath(src), dst)

    with open(os.path.join(args.stage, "stage_manifest.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["stem", "session", "src_rel", "action", "first_occurrence"])
        w.writerows(rows)
    with open(os.path.join(args.stage, "leaf_map.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["leaf", "session"])
        for k, v in sorted(leaf_map.items()):
            w.writerow([k, v])
    print(f"source csvs: {len(files)}  linked: {n_link} (of which renamed conflicts: {n_dup_diff})  "
          f"dup identical skipped: {n_dup_same}  sessions: {len(leaf_map)}")
    if not args.dry_run:
        n = sum(len([x for x in os.listdir(os.path.join(args.stage, d)) if x.endswith('.csv')])
                for d in os.listdir(args.stage) if os.path.isdir(os.path.join(args.stage, d)))
        print(f"stage links present: {n}")


if __name__ == "__main__":
    main()
