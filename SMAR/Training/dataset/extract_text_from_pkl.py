#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_conf_idd_only.py

For each .pkl in --in-dir, write {conf}.json to --out-dir
containing a list of {"conf": <conf>, "idd": <idd>} objects.
"""

import argparse, json, os, pickle
from glob import glob
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Dir with *.pkl (e.g., v1.0)")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pkl_paths = sorted(glob(os.path.join(args.in_dir, "*.pkl")))
    if not pkl_paths:
        raise SystemExit(f"No .pkl files found under {args.in_dir}")

    for pkl_path in pkl_paths:
        conf = os.path.splitext(os.path.basename(pkl_path))[0]
        out_path = os.path.join(args.out_dir, f"{conf}.json")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        out = []
        for item in tqdm(data, desc=f"Processing {conf}.pkl", ascii=True):
            out.append({"conf": conf, "idd": item.get("idd")})

        with open(out_path, "w", encoding="utf-8") as fout:
            json.dump(out, fout, ensure_ascii=False, indent=2)

        print(f"Wrote {len(out)} rows → {out_path}")

if __name__ == "__main__":
    main()
