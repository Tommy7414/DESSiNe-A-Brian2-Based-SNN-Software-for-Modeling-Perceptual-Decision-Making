#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import pickle
from typing import Any, Dict, Optional

STRENGTH_RE = re.compile(r"_strength([0-9]+(?:\.[0-9]+)?)_", re.IGNORECASE)

def infer_strength_from_filename(path: str) -> Optional[float]:
    m = STRENGTH_RE.search(os.path.basename(path))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None

def load_pkl(fp: str) -> Any:
    with open(fp, "rb") as f:
        return pickle.load(f)

def save_pkl(fp: str, obj: Any) -> None:
    with open(fp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def get_metadata(obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    md = obj.get("metadata", None)
    return md if isinstance(md, dict) else None

def choose_ev(meta: Dict[str, Any], fp: str) -> Optional[float]:
    # evidence_strength > coherence > strength > filename
    for k in ("evidence_strength", "coherence", "strength"):
        if k in meta:
            try:
                return float(meta[k])
            except Exception:
                pass
    s = infer_strength_from_filename(fp)
    return float(s) if s is not None else None

def fix_one(fp: str, dry: bool, backup: bool, keep_strength: bool, verbose: bool=True) -> bool:
    try:
        obj = load_pkl(fp)
    except Exception as e:
        if verbose:
            print(f"[SKIP] read fail: {fp}\n       {e}")
        return False

    meta = get_metadata(obj)
    if meta is None:
        if verbose:
            print(f"[SKIP] no dict metadata: {fp}")
        return False

    ev = choose_ev(meta, fp)
    if ev is None:
        if verbose:
            print(f"[SKIP] cannot infer evidence strength: {fp}")
        return False

    changed = False

    # 強制一致：coherence 與 evidence_strength 一定同值
    if float(meta.get("evidence_strength", float("nan"))) != ev:
        meta["evidence_strength"] = float(ev)
        changed = True

    if float(meta.get("coherence", float("nan"))) != ev:
        meta["coherence"] = float(ev)
        changed = True

    # strength 欄位：你要保留就同步，不保留就刪掉
    if keep_strength:
        if float(meta.get("strength", float("nan"))) != ev:
            meta["strength"] = float(ev)
            changed = True
    else:
        if "strength" in meta:
            meta.pop("strength", None)
            changed = True

    if not changed:
        if verbose:
            print(f"[OK ] {os.path.basename(fp)}  ev={ev} (already consistent)")
        return True

    if dry:
        if verbose:
            print(f"[DRY] {os.path.basename(fp)}  -> set coherence=evidence_strength={ev}")
        return True

    if backup:
        bak = fp + ".bak"
        if not os.path.exists(bak):
            try:
                save_pkl(bak, obj)
            except Exception as e:
                if verbose:
                    print(f"[FAIL] backup fail: {fp}\n       {e}")
                return False

    try:
        save_pkl(fp, obj)
    except Exception as e:
        if verbose:
            print(f"[FAIL] write fail: {fp}\n       {e}")
        return False

    if verbose:
        print(f"[FIX] {os.path.basename(fp)}  -> coherence=evidence_strength={ev} (backup={'yes' if backup else 'no'})")
    return True

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="data 資料夾路徑")
    ap.add_argument("--glob", default="*.pkl", help="預設掃描 *.pkl")
    ap.add_argument("--dry", action="store_true", help="只顯示，不寫回")
    ap.add_argument("--no_backup", action="store_true", help="不產生 .bak")
    ap.add_argument("--drop_strength", action="store_true", help="把 strength 欄位移除（預設保留）")
    args = ap.parse_args()

    pattern = os.path.join(args.data_dir, args.glob)
    files = sorted(glob.glob(pattern))
    print(f"[INFO] Found {len(files)} files: {pattern}")

    ok = 0
    for fp in files:
        if fix_one(
            fp,
            dry=args.dry,
            backup=(not args.no_backup),
            keep_strength=(not args.drop_strength),
            verbose=True
        ):
            ok += 1

    print(f"[DONE] success {ok}/{len(files)}")

if __name__ == "__main__":
    main()
