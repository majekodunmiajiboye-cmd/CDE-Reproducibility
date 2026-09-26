from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BASE = "https://huggingface.co/datasets/Gax-c/VeriContest/resolve/main/"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_one(item: dict, out_root: Path) -> dict:
    # manifest.jsonl records the uploader's staging path (`hf_benchmark_stage/...`),
    # while the published HF repository exposes `codeforces/...` and
    # `leetcode/...` at the dataset root.
    rel = item["test_path"].removeprefix("hf_benchmark_stage/")
    dest = out_root / item["family"] / item["problem"] / "testcases.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size != item["bytes"]:
        tmp = dest.with_suffix(".jsonl.part")
        req = urllib.request.Request(BASE + rel, headers={"User-Agent": "ResearchOS-CDE-R4b/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
            while True:
                b = r.read(1024 * 1024)
                if not b:
                    break
                f.write(b)
        if tmp.stat().st_size != item["bytes"]:
            raise RuntimeError(f"size mismatch {item['problem']}: {tmp.stat().st_size} != {item['bytes']}")
        tmp.replace(dest)
    return {
        "family": item["family"],
        "problem": item["problem"],
        "split": item["split"],
        "bytes": dest.stat().st_size,
        "sha256": sha(dest),
        "path": str(dest),
        "hf_path": rel,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--hf-manifest", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--out-manifest", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    sel = json.loads(args.selection.read_text())
    wanted = {}
    for split in ["pilot", "confirm"]:
        for x in sel[split]:
            wanted[x["problem"]] = {"family": x["family"], "split": split}

    available = {}
    for line in args.hf_manifest.read_text().splitlines():
        r = json.loads(line)
        if r["problem"] not in wanted:
            continue
        positive = next(
            x for x in r["testcase_files"]
            if x["path"].endswith("/testcases.jsonl") and not x["path"].endswith("/mutated_testcases.jsonl")
        )
        available[r["problem"]] = {
            **wanted[r["problem"]],
            "problem": r["problem"],
            "bytes": int(positive["bytes"]),
            "test_path": positive["path"],
        }
    missing = sorted(set(wanted) - set(available))
    if missing:
        raise RuntimeError(f"selected problems missing from HF manifest: {missing}")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch_one, item, args.out_root) for item in available.values()]
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 10 == 0 or i == len(futs):
                print(f"downloaded/verified {i}/{len(futs)}", flush=True)

    results.sort(key=lambda x: (x["split"], x["family"], x["problem"]))
    out = {
        "source": "Gax-c/VeriContest Hugging Face positive testcases",
        "selection_manifest": str(args.selection),
        "n_files": len(results),
        "total_bytes": sum(x["bytes"] for x in results),
        "files": results,
    }
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"n_files": len(results), "total_bytes": out["total_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
