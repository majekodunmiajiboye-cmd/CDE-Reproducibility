from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def normalize_cf(s: str) -> str:
    lines = [ln.rstrip() for ln in s.splitlines()]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def camel_to_snake(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def add_json_key_aliases(rec: dict) -> dict:
    """Add snake_case aliases for LeetCode JSON input field names.

    The released testcase JSON sometimes preserves the LeetCode camelCase
    starter-code names while the current Rust benchmark signature uses
    snake_case.  Adding an alias is an interface-only normalization: the input
    value and expected semantic output are unchanged.
    """
    out = dict(rec)
    inp = rec.get("input")
    if isinstance(inp, dict):
        norm = dict(inp)
        for key, value in inp.items():
            norm.setdefault(camel_to_snake(str(key)), value)
        out["input"] = norm
    return out


def ensure_test_link(repo: Path, data_root: Path, family: str, problem: str) -> Path:
    pdir = repo / "benchmark" / family / problem
    tests = pdir / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    src = (data_root / family / problem / "testcases.jsonl").resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    dst = tests / "testcases.jsonl"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)
    return pdir


def validate_leetcode(repo: Path, pdir: Path, testfile: Path, chunk_lines: int = 200) -> dict:
    build = repo / "test_gen" / "build_harness.py"
    proc = subprocess.run(
        ["python3", str(build), "--problem", str(pdir), "--compile"],
        cwd=str(repo), capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0:
        return {"ok": False, "stage": "compile", "error": (proc.stderr or proc.stdout)[-4000:]}
    binp = pdir / "tests" / "harness_bin"
    if not binp.is_file():
        return {"ok": False, "stage": "compile", "error": "harness_bin missing after successful build"}

    n = 0
    chunk_raw: list[str] = []
    chunk_expected: list[object] = []

    def check_chunk() -> tuple[bool, str]:
        if not chunk_raw:
            return True, ""
        r = subprocess.run(
            [str(binp)], input="\n".join(chunk_raw) + "\n",
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            return False, (r.stderr or r.stdout)[-4000:]
        got_lines = [x for x in r.stdout.splitlines() if x.strip()]
        if len(got_lines) != len(chunk_expected):
            return False, f"output line count {len(got_lines)} != expected {len(chunk_expected)}"
        for i, (line, expected) in enumerate(zip(got_lines, chunk_expected)):
            try:
                got = json.loads(line)["output"]
            except Exception as e:
                return False, f"bad harness JSON at chunk row {i}: {e}: {line[:300]}"
            if got != expected:
                return False, f"semantic mismatch at chunk row {i}: got={got!r}, expected={expected!r}"
        return True, ""

    with testfile.open() as f:
        for line in f:
            rec = json.loads(line)
            rec_for_harness = add_json_key_aliases(rec)
            chunk_raw.append(json.dumps(rec_for_harness, separators=(",", ":")))
            chunk_expected.append(rec["output"])
            n += 1
            if len(chunk_raw) >= chunk_lines:
                ok, err = check_chunk()
                if not ok:
                    return {"ok": False, "stage": "reference_check", "n_cases_checked": n, "error": err}
                chunk_raw.clear(); chunk_expected.clear()
        ok, err = check_chunk()
        if not ok:
            return {"ok": False, "stage": "reference_check", "n_cases_checked": n, "error": err}
    return {"ok": True, "stage": "complete", "n_cases_checked": n}


def validate_codeforces(pdir: Path, testfile: Path) -> dict:
    outbin = pdir / "tests" / "cf_reference_bin"
    proc = subprocess.run(
        ["rustc", "--edition", "2021", "-O", str(pdir / "main.rs"), "-o", str(outbin)],
        cwd=str(pdir), capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0:
        return {"ok": False, "stage": "compile", "error": (proc.stderr or proc.stdout)[-4000:]}
    n = 0
    with testfile.open() as f:
        for line in f:
            rec = json.loads(line)
            r = subprocess.run(
                [str(outbin)], input=rec["input"], capture_output=True, text=True, timeout=30,
            )
            n += 1
            if r.returncode != 0:
                return {"ok": False, "stage": "reference_check", "n_cases_checked": n,
                        "error": (r.stderr or r.stdout)[-4000:]}
            if normalize_cf(r.stdout) != normalize_cf(rec["output"]):
                return {"ok": False, "stage": "reference_check", "n_cases_checked": n,
                        "error": f"semantic mismatch: got={r.stdout[:500]!r}, expected={rec['output'][:500]!r}"}
    return {"ok": True, "stage": "complete", "n_cases_checked": n}


def validate_one(item: dict, repo: Path, data_root: Path) -> dict:
    family = item["family"]; problem = item["problem"]
    try:
        pdir = ensure_test_link(repo, data_root, family, problem)
        testfile = (data_root / family / problem / "testcases.jsonl").resolve()
        if family == "leetcode":
            result = validate_leetcode(repo, pdir, testfile)
        else:
            result = validate_codeforces(pdir, testfile)
    except Exception as e:
        result = {"ok": False, "stage": "exception", "error": repr(e)}
    return {"family": family, "problem": problem, "split": item["split"], **result}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()
    sel = json.loads(args.selection.read_text())
    items = []
    for split in ["pilot", "confirm"]:
        for x in sel[split]:
            items.append({"family": x["family"], "problem": x["problem"], "split": split})

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(validate_one, item, args.repo.resolve(), args.data_root.resolve()) for item in items]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result(); results.append(r)
            print(f"{i:3d}/{len(futs)} {r['split']:7s} {r['family']:10s} {r['problem']:10s} "
                  f"{'PASS' if r['ok'] else 'FAIL'} {r.get('n_cases_checked','')}", flush=True)

    results.sort(key=lambda x: (x["split"], x["family"], x["problem"]))
    summary = {
        "n": len(results),
        "n_pass": sum(bool(x["ok"]) for x in results),
        "n_fail": sum(not bool(x["ok"]) for x in results),
        "pilot_pass": sum(bool(x["ok"]) and x["split"] == "pilot" for x in results),
        "confirm_pass": sum(bool(x["ok"]) and x["split"] == "confirm" for x in results),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ["n", "n_pass", "n_fail", "pilot_pass", "confirm_pass"]}, indent=2))


if __name__ == "__main__":
    main()
