from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


SEED = "20260926"
RELEASE_COMMIT = "fce8da7"
SEMANTIC_FILES = {"spec.rs", "code.rs", "main.rs"}

PILOT_QUOTAS = {
    ("leetcode", "Easy"): 9,
    ("leetcode", "Medium"): 7,
    ("leetcode", "Hard"): 2,
    ("codeforces", "<=1000"): 3,
    ("codeforces", "1100-1400"): 2,
    ("codeforces", "1500-1800"): 1,
}

CONFIRM_QUOTAS = {
    ("leetcode", "Easy"): 27,
    ("leetcode", "Medium"): 22,
    ("leetcode", "Hard"): 5,
    ("codeforces", "<=1000"): 10,
    ("codeforces", "1100-1400"): 4,
    ("codeforces", "1500-1800"): 3,
    ("codeforces", "Unknown"): 1,
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rank_key(problem: str, family: str, stratum: str) -> str:
    return hashlib.sha256(f"{SEED}|{family}|{stratum}|{problem}".encode()).hexdigest()


def parse_tags(family: str, path: Path) -> tuple[str, list[str], dict]:
    lines = [x.strip() for x in path.read_text(errors="ignore").splitlines() if x.strip()]
    meta: dict = {"raw": lines}
    alg_tags: list[str] = []
    if family == "leetcode":
        stratum = next((x for x in lines if x in {"Easy", "Medium", "Hard"}), "Unknown")
        for line in lines:
            if line in {"Easy", "Medium", "Hard"} or line.lower().startswith("acceptance rate"):
                continue
            alg_tags.extend([t.strip().lower() for t in line.split(",") if t.strip()])
        meta["difficulty"] = stratum
    else:
        rating = None
        for line in lines:
            m = re.search(r"Rating:\s*(\d+)", line)
            if m:
                rating = int(m.group(1))
            elif not line.lower().startswith("solved by"):
                alg_tags.extend([t.strip().lower() for t in line.split(",") if t.strip()])
        if rating is None:
            stratum = "Unknown"
        elif rating <= 1000:
            stratum = "<=1000"
        elif rating <= 1400:
            stratum = "1100-1400"
        elif rating <= 1800:
            stratum = "1500-1800"
        else:
            stratum = ">=1900"
        meta["rating"] = rating
    return stratum, alg_tags, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    repo = args.repo.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    changed = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--name-only", f"{RELEASE_COMMIT}..HEAD", "--", "benchmark/leetcode", "benchmark/codeforces"],
        text=True,
    ).splitlines()
    changed_semantic_dirs = {
        "/".join(parts[:3])
        for x in changed
        if len(parts := x.split("/")) >= 4 and parts[-1] in SEMANTIC_FILES
    }

    eligible = []
    for family in ["leetcode", "codeforces"]:
        root = repo / "benchmark" / family
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            rel_dir = f"benchmark/{family}/{p.name}"
            if rel_dir in changed_semantic_dirs:
                continue
            stratum, alg_tags, tag_meta = parse_tags(family, p / "tags")
            item = {
                "problem": p.name,
                "family": family,
                "stratum": stratum,
                "algorithm_tags": alg_tags,
                "tag_metadata": tag_meta,
                "repo_path": rel_dir,
                "description_sha256": sha(p / "description.md"),
                "spec_sha256": sha(p / "spec.rs"),
                "code_sha256": sha(p / "code.rs"),
                "selection_rank": rank_key(p.name, family, stratum),
            }
            if (p / "main.rs").exists():
                item["main_sha256"] = sha(p / "main.rs")
            eligible.append(item)

    by_stratum: dict[tuple[str, str], list[dict]] = {}
    for item in eligible:
        by_stratum.setdefault((item["family"], item["stratum"]), []).append(item)
    for vals in by_stratum.values():
        vals.sort(key=lambda x: x["selection_rank"])

    pilot = []
    confirm = []
    for key, n in PILOT_QUOTAS.items():
        vals = by_stratum.get(key, [])
        if len(vals) < n + CONFIRM_QUOTAS.get(key, 0):
            raise RuntimeError(f"insufficient eligible problems for {key}: {len(vals)}")
        pilot.extend(vals[:n])
    pilot_ids = {x["problem"] for x in pilot}

    for key, n in CONFIRM_QUOTAS.items():
        vals = [x for x in by_stratum.get(key, []) if x["problem"] not in pilot_ids]
        if len(vals) < n:
            raise RuntimeError(f"insufficient confirm problems for {key}: {len(vals)}")
        confirm.extend(vals[:n])

    pilot.sort(key=lambda x: (x["family"], x["stratum"], x["selection_rank"]))
    confirm.sort(key=lambda x: (x["family"], x["stratum"], x["selection_rank"]))
    out = {
        "purpose": "CDE R4b outcome-blind development/confirmation task split",
        "seed": SEED,
        "vericontest_release_commit": RELEASE_COMMIT,
        "vericontest_head_commit": head,
        "semantic_alignment_rule": "Exclude any main-benchmark problem whose spec.rs, code.rs, or main.rs changed between the original 2026-05-08 benchmark upload and current GitHub HEAD. This conservatively avoids packaged Hugging Face test cases known to be stale for post-release semantic/specification changes.",
        "n_semantically_changed_excluded": len(changed_semantic_dirs),
        "n_eligible": len(eligible),
        "pilot_quotas": {f"{k[0]}::{k[1]}": v for k, v in PILOT_QUOTAS.items()},
        "confirm_quotas": {f"{k[0]}::{k[1]}": v for k, v in CONFIRM_QUOTAS.items()},
        "pilot": pilot,
        "confirm": confirm,
    }
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "eligible": len(eligible),
        "excluded_semantic_changes": len(changed_semantic_dirs),
        "pilot": len(pilot),
        "confirm": len(confirm),
        "head": head,
    }, indent=2))


if __name__ == "__main__":
    main()
