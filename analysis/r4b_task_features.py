from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def strip_starter(text: str) -> str:
    return text.split("## Starter Code", 1)[0]


def token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def target_signature(spec: str) -> str:
    # Prefer an executable pub fn whose nearby suffix contains requires/ensures.
    matches = list(re.finditer(r"pub\s+fn\s+\w+\s*\([^)]*\)\s*(?:->\s*[^\n{]+)?", spec, re.S))
    for m in reversed(matches):
        tail = spec[m.end():m.end() + 700]
        if "requires" in tail or "ensures" in tail:
            return re.sub(r"\s+", " ", m.group(0)).strip()
    return re.sub(r"\s+", " ", matches[-1].group(0)).strip() if matches else ""


def interface_features(sig: str) -> tuple[int, int]:
    if not sig or "(" not in sig:
        return 0, 0
    args = sig[sig.find("(") + 1:sig.rfind(")")]
    depth = 0
    parts = []
    cur = []
    for ch in args:
        if ch in "<([": depth += 1
        elif ch in ">)]": depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip(): parts.append("".join(cur))
    n_params = len([p for p in parts if ":" in p])
    type_nesting = sig.count("Vec<") + sig.count("Seq<") + sig.count("Option<") + sig.count("(") - 1
    return n_params, max(0, type_nesting)


def difficulty_value(item: dict) -> float:
    if item["family"] == "leetcode":
        return {"Easy": 0.0, "Medium": 0.5, "Hard": 1.0}.get(item["stratum"], 0.5)
    rating = item.get("tag_metadata", {}).get("rating")
    if rating is None:
        return 0.5
    # Fixed public Codeforces scale; clipping avoids one extreme task dominating.
    return float(min(1.0, max(0.0, (float(rating) - 800.0) / 1600.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sel = json.loads(args.selection.read_text())
    rows = []
    for split in ["pilot", "confirm"]:
        for item in sel[split]:
            p = args.repo / item["repo_path"]
            desc = (p / "description.md").read_text(errors="replace")
            spec = (p / "spec.rs").read_text(errors="replace")
            nl = strip_starter(desc)
            sig = target_signature(spec)
            n_params, type_nesting = interface_features(sig)
            logic_terms = len(re.findall(r"\b(?:forall|exists|requires|ensures|recommends)\b", spec))
            helper_spec_fns = len(re.findall(r"\b(?:open\s+)?spec\s+fn\b", spec))
            spec_tokens = token_count(spec)
            rows.append({
                "split": split,
                "family": item["family"],
                "problem": item["problem"],
                "target_signature": sig,
                # Frozen R4b task-demand proxies. Analysis standardizes them.
                "d_algorithmic_difficulty": difficulty_value(item),
                "d_nl_log_tokens": math.log1p(token_count(nl)),
                "d_spec_log_tokens": math.log1p(spec_tokens),
                "d_spec_logic_density": float(logic_terms + helper_spec_fns) / max(1, spec_tokens),
                "d_interface_complexity": float(n_params + type_nesting),
                "raw": {
                    "nl_tokens": token_count(nl),
                    "spec_tokens": spec_tokens,
                    "logic_terms": logic_terms,
                    "helper_spec_fns": helper_spec_fns,
                    "n_params": n_params,
                    "type_nesting": type_nesting,
                },
            })
    out = {
        "purpose": "Outcome-blind R4b task-demand proxy features frozen before local pilot outcomes",
        "features": [
            "d_algorithmic_difficulty",
            "d_nl_log_tokens",
            "d_spec_log_tokens",
            "d_spec_logic_density",
            "d_interface_complexity",
        ],
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"n": len(rows), "features": out["features"]}, indent=2))


if __name__ == "__main__":
    main()
