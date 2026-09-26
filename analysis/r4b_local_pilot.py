from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path


SEED = 20260926
ARMS = ("nl", "spec", "both")

COMMON = """You are solving a Rust programming task.
Return ONLY one `impl Solution { ... }` block containing the requested function implementation.
Do not use Markdown fences. Do not output `struct Solution`, `main`, explanations, proofs, or Verus annotations.
Use stable Rust 2021 with the standard library only. Preserve exactly the target function name, parameter types, and return type shown below. You may add private helper methods inside the same `impl Solution` block.
Your implementation will be inserted into a fixed harness and judged on hidden semantic test cases. This is one-shot pass@1: there will be no compiler feedback or repair round.
"""


def strip_starter(desc: str) -> str:
    return desc.split("## Starter Code", 1)[0].strip()


def function_name_from_spec(spec: str) -> str:
    matches = list(re.finditer(r"pub\s+fn\s+(\w+)\s*\(", spec))
    for m in reversed(matches):
        tail = spec[m.end():m.end() + 800]
        if "requires" in tail or "ensures" in tail:
            return m.group(1)
    if not matches:
        raise ValueError("no pub fn in spec")
    return matches[-1].group(1)


def starter_signature(desc: str, fn_name: str) -> str:
    starter = desc.split("## Starter Code", 1)[1] if "## Starter Code" in desc else desc
    m = re.search(
        rf"pub\s+fn\s+{re.escape(fn_name)}\s*\([^{{]*?\)\s*(?:->\s*[^{{\n]+)?\s*\{{",
        starter,
        re.S,
    )
    if not m:
        raise ValueError(f"starter signature not found for {fn_name}")
    return re.sub(r"\s+", " ", m.group(0)[:-1]).strip()


def build_prompt(problem_dir: Path, arm: str) -> tuple[str, str]:
    desc = (problem_dir / "description.md").read_text(errors="replace")
    spec = (problem_dir / "spec.rs").read_text(errors="replace")
    fn_name = function_name_from_spec(spec)
    sig = starter_signature(desc, fn_name)
    pieces = [COMMON, f"\nTARGET SIGNATURE:\n{sig}\n"]
    if arm in ("nl", "both"):
        pieces.append("\nNATURAL-LANGUAGE TASK:\n" + strip_starter(desc) + "\n")
    if arm in ("spec", "both"):
        pieces.append("\nFORMAL VERUS SPECIFICATION:\n" + spec + "\n")
    pieces.append("\nReturn only the `impl Solution { ... }` block now.\n")
    return "".join(pieces), sig


def ollama_generate(model: str, prompt: str) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "seed": SEED,
            "num_ctx": 32768,
            "num_predict": 4096,
        },
    }
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read())
    out["wall_seconds_client"] = time.time() - t0
    return out


def extract_impl(text: str) -> str | None:
    # Tolerate a model ignoring the no-fence instruction, but never repair code.
    fence = re.search(r"```(?:rust)?\s*(.*?)```", text, re.S | re.I)
    if fence:
        text = fence.group(1)
    m = re.search(r"impl\s+Solution\s*\{", text)
    if not m:
        return None
    brace = text.find("{", m.start())
    depth = 0
    in_str = False
    esc = False
    for i in range(brace, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[m.start():i + 1].strip()
    return None


def camel_to_snake(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def lc_input_line(rec: dict) -> str:
    out = dict(rec)
    if isinstance(rec.get("input"), dict):
        inp = dict(rec["input"])
        for k, v in rec["input"].items():
            inp.setdefault(camel_to_snake(str(k)), v)
        out["input"] = inp
    return json.dumps(out, separators=(",", ":"))


def splice_leetcode(harness: str, impl: str) -> str:
    marker = "use std::io::{self, BufRead, Write, BufWriter};"
    idx = harness.find(marker)
    if idx < 0:
        raise ValueError("harness marker missing")
    pre = harness[:idx]
    m = re.search(r"pub\s+struct\s+Solution\s*;\s*\n", pre)
    if not m:
        raise ValueError("Solution declaration missing")
    return pre[:m.end()] + "\n" + impl + "\n\n" + harness[idx:]


def block_end(src: str, brace: int) -> int:
    depth = 0
    in_str = False
    esc = False
    for i in range(brace, len(src)):
        ch = src[i]
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"': in_str = True
        elif ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: return i + 1
    raise ValueError("unbalanced impl")


def splice_codeforces(main_src: str, impl: str) -> str:
    m = re.search(r"^[ \t]*impl\s+Solution\s*\{", main_src, re.M)
    if not m:
        raise ValueError("impl Solution missing in main.rs")
    brace = main_src.index("{", m.start())
    end = block_end(main_src, brace)
    return main_src[:m.start()] + impl + main_src[end:]


def normalize_cf(s: str) -> str:
    lines = [x.rstrip() for x in s.splitlines()]
    while lines and not lines[-1]: lines.pop()
    return "\n".join(lines)


def evaluate_impl(problem_dir: Path, family: str, impl: str, testcase: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="cde_r4b_") as td:
        td = Path(td)
        src = td / "candidate.rs"; binp = td / "candidate_bin"
        if family == "leetcode":
            harness = (problem_dir / "tests" / "harness.rs").read_text()
            src.write_text(splice_leetcode(harness, impl))
        else:
            src.write_text(splice_codeforces((problem_dir / "main.rs").read_text(), impl))
        c = subprocess.run(
            ["rustc", "--edition", "2021", "-O", "-A", "warnings", str(src), "-o", str(binp)],
            capture_output=True, text=True, timeout=120,
        )
        if c.returncode != 0:
            return {"compile_ok": False, "semantic_pass": False, "error": (c.stderr or c.stdout)[-3000:]}

        n = 0
        if family == "codeforces":
            with testcase.open() as f:
                for line in f:
                    rec = json.loads(line); n += 1
                    try:
                        r = subprocess.run([str(binp)], input=rec["input"], capture_output=True, text=True, timeout=10)
                    except subprocess.TimeoutExpired:
                        return {"compile_ok": True, "semantic_pass": False, "n_checked": n, "error": "runtime timeout"}
                    if r.returncode != 0 or normalize_cf(r.stdout) != normalize_cf(rec["output"]):
                        return {"compile_ok": True, "semantic_pass": False, "n_checked": n,
                                "error": (r.stderr or "wrong output")[-1000:]}
        else:
            chunk = []
            expected = []
            def check() -> tuple[bool, str]:
                if not chunk: return True, ""
                try:
                    r = subprocess.run([str(binp)], input="\n".join(chunk) + "\n", capture_output=True, text=True, timeout=60)
                except subprocess.TimeoutExpired:
                    return False, "runtime timeout"
                if r.returncode != 0: return False, (r.stderr or "runtime failure")[-1000:]
                outs = [json.loads(x)["output"] for x in r.stdout.splitlines() if x.strip()]
                return (outs == expected, "wrong output" if outs != expected else "")
            with testcase.open() as f:
                for line in f:
                    rec = json.loads(line); n += 1
                    chunk.append(lc_input_line(rec)); expected.append(rec["output"])
                    if len(chunk) == 200:
                        ok, err = check()
                        if not ok: return {"compile_ok": True, "semantic_pass": False, "n_checked": n, "error": err}
                        chunk.clear(); expected.clear()
                ok, err = check()
                if not ok: return {"compile_ok": True, "semantic_pass": False, "n_checked": n, "error": err}
        return {"compile_ok": True, "semantic_pass": True, "n_checked": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--test-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=0, help="deterministic prefix of frozen pilot; 0=all 24")
    ap.add_argument("--problems", nargs="*", default=None,
                    help="optional pre-frozen subset of pilot problem IDs; order follows the frozen pilot manifest")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sel = json.loads(args.selection.read_text())
    tasks = list(sel["pilot"])
    if args.problems:
        wanted = set(args.problems)
        available = {x["problem"] for x in tasks}
        missing = sorted(wanted - available)
        if missing:
            raise ValueError(f"requested problems are not in frozen pilot: {missing}")
        tasks = [x for x in tasks if x["problem"] in wanted]
    if args.limit:
        tasks = tasks[:args.limit]
    results = []
    for model in args.models:
        for ti, item in enumerate(tasks, 1):
            pdir = args.repo / item["repo_path"]
            testcase = args.test_root / item["family"] / item["problem"] / "testcases.jsonl"
            for arm in ARMS:
                prompt, sig = build_prompt(pdir, arm)
                raw = ollama_generate(model, prompt)
                text = raw.get("response", "")
                impl = extract_impl(text)
                ev = {"compile_ok": False, "semantic_pass": False, "error": "no impl Solution block"} if impl is None else evaluate_impl(pdir, item["family"], impl, testcase)
                rec = {
                    "model": model, "task_index": ti, "family": item["family"], "problem": item["problem"],
                    "arm": arm, "target_signature": sig, "prompt_chars": len(prompt),
                    "response_chars": len(text), "impl_extracted": impl is not None,
                    "eval": ev,
                    "ollama": {k: raw.get(k) for k in ["done_reason", "total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration", "wall_seconds_client"]},
                }
                raw_path = args.out_dir / "raw" / re.sub(r"[^A-Za-z0-9_.-]+", "_", model) / item["problem"]
                raw_path.mkdir(parents=True, exist_ok=True)
                (raw_path / f"{arm}.json").write_text(json.dumps({"prompt": prompt, "response": text, "impl": impl, "ollama": raw}, indent=2) + "\n")
                results.append(rec)
                print(f"{model} {ti:02d}/{len(tasks)} {item['problem']} {arm}: compile={ev.get('compile_ok')} pass={ev.get('semantic_pass')} wall={raw.get('wall_seconds_client',0):.1f}s", flush=True)
        # Write a checkpoint after every model.
        (args.out_dir / "pilot_results.json").write_text(json.dumps({"results": results}, indent=2, sort_keys=True) + "\n")
    summary = {}
    for model in args.models:
        summary[model] = {}
        for arm in ARMS:
            r = [x for x in results if x["model"] == model and x["arm"] == arm]
            summary[model][arm] = {
                "n": len(r),
                "compile_rate": sum(x["eval"].get("compile_ok", False) for x in r) / max(1, len(r)),
                "semantic_success_rate": sum(x["eval"].get("semantic_pass", False) for x in r) / max(1, len(r)),
            }
    out = {"seed": SEED, "arms": ARMS, "models": args.models, "n_tasks": len(tasks), "summary": summary, "results": results}
    (args.out_dir / "pilot_result.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
