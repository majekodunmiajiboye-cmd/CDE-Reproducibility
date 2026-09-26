# CDE reproducibility bundle

This repository packages the analysis code, frozen task and protocol manifests, result artifacts, model generations, and provenance needed to audit the paper:

**Task content survives strong controls; taxonomy-specific and structured environment-transform claims do not** (26 September 2026).

## What is included

- `analysis/` — scientific analysis scripts used for the reported results.
- `results/adele/` — governed R1 result, strong-comparator analysis, nonlinear robustness, generic-text control, OOF predictions, and R2/R3 diagnostics.
- `results/environment/` — Fan reanalysis result and HarnessBench result/coding artifacts.
- `results/r4b/` — controlled representation pilot outputs, analysis, competence screen and response-only local-model generations.
- `manifests/` — frozen task splits, test-case hashes, task features, protocols and upstream source hashes.
- `environment/` — exact Python/Rust/Ollama/package/model versions.
- `third_party/` — licensed harness-document materials, copied licenses and redistribution notes.
- `scripts/verify_bundle.py` — verifies this package byte-for-byte.
- `scripts/verify_reported_values.py` — re-derives headline/supplementary numbers from the bundled outputs.

## Quick audit

Use Python 3.11 with the historical package pins below; they are not compatible with Python 3.13.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r environment/requirements.txt
python scripts/verify_bundle.py
python scripts/verify_reported_values.py
```

The two commands above do **not** require re-downloading the ~GB-scale upstream datasets or rerunning local language models.

## Full reproduction

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). Raw upstream ADeLe and VeriContest/HarnessBench inputs are intentionally not duplicated where licensing, size or provenance makes direct redistribution undesirable. Exact source URLs, commits and hashes are in `manifests/source_manifest.json`.

## Protocol status

The main protocol and later R4b development gate were prospectively frozen in the local project ledger before the relevant outcomes were inspected. This was **not** an external public preregistration. The internal project ledger is retained privately; public task manifests and reported results are included.

## Licensing

Original CDE analysis code is released under the [MIT License](LICENSE), copyright 2026 Majekodunmi Ajiboye. See [`THIRD_PARTY.md`](THIRD_PARTY.md) for upstream materials and derived-data terms. ADeLe-derived data remain subject to CC BY-SA 4.0.

Archive DOI: https://doi.org/10.5281/zenodo.22984125
Latest study version: https://doi.org/10.5281/zenodo.22984124

The generated response text and extracted implementations are retained unchanged. Original benchmark prompts and duplicated model-response envelopes are not redistributed; the original combined records are retained privately. Each response-only record includes the original record digest.

The historical HarnessBench coding artifacts were produced by two AI-assisted coders, as identified in the retained coder records. They are not human-rater assessments.
