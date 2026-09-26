# Reproduction guide

## 1. Environment

The original run used Python 3.9.6 on Apple Silicon/macOS. Use Python 3.11 to install the historical pinned scientific packages (not Python 3.13):

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r environment/requirements.txt
```

R4b candidate generation additionally requires Ollama 0.34.0 and the exact model builds listed in `environment/models.json`; Rust 1.98.1 was used for semantic judging.

## 2. Verify the archived outputs first

```bash
python scripts/verify_bundle.py
python scripts/verify_reported_values.py
```

This is the fastest route for reviewers: it verifies file integrity and recomputes the manuscript-only secondary comparisons from OOF/pilot outputs.

## 3. Fetch public upstream inputs

```bash
python scripts/fetch_public_inputs.py
```

This pins ADeLe, VeriContest and HarnessBench to the commits recorded in `manifests/source_manifest.json` and downloads the VeriContest HF manifest. Raw third-party datasets are not committed here.

## 4. ADeLe primary and robustness analyses

After fetching ADeLe, set:

```bash
ADELE=inputs/upstream/adele/ADeLe_battery_data
SUBJECTS=$ADELE/subject_specific_instance_level_data
BATTERY=$ADELE/ADeLe_batterry_v1dot0.csv
mkdir -p reproduced
```

Primary R1:
```bash
python analysis/r1_adele.py --data-dir "$SUBJECTS" --battery-file "$BATTERY" --out-dir reproduced/r1
```

Strong response-profile comparator:
```bash
python analysis/r1_strong_comparator.py --data-dir "$SUBJECTS" --battery-file "$BATTERY" --out-dir reproduced/r1_strong --n-jobs 8
```

Nonlinear response-only robustness:
```bash
python analysis/r1_nonlinear_robustness.py --data-dir "$SUBJECTS" --battery-file "$BATTERY" --out-dir reproduced/r1_nonlinear --n-jobs 8
```

Generic-text control requires Ollama with `nomic-embed-text:latest` matching `environment/models.json`:
```bash
ollama pull nomic-embed-text
python analysis/r1_text_control.py --data-dir "$SUBJECTS" --battery-file "$BATTERY" \
  --embedding-cache reproduced/nomic_question_embeddings.npz --out-dir reproduced/r1_text
```

Diagnostics:
```bash
python analysis/r2_adele_split_half.py --battery-file "$BATTERY" --response-dir "$SUBJECTS" --out reproduced/r2.json
python analysis/r3_adele.py --data-dir "$SUBJECTS" --battery-file "$BATTERY" --out reproduced/r3.json
```

## 5. Controlled R4b pilot

The frozen split and task features are already in `manifests/`. Fetch the public hidden positive tests using the upstream HF manifest:

```bash
python analysis/r4b_fetch_tests.py \
  --selection manifests/r4b_task_manifest.json \
  --hf-manifest inputs/upstream/vericontest_hf_manifest.jsonl \
  --out-root inputs/vericontest_tests \
  --out-manifest reproduced/r4b_testcase_manifest.json
```

The raw model generations from the original run are bundled under `results/r4b/raw/`, so the pilot analysis can be reproduced without rerunning Ollama:

```bash
python analysis/r4b_analyze_pilot.py \
  --pilot-result results/r4b/full_pilot/pilot_result.json \
  --task-features manifests/r4b_task_features.json \
  --model-a devstral-small-2:24b-instruct-2512-q4_K_M \
  --model-b gemma3:12b \
  --out reproduced/r4b_pilot_analysis.json
```

To rerun candidate generation itself, use `analysis/r4b_local_pilot.py` with the exact model builds/settings in `environment/models.json` and `environment/system.json`.

## 6. HarnessBench and Fan supporting analyses

HarnessBench's inspected checkout contained no LICENSE file, so the source leaderboard data are not redistributed. The exact input hash and upstream commit are in `manifests/source_manifest.json`; the blinded coder materials and feature consensus are included. Once the upstream score file has been obtained:

```bash
python analysis/r4_harnessbench.py \
  --scores /path/to/leaderboard_scores.json \
  --consensus results/environment/harnessbench/feature_consensus.json \
  --docs-dir third_party/harness_docs \
  --out-dir reproduced/r4_harnessbench
```

The Fan source tables are likewise not redistributed. With the paper source directory available:

```bash
python analysis/r4_fan.py --source-dir /path/to/fan-paper-source --out reproduced/r4_fan.json
```

## 7. Expected frozen identifiers

- Main protocol hash: `1d6423c6da1fd42dd03ae7024fac4aedd652b569b887079e416a5b56f9e1c4ee`
- Design hash: `588b0c3897b8a16fcf1376422883f7376b14c8c6ef3273afb9805226a4270d4b`
- Seed: `20260926`

See `manifests/source_manifest.json` for public upstream provenance. The internal project-management ledger is retained privately; this archive does not contain the complete internal provenance chain.

The generated response text and extracted implementations are retained unchanged. Original benchmark prompts and duplicated model-response envelopes are not redistributed; the original combined records are retained privately. Each response-only record includes the original record digest.
