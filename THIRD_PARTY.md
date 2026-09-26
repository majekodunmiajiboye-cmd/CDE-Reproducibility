# Third-party materials and redistribution

This bundle separates original analysis code/results from upstream datasets and software.

- **ADeLe**: code is MIT; the released battery and subject-response data are CC BY-SA 4.0. Raw ADeLe data are not duplicated in this repository. The ADeLe-derived OOF prediction files in `results/adele/` should be treated as derived data and redistributed under the same CC BY-SA 4.0 terms; the upstream dataset license is copied in `third_party/licenses/`.
- **VeriContest**: Apache-2.0. The repository does not bundle the hidden positive test-case files; `analysis/r4b_fetch_tests.py` fetches them from the public Hugging Face dataset using the frozen manifest. Generated responses and extracted implementations are included because they are necessary to reanalyse the reported pilot. Original LeetCode/Codeforces problem statements and combined prompt/response envelopes are not redistributed; retrieve upstream tasks using the pinned source manifest.
- **HarnessBench**: no LICENSE file was found in the inspected checkout. The upstream leaderboard source data are therefore **not redistributed**. The bundle includes our aggregate result, coding decisions, and analysis script, plus the exact upstream commit/hash needed to reconstruct the analysis after obtaining the data from the source.
- **Harness documentation** used for blinded coding is redistributed only from repositories with explicit MIT or Apache-2.0 licenses. Their license texts and exact commits are included.
- **Fan et al. source tables** are not redistributed. The analysis script and aggregate derived result are included; obtain the paper/source separately if reproducing that supporting analysis.

Original CDE analysis code in this repository is licensed under MIT. That license does not override third-party or derived-data terms listed above, and it does not license the manuscript text unless separately stated.
