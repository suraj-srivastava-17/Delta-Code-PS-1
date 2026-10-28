# Delta Code PS1 - Maximum Bipartite Matching via Augmenting Paths

Capacitated candidate-to-role allocation (2,000 candidates, 320 roles) with a hand-written Hopcroft-Karp engine.

| File | What it is |
|---|---|
| `allocate.py` | Engine (greedy + Hopcroft-Karp, König certificate), synthetic data generators, ILP benchmark, plots |
| `assignment.csv` | Final assignment (maximum hires, max skill fit) with per-candidate `fit` and plain-language `explanation` (matched/missing skills, or why unseated), plus greedy and plain-HK roles |
| `path_traces.txt` | Every augmenting path found on the main instance |
| `role_summary.csv` | Per-role capacity and fill, greedy vs final |
| `matching_vis.png` | Bipartite graph, augmented edges in red |
| `benchmark.png` | Runtime vs ILP, greedy gap, HK scaling |
| `presentation.pdf` / `report.md` | Proofs (Berge, König), complexity, traces, benchmarks |
| `results.json` | Raw numbers behind the report |
| `match_csv.py` | Run the engine on your own files: `python match_csv.py candidates.csv roles.csv` |
| `merit.py` | Hand-written min-cost max-flow: among all maximum matchings picks the best total skill fit; ties broken by hashed name (not file order) |
| `sample_candidates.csv`, `sample_roles.csv` | Example input format |
| `tests/` | pytest suite (greedy loss, ILP/max-flow/König agreement, edge cases, merit fairness, CLI) |
| `make_report.py` | Rebuilds `report.md` and `presentation.pdf` from `results.json` |

## Run
```
pip install -r requirements.txt
python allocate.py
python make_report.py
python -m pytest tests
python match_csv.py sample_candidates.csv sample_roles.csv
```
The matching engine uses only the standard library. numpy/scipy are used only for the ILP benchmark; networkx only for independent max-flow validation; the figures are drawn with matplotlib.
Data is synthetic (no dataset was provided); seeds are fixed, so results are reproducible.

`roles.csv` may include an optional `mandatory_skills` column (hard requirement: candidate is ineligible without all of them). A candidate with an empty skills field is reported as MISSING INFORMATION, not as a skill gap. The merit step takes ~9 s on the 2000x320 instance.
