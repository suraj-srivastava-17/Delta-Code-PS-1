# Delta Code PS1 - Maximum Bipartite Matching via Augmenting Paths

Capacitated candidate-to-role allocation with a hand-written Hopcroft-Karp engine.

| File | What it is |
|---|---|
| `allocate.py` | Engine (greedy + Hopcroft-Karp, König certificate), synthetic data generators, ILP benchmark, plots |
| `assignment.csv` | Final candidate-to-role assignment (maximum hires), with the greedy role for comparison |
| `role_summary.csv` | Per-role capacity and fill, greedy vs final |
| `matching_vis.png` | Bipartite graph, augmented edges in red |
| `benchmark.png` | Runtime vs ILP, greedy gap, HK scaling |
| `presentation.pdf` / `report.md` | Proofs (Berge, König), complexity, traces, benchmarks |
| `results.json` | Raw numbers behind the report |
| `make_report.py` | Rebuilds `report.md` and `presentation.pdf` from `results.json` |

## Run
```
pip install -r requirements.txt
python allocate.py
python make_report.py
```
The matching engine uses only the standard library. numpy/scipy are used only for the ILP benchmark; networkx only for drawing and for independent max-flow validation.
Data is synthetic (no dataset was provided); seeds are fixed, so results are reproducible.
