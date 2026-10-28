# Delta Code PS1 - Maximum Bipartite Matching via Augmenting Paths

Capacitated candidate-to-role allocation (2,000 candidates, 320 roles) with a **hand-written Hopcroft-Karp engine** plus several features that go beyond a standard student solution.

## What makes this submission different (Uniqueness)

Most teams implement classical Hopcroft-Karp + show the greedy gap + verify with ILP.  
We deliberately added three deeper layers that are rare in IIT-level submissions:

1. **Bottleneck / Criticality Analysis**  
   After computing the maximum matching we identify:
   - *Tight roles* – filled to capacity
   - *Critical roles* – roles whose capacity cannot be usefully increased (zero shadow price)
   - *Sensitivity vector* – for every role, whether `capacity + 1` would raise the optimum  
   This answers the practical workforce question: “Which roles are the true bottlenecks?”

2. **Explicit Dual LP Certificate**  
   In addition to the combinatorial König cover we construct a feasible dual solution  
   `(y_u, z_j)` of the matching LP whose objective equals `|M|`.  
   This is a second, independent optimality proof that sits between pure combinatorics and full LP duality.

3. **Merit + Fairness post-processing**  
   Among all maximum-cardinality matchings we select the one with maximum total skill-fit  
   (min-cost max-flow, ties broken by cryptographic hash of the candidate name so that  
   file order never influences the outcome). A secondary local-search pass can further  
   reduce variance of fit among the hired set.

These three extras turn a correct algorithmic solution into a *workforce-insight* tool.

## Files

| File | What it is |
|---|---|
| `allocate.py` | Engine (greedy + Hopcroft-Karp, König certificate, **criticality analysis**, **dual certificate**), synthetic data generators, ILP benchmark, plots |
| `unique_enhancements.py` | Stand-alone module containing the capacity-aware matching sketch and analysis helpers |
| `assignment.csv` | Final assignment (maximum hires, max skill fit) with per-candidate fit and plain-language explanation |
| `path_traces.txt` | Every augmenting path found on the main instance |
| `role_summary.csv` | Per-role capacity and fill, greedy vs final |
| `matching_vis.png` | Bipartite graph, augmented edges in red |
| `benchmark.png` | Runtime vs ILP, greedy gap, HK scaling |
| `presentation.pdf` / `report.md` | Proofs (Berge, König), complexity, traces, benchmarks, **criticality & dual sections** |
| `results.json` | Raw numbers behind the report (includes uniqueness metrics) |
| `match_csv.py` | Run the engine on your own files: `python match_csv.py candidates.csv roles.csv` |
| `merit.py` | Hand-written min-cost max-flow for best skill fit |
| `sample_*.csv` | Example input format |
| `test_core.py` | pytest suite |

## Run

```bash
pip install -r requirements.txt
python allocate.py
python make_report.py
python -m pytest test_core.py
python match_csv.py sample_candidates.csv sample_roles.csv
```

The matching engine uses only the standard library. numpy/scipy are used only for the ILP benchmark; networkx only for independent max-flow validation; figures are drawn with matplotlib.

Data is synthetic (fixed seeds → fully reproducible).

## Key results (seed=1)

- 2000 candidates, 320 roles, total capacity 1696
- Greedy: 1536 hires → **Hopcroft-Karp: 1696 hires** (gap 160)
- König cover size = matching size (Berge + König optimality)
- Dual LP objective also equals 1696
- Criticality analysis reports how many roles are structural bottlenecks vs. merely full
