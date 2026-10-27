"""
allocate.py  -  Delta Code PS1: Maximum Bipartite Matching via Augmenting Paths
================================================================================
Capacitated candidate -> role allocation.

* The matching engine (greedy + Hopcroft-Karp) is written by hand with only the
  standard library.  numpy/scipy are used ONLY for the ILP benchmark and
  networkx ONLY for drawing and for validating the answer with an independent
  max-flow solver.

Model
-----
Candidates u in L, roles j in R with capacity c_j.  Flow network:
    source -> u   (capacity 1)
    u -> j        (capacity 1)   if u is eligible for j
    j -> sink     (capacity c_j)
Expanding role j into c_j identical "slots" turns this into an ordinary
bipartite graph with unit capacities, where max-flow == max matching, so plain
Hopcroft-Karp (O(E sqrt(V))) applies.

Run:  python allocate.py          (writes all outputs next to this file)
"""
import csv
import json
import os
import random
import time
from collections import deque

INF = float("inf")
HERE = os.path.dirname(os.path.abspath(__file__))

SKILLS = ["python", "sql", "ml", "stats", "cloud", "security", "java", "web",
          "devops", "nlp", "vision", "math"]


# --------------------------------------------------------------------------
# 1. Instance generation (synthetic, reproducible)
# --------------------------------------------------------------------------
def make_instance(n_cand, n_roles, seed, tightness=0.85, min_overlap=2):
    """Candidates have skill sets; each role requires 3 skills.
    Candidate u is eligible for role j if they cover >= min_overlap required skills.
    Total capacity ~= tightness * n_cand, so roles are scarce and choices matter."""
    rng = random.Random(seed)
    ns = len(SKILLS)
    cands = [frozenset(rng.sample(range(ns), rng.choice([2, 3, 3, 4, 5, 6])))
             for _ in range(n_cand)]
    roles = [frozenset(rng.sample(range(ns), 3)) for _ in range(n_roles)]
    w = [rng.random() + 0.3 for _ in range(n_roles)]
    total = max(n_roles, round(tightness * n_cand))
    caps = [max(1, round(x / sum(w) * total)) for x in w]
    eligible = [[j for j in range(n_roles) if len(cands[u] & roles[j]) >= min_overlap]
                for u in range(n_cand)]
    return cands, roles, caps, eligible


def make_sparse_instance(n_cand, n_roles, seed, tightness=0.9, max_deg=6):
    """Large-scale benchmark generator: each candidate is eligible for 1..max_deg roles,
    drawn with a skewed popularity (some roles are in demand by many candidates), so
    average degree stays constant as n grows and contention is realistic."""
    rng = random.Random(seed)
    pop = [1.0 / (j + 1) ** 0.6 for j in range(n_roles)]
    eligible = []
    for _ in range(n_cand):
        d = rng.randint(1, max_deg)
        chosen = set()
        while len(chosen) < min(d, n_roles):
            chosen.add(rng.choices(range(n_roles), weights=pop)[0])
        eligible.append(sorted(chosen))
    w = [rng.random() + 0.3 for _ in range(n_roles)]
    total = max(n_roles, round(tightness * n_cand))
    caps = [max(1, round(x / sum(w) * total)) for x in w]
    return caps, eligible


def build_slots(n_cand, caps, eligible):
    """Expand every role j into caps[j] slots. Returns (adj, slot_role)."""
    slot_role = [j for j, c in enumerate(caps) for _ in range(c)]
    role_slots = {}
    for s, j in enumerate(slot_role):
        role_slots.setdefault(j, []).append(s)
    adj = [[s for j in eligible[u] for s in role_slots.get(j, [])]
           for u in range(n_cand)]
    return adj, slot_role


# --------------------------------------------------------------------------
# 2. Greedy baseline (first eligible open role) and Hopcroft-Karp
# --------------------------------------------------------------------------
def greedy(adj, n_slots):
    """Irrevocable first-fit: each candidate takes the first open eligible slot."""
    ml, mr = [-1] * len(adj), [-1] * n_slots
    for u, nbrs in enumerate(adj):
        for v in nbrs:
            if mr[v] == -1:
                ml[u], mr[v] = v, u
                break
    return ml, mr


def hopcroft_karp(adj, n_slots, init=None, log=None):
    """Maximum bipartite matching. Optional warm start `init=(ml, mr)`.

    Each phase: (1) BFS builds the level graph from all free left vertices,
    (2) DFS finds a maximal set of vertex-disjoint shortest augmenting paths.
    If `log` is a list, every augmenting path is appended as a list of
    (candidate, old_slot, new_slot) triples.  Returns (ml, mr, phases).
    """
    n = len(adj)
    if init:
        ml, mr = list(init[0]), list(init[1])
    else:
        ml, mr = [-1] * n, [-1] * n_slots
    phases = 0

    def bfs():
        dist = [INF] * n
        q = deque()
        for u in range(n):
            if ml[u] == -1:
                dist[u] = 0
                q.append(u)
        found = False
        while q:
            u = q.popleft()
            for v in adj[u]:
                w = mr[v]
                if w == -1:
                    found = True
                elif dist[w] == INF:
                    dist[w] = dist[u] + 1
                    q.append(w)
        return found, dist

    def dfs(root, dist, ptr):
        """Iterative DFS (no recursion limit issues on big graphs)."""
        stack, vs = [root], []
        while stack:
            u = stack[-1]
            if ptr[u] < len(adj[u]):
                v = adj[u][ptr[u]]
                ptr[u] += 1
                w = mr[v]
                if w == -1:                       # free slot: augment along stack
                    vs.append(v)
                    path = []
                    for x, y in zip(stack, vs):
                        path.append((x, ml[x], y))
                        ml[x], mr[y] = y, x
                    if log is not None:
                        log.append(path)
                    return True
                if dist[w] == dist[u] + 1:        # follow matched edge to next level
                    vs.append(v)
                    stack.append(w)
            else:                                 # dead end
                dist[u] = INF
                stack.pop()
                if vs:
                    vs.pop()
        return False

    while True:
        found, dist = bfs()
        if not found:
            break
        phases += 1
        ptr = [0] * n
        for u in range(n):
            if ml[u] == -1:
                dfs(u, dist, ptr)
    return ml, mr, phases


# --------------------------------------------------------------------------
# 3. Optimality certificates
# --------------------------------------------------------------------------
def konig_certificate(adj, ml, mr):
    """Min vertex cover from a maximum matching (Konig). Z = vertices reachable
    from free left vertices by alternating paths. Cover = (L \\ Z) U (R & Z).
    |cover| == |M| proves maximality (weak duality) AND shows no augmenting path
    exists (Berge): if a free slot were in Z we would have an augmenting path."""
    n = len(adj)
    zl, zr = [False] * n, [False] * len(mr)
    q = deque(u for u in range(n) if ml[u] == -1)
    for u in q:
        zl[u] = True
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not zr[v]:
                zr[v] = True
                w = mr[v]
                if w != -1 and not zl[w]:
                    zl[w] = True
                    q.append(w)
    free_slot_reached = any(zr[v] and mr[v] == -1 for v in range(len(mr)))
    cover_l = [u for u in range(n) if not zl[u]]
    cover_r = [v for v in range(len(mr)) if zr[v]]
    covers_all = all((not zl[u]) or zr[v] for u in range(n) for v in adj[u])
    return cover_l, cover_r, covers_all, free_slot_reached


def flow_check(n_cand, caps, eligible):
    """Independent validation: networkx max-flow on the capacitated network."""
    import networkx as nx
    G = nx.DiGraph()
    for u in range(n_cand):
        G.add_edge("s", ("c", u), capacity=1)
        for j in eligible[u]:
            G.add_edge(("c", u), ("r", j), capacity=1)
    for j, c in enumerate(caps):
        G.add_edge(("r", j), "t", capacity=c)
    if "s" not in G or "t" not in G:
        return 0
    return nx.maximum_flow_value(G, "s", "t")


def ilp_max(n_cand, caps, eligible):
    """ILP:  max sum x_uj  s.t.  sum_j x_uj <= 1,  sum_u x_uj <= c_j,  x binary."""
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import coo_matrix
    edges = [(u, j) for u in range(n_cand) for j in eligible[u]]
    m, R = len(edges), len(caps)
    if m == 0:
        return 0
    rows, cols = [], []
    for e, (u, j) in enumerate(edges):
        rows += [u, n_cand + j]
        cols += [e, e]
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_cand + R, m)).tocsr()
    ub = np.array([1] * n_cand + list(caps), dtype=float)
    res = milp(c=-np.ones(m), constraints=LinearConstraint(A, -np.inf, ub),
               integrality=np.ones(m), bounds=Bounds(0, 1))
    return int(round(-res.fun))


def size(ml):
    return sum(1 for x in ml if x != -1)


# --------------------------------------------------------------------------
# 4. Experiments
# --------------------------------------------------------------------------
def solve_instance(n_cand, caps, eligible, with_ilp=True):
    adj, slot_role = build_slots(n_cand, caps, eligible)
    S = len(slot_role)
    g_ml, g_mr = greedy(adj, S)
    log = []
    t0 = time.perf_counter()
    ml, mr, ph = hopcroft_karp(adj, S, init=(g_ml, g_mr), log=log)
    t_hk = time.perf_counter() - t0
    t0 = time.perf_counter()
    ml0, _, ph0 = hopcroft_karp(adj, S)
    t_hk0 = time.perf_counter() - t0
    out = dict(adj=adj, slot_role=slot_role, g_ml=g_ml, ml=ml, mr=mr, log=log,
               greedy_size=size(g_ml), hk_size=size(ml), hk_scratch_size=size(ml0),
               phases=ph, phases_scratch=ph0, t_hk=t_hk, t_hk_scratch=t_hk0,
               edges=sum(len(a) for a in adj), slots=S)
    if with_ilp:
        t0 = time.perf_counter()
        out["ilp_size"] = ilp_max(n_cand, caps, eligible)
        out["t_ilp"] = time.perf_counter() - t0
    return out


def role_of(slot_role, s):
    return None if s == -1 else slot_role[s]


def describe_paths(log, slot_role, names_c, names_r):
    """Human-readable augmenting-path traces."""
    lines = []
    for k, path in enumerate(log, 1):
        steps = []
        for (u, old, new) in path:
            o = "unmatched" if old == -1 else names_r[slot_role[old]]
            nw = names_r[slot_role[new]]
            if o != nw:      # slot swaps inside the same role are invisible at role level
                steps.append(f"{names_c[u]}: {o} -> {nw}")
        lines.append(f"path {k} (+1 hire, {len(steps)} role change(s)): " + ";  ".join(steps))
    return lines


def demo_example():
    """Hand-traceable 6-candidate example in which greedy provably loses a hire."""
    names_c = ["c1", "c2", "c3", "c4", "c5", "c6"]
    names_r = ["R1", "R2", "R3", "R4"]
    caps = [1, 1, 1, 3]
    elig = [[0, 1], [0], [1, 2], [2, 3], [3], [3]]
    r = solve_instance(6, caps, elig, with_ilp=True)
    return dict(
        eligibility={names_c[u]: [names_r[j] for j in elig[u]] for u in range(6)},
        capacities=dict(zip(names_r, caps)),
        greedy=[(names_c[u], role_of(r["slot_role"], r["g_ml"][u]) is not None and
                 names_r[role_of(r["slot_role"], r["g_ml"][u])] or "unmatched") for u in range(6)],
        final=[(names_c[u], names_r[role_of(r["slot_role"], r["ml"][u])]
                if r["ml"][u] != -1 else "unmatched") for u in range(6)],
        greedy_size=r["greedy_size"], hk_size=r["hk_size"], ilp_size=r["ilp_size"],
        traces=describe_paths(r["log"], r["slot_role"], names_c, names_r))


def draw_matching(path, n_cand, caps, eligible, res, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    slot_role = res["slot_role"]
    final = {u: role_of(slot_role, res["ml"][u]) for u in range(n_cand)}
    greedy_a = {u: role_of(slot_role, res["g_ml"][u]) for u in range(n_cand)}
    G = nx.Graph()
    L = [("c", u) for u in range(n_cand)]
    R = [("r", j) for j in range(len(caps))]
    G.add_nodes_from(L, bipartite=0)
    G.add_nodes_from(R, bipartite=1)
    pos = {}
    for i, n in enumerate(L):
        pos[n] = (0, -i)
    scale = (n_cand - 1) / max(1, len(R) - 1)
    for i, n in enumerate(R):
        pos[n] = (3, -i * scale)
    unmatched_e, kept_e, moved_e = [], [], []
    for u in range(n_cand):
        for j in eligible[u]:
            e = (("c", u), ("r", j))
            if final[u] == j:
                (kept_e if greedy_a[u] == j else moved_e).append(e)
            else:
                unmatched_e.append(e)
    fig, ax = plt.subplots(figsize=(11, max(8, n_cand * 0.28)))
    nx.draw_networkx_edges(G, pos, edgelist=unmatched_e, edge_color="#cccccc", width=0.6, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=kept_e, edge_color="#2a9d4a", width=2.2, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=moved_e, edge_color="#e63946", width=3.0, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in L if final[n[1]] is not None],
                           node_color="#2a9d4a", node_size=170, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in L if final[n[1]] is None],
                           node_color="#999999", node_size=170, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=R, node_color="#457b9d", node_size=1300, node_shape="s", ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: f"c{n[1]}" for n in L}, font_size=6, font_color="white", ax=ax)
    used = [0] * len(caps)
    for u in range(n_cand):
        if final[u] is not None:
            used[final[u]] += 1
    nx.draw_networkx_labels(G, pos, labels={("r", j): f"R{j}\n{used[j]}/{caps[j]}" for j in range(len(caps))},
                            font_size=8, font_color="white", ax=ax)
    ax.plot([], [], color="#2a9d4a", lw=2.2, label="matched (same as greedy)")
    ax.plot([], [], color="#e63946", lw=3, label="matched after augmentation (greedy choice undone / new hire)")
    ax.plot([], [], color="#cccccc", lw=1, label="eligible, not used")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.04), fontsize=8, ncol=1, frameon=False)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    N, R = 40, 8
    # ---- main instance: first seed whose greedy gap is non-trivial (documented) ----
    seed = None
    for s in range(1, 500):
        cands, roles, caps, eligible = make_instance(N, R, s)
        adj, sr = build_slots(N, caps, eligible)
        gml, _ = greedy(adj, len(sr))
        ml, _, _ = hopcroft_karp(adj, len(sr))
        if size(ml) - size(gml) >= 4:
            seed = s
            break
    cands, roles, caps, eligible = make_instance(N, R, seed)
    res = solve_instance(N, caps, eligible)
    slot_role = res["slot_role"]
    flow = flow_check(N, caps, eligible)
    cl, cr, covers, free_reached = konig_certificate(res["adj"], res["ml"], res["mr"])
    assert res["hk_size"] == res["ilp_size"] == flow == res["hk_scratch_size"], "optimality mismatch"
    assert covers and not free_reached and len(cl) + len(cr) == res["hk_size"]

    names_c = [f"c{u}" for u in range(N)]
    names_r = [f"R{j}" for j in range(R)]
    with open(os.path.join(HERE, "assignment.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate", "skills", "assigned_role", "greedy_role", "changed_by_augmentation"])
        for u in range(N):
            a, g = role_of(slot_role, res["ml"][u]), role_of(slot_role, res["g_ml"][u])
            w.writerow([names_c[u], "|".join(SKILLS[k] for k in sorted(cands[u])),
                        names_r[a] if a is not None else "UNASSIGNED",
                        names_r[g] if g is not None else "UNASSIGNED",
                        "yes" if a != g else "no"])
    with open(os.path.join(HERE, "role_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["role", "required_skills", "capacity", "filled_greedy", "filled_final"])
        for j in range(R):
            fg = sum(1 for u in range(N) if role_of(slot_role, res["g_ml"][u]) == j)
            ff = sum(1 for u in range(N) if role_of(slot_role, res["ml"][u]) == j)
            w.writerow([names_r[j], "|".join(SKILLS[k] for k in sorted(roles[j])), caps[j], fg, ff])

    traces = describe_paths(res["log"], slot_role, names_c, names_r)
    draw_matching(os.path.join(HERE, "matching_vis.png"), N, caps, eligible, res,
                  f"Maximum matching: {res['hk_size']} hires (greedy: {res['greedy_size']}, "
                  f"gap = {res['hk_size'] - res['greedy_size']})")

    demo = demo_example()

    # ---- benchmark vs ILP over growing n (3 seeds each) ----
    bench = []
    for n in [100, 200, 400, 800, 1600, 3200]:
        rows = []
        for s in range(3):
            cp, el = make_sparse_instance(n, max(4, n // 8), 1000 + s)
            rr = solve_instance(n, cp, el)
            assert rr["hk_size"] == rr["ilp_size"] == rr["hk_scratch_size"], f"mismatch n={n}"
            rows.append(rr)
        avg = lambda k: sum(x[k] for x in rows) / len(rows)
        bench.append(dict(n=n, roles=max(4, n // 8), edges=avg("edges"), slots=avg("slots"),
                          greedy=avg("greedy_size"), optimum=avg("hk_size"),
                          gap=avg("hk_size") - avg("greedy_size"),
                          t_hk=avg("t_hk_scratch"), t_ilp=avg("t_ilp"), phases=avg("phases_scratch")))

    # ---- HK-only scaling for the sqrt(V) / E sqrt(V) claim ----
    scaling = []
    for n in [1000, 2000, 4000, 8000, 16000, 32000, 64000]:
        cp, el = make_sparse_instance(n, n // 8, 7)
        adj, sr = build_slots(n, cp, el)
        t0 = time.perf_counter()
        ml, mr, ph = hopcroft_karp(adj, len(sr))
        t = time.perf_counter() - t0
        scaling.append(dict(n=n, edges=sum(len(a) for a in adj), t=t, phases=ph, hires=size(ml)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ns = [b["n"] for b in bench]
    ax[0].loglog(ns, [b["t_hk"] for b in bench], "o-", label="Hopcroft-Karp (hand-written)")
    ax[0].loglog(ns, [b["t_ilp"] for b in bench], "s-", label="ILP (scipy milp)")
    ax[0].set_xlabel("candidates n"); ax[0].set_ylabel("seconds"); ax[0].legend(); ax[0].set_title("Runtime vs ILP")
    ax[1].bar(range(len(ns)), [b["optimum"] for b in bench], color="#2a9d4a", label="optimum (HK = ILP)")
    ax[1].bar(range(len(ns)), [b["greedy"] for b in bench], color="#e63946", label="greedy")
    ax[1].set_xticks(range(len(ns))); ax[1].set_xticklabels(ns)
    ax[1].set_xlabel("candidates n"); ax[1].set_ylabel("hires"); ax[1].legend(); ax[1].set_title("Greedy Capacity Gap")
    sn = [s["n"] for s in scaling]
    ev = [s["edges"] * (s["n"]) ** 0.5 for s in scaling]
    k = scaling[-1]["t"] / ev[-1]
    ax[2].loglog(sn, [s["t"] for s in scaling], "o-", label="measured HK time")
    ax[2].loglog(sn, [k * e for e in ev], "--", label="c * E*sqrt(V) reference")
    ax[2].set_xlabel("candidates n"); ax[2].set_ylabel("seconds"); ax[2].legend(); ax[2].set_title("HK scaling")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "benchmark.png"), dpi=150)
    plt.close()

    results = dict(
        main=dict(seed=seed, candidates=N, roles=R, capacities=caps, total_capacity=sum(caps),
                  edges=res["edges"], slots=res["slots"], greedy_size=res["greedy_size"],
                  hk_size=res["hk_size"], ilp_size=res["ilp_size"], networkx_flow=flow,
                  gap=res["hk_size"] - res["greedy_size"], phases=res["phases"],
                  augmenting_paths=len(res["log"]), t_hk=res["t_hk"], t_ilp=res["t_ilp"],
                  cover_size=len(cl) + len(cr), cover_left=len(cl), cover_right=len(cr),
                  unassigned=N - res["hk_size"], traces=traces),
        demo=demo, benchmark=bench, scaling=scaling)
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    m = results["main"]
    print(f"[main] seed={seed}  n={N} roles={R} capacity={m['total_capacity']} edges={m['edges']}")
    print(f"[main] greedy={m['greedy_size']}  HK={m['hk_size']}  ILP={m['ilp_size']}  "
          f"networkx flow={m['networkx_flow']}  gap={m['gap']}")
    print(f"[main] König cover size={m['cover_size']} (= matching size)  phases={m['phases']}")
    print("[demo]", demo["greedy_size"], "->", demo["hk_size"])
    for t in demo["traces"]:
        print("   ", t)
    print("[bench] n, greedy, optimum, t_hk, t_ilp")
    for b in bench:
        print(f"   {b['n']:>5} {b['greedy']:>8.1f} {b['optimum']:>8.1f} {b['t_hk']:.4f} {b['t_ilp']:.4f}")
    print("[scaling] n, edges, time, phases")
    for s in scaling:
        print(f"   {s['n']:>6} {s['edges']:>8} {s['t']:.3f}s phases={s['phases']}")


if __name__ == "__main__":
    main()
