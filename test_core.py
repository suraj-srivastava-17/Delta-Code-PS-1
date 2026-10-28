import os, random, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from allocate import (build_slots, flow_check, greedy, hopcroft_karp, ilp_max, konig_certificate,
                      make_instance, size)
from merit import merit_assign

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def solve(n, caps, el):
    adj, sr = build_slots(n, caps, el)
    g = greedy(adj, len(sr))
    return adj, hopcroft_karp(adj, len(sr), init=g)


def test_greedy_loses_hk_recovers():
    el = [[0, 1], [0], [1, 2], [2, 3], [3], [3]]
    adj, (ml, mr, _) = solve(6, [1, 1, 1, 3], el)
    assert size(greedy(adj, len(mr))[0]) == 5 and size(ml) == 6


def test_random_matches_ilp_and_flow_and_konig():
    for seed in range(6):
        rng = random.Random(seed)
        n, R = 60, 9
        caps = [rng.randint(1, 8) for _ in range(R)]
        el = [sorted(rng.sample(range(R), rng.randint(0, 3))) for _ in range(n)]
        adj, (ml, mr, _) = solve(n, caps, el)
        assert size(ml) == ilp_max(n, caps, el) == flow_check(n, caps, el)
        cl, cr, covers, free = konig_certificate(adj, ml, mr)
        assert covers and not free and len(cl) + len(cr) == size(ml)


def test_edge_cases():
    assert size(solve(0, [1], [])[1][0]) == 0
    assert size(solve(3, [2], [[], [], []])[1][0]) == 0          # nobody eligible
    assert size(solve(3, [0, 0], [[0], [1], [0, 1]])[1][0]) == 0  # zero capacity
    assert size(solve(1, [5], [[0]])[1][0]) == 1


def test_merit_keeps_cardinality_and_improves_fit():
    n, caps, el = 6, [1, 1, 1, 3], [[0, 1], [0], [1, 2], [2, 3], [3], [3]]
    fit = {(u, j): 1 for u in range(n) for j in el[u]}
    fit[(0, 1)] = 5                      # c0 is a far better fit for R2 than R1
    fit[(1, 0)] = 1
    out, tot = merit_assign(n, caps, el, fit, [f"c{u}" for u in range(n)])
    assert sum(x != -1 for x in out) == 6 and out[0] == 1


def test_merit_order_independent_for_equal_profiles():
    # 2 identical candidates, 1 seat: winner must not depend on file order
    el, caps = [[0], [0]], [1]
    fit = {(0, 0): 3, (1, 0): 3}
    w1 = merit_assign(2, caps, el, fit, ["a", "b"])[0]
    w2 = merit_assign(2, caps, el, fit, ["b", "a"])[0]
    win1 = "ab"[w1.index(0)]
    win2 = "ba"[w2.index(0)]
    assert win1 == win2


def test_cli_sample(tmp_path):
    out = tmp_path / "o.csv"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "match_csv.py"),
                        os.path.join(ROOT, "sample_candidates.csv"),
                        os.path.join(ROOT, "sample_roles.csv"), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "certificate (vertex cover = matching size): OK" in r.stdout
    assert "explanation" in out.read_text()


def test_cli_bad_input(tmp_path):
    bad = tmp_path / "b.csv"
    bad.write_text("x,y\n1,2\n")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "match_csv.py"), str(bad), str(bad)],
                       capture_output=True, text=True)
    assert r.returncode != 0


def test_main_instance_gap():
    cands, roles, caps, el = make_instance(2000, 320, 1)
    adj, (ml, mr, _) = solve(2000, caps, el)
    assert size(ml) == flow_check(2000, caps, el) == 1696


def _run(tmp_path, cands, roles):
    c, r, o = tmp_path / "c.csv", tmp_path / "r.csv", tmp_path / "o.csv"
    c.write_text(cands); r.write_text(roles)
    p = subprocess.run([sys.executable, os.path.join(ROOT, "match_csv.py"), str(c), str(r),
                        "--out", str(o)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    import csv
    return {x["candidate"]: x for x in csv.DictReader(open(o))}


def test_mandatory_missing_info_and_tie(tmp_path):
    rows = _run(tmp_path,
        "candidate,skills\nA,python|sql\nB,python|sql\nC,\nD,python|stats\n",
        "role,capacity,required_skills,mandatory_skills\nX,1,python|sql|ml,sql\nY,1,python|stats|sql,\n")
    assert "MISSING INFORMATION" in rows["C"]["explanation"]
    assert rows["C"]["assigned_role"] == "UNASSIGNED"
    assert "mandatory" in rows["D"]["explanation"] or rows["D"]["assigned_role"] == "Y"
    assert "TIE" in rows["A"]["explanation"] or "TIE" in rows["B"]["explanation"] \
        or "next best" in rows["A"]["explanation"] + rows["B"]["explanation"]
    assert rows["D"]["assigned_role"] != "X"      # D lacks mandatory sql for X
