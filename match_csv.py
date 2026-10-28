"""
match_csv.py - run the matching engine on YOUR OWN files.

    python match_csv.py candidates.csv roles.csv
    python match_csv.py candidates.csv roles.csv --min-overlap 2 --out my_assignment.csv

candidates.csv  columns:  candidate, skills            (skills separated by | or ;)
                    OR :  candidate, eligible_roles    (role names separated by | or ;)
roles.csv       columns:  role, capacity [, required_skills]   (required_skills needed
                          only when candidates are given as skills)

A candidate is eligible for a role if they have at least --min-overlap of the role's
required skills (default 2, or all of them if the role requires fewer).
"""
import argparse
import csv
import re
import sys
import time

from merit import explain, merit_assign
from allocate import (build_slots, flow_check, greedy, hopcroft_karp, ilp_max,
                      konig_certificate, role_of, size)

split = lambda s: [x.strip().lower() for x in re.split(r"[|;]", s or "") if x.strip()]


def read(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [{(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
                for row in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("roles")
    ap.add_argument("--min-overlap", type=int, default=2)
    ap.add_argument("--out", default="assignment_custom.csv")
    ap.add_argument("--no-ilp", action="store_true", help="skip the ILP check (faster on huge files)")
    a = ap.parse_args()

    C, Rr = read(a.candidates), read(a.roles)
    for need, rows, name in (("candidate", C, "candidates"), ("role", Rr, "roles"), ("capacity", Rr, "roles")):
        if not rows or need not in rows[0]:
            sys.exit(f"{name} file needs a '{need}' column")
    names_c = [r["candidate"] for r in C]
    names_r = [r["role"] for r in Rr]
    idx = {r.lower(): j for j, r in enumerate(names_r)}
    try:
        caps = [int(float(r["capacity"])) for r in Rr]
    except ValueError:
        sys.exit("capacity must be a number in every row of the roles file")

    if "eligible_roles" in C[0]:
        eligible = []
        for r in C:
            bad = [x for x in split(r["eligible_roles"]) if x not in idx]
            if bad:
                sys.exit(f"{r['candidate']}: unknown role(s) {bad}")
            eligible.append(sorted(idx[x] for x in split(r["eligible_roles"])))
    elif "skills" in C[0] and "required_skills" in Rr[0]:
        cs = [set(split(r["skills"])) for r in C]
        rs = [set(split(r["required_skills"])) for r in Rr]
        ms = [set(split(r.get("mandatory_skills", ""))) for r in Rr]
        eligible = [[j for j in range(len(Rr)) if ms[j] <= cs[u]
                     and len(cs[u] & rs[j]) >= min(a.min_overlap, max(1, len(rs[j])))]
                    for u in range(len(C))]
    else:
        sys.exit("candidates need 'eligible_roles' OR ('skills' with roles having 'required_skills')")

    n = len(C)
    adj, slot_role = build_slots(n, caps, eligible)
    S = len(slot_role)
    t0 = time.perf_counter()
    gml, gmr = greedy(adj, S)
    ml, mr, phases = hopcroft_karp(adj, S, init=(gml, gmr))
    dt = time.perf_counter() - t0
    cl, cr, covers, free = konig_certificate(adj, ml, mr)

    skill_mode = "skills" in C[0] and "required_skills" in Rr[0] and "eligible_roles" not in C[0]
    if skill_mode:
        fit = {(u, j): len(cs[u] & rs[j]) for u in range(n) for j in eligible[u]}
    else:
        fit = {(u, j): 1 for u in range(n) for j in eligible[u]}
    mrole, mfit = merit_assign(n, caps, eligible, fit, names_c)
    assert sum(1 for x in mrole if x != -1) == size(ml), "merit assignment lost cardinality"
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate", "assigned_role", "fit", "explanation", "greedy_role",
                    "hk_role", "changed_by_augmentation"])
        for u in range(n):
            x = mrole[u] if mrole[u] != -1 else None
            h, g = role_of(slot_role, ml[u]), role_of(slot_role, gml[u])
            if skill_mode and not cs[u]:
                ex = "MISSING INFORMATION: no skills listed for this candidate (not evaluated, not a skill gap)"
            elif x is not None and skill_mode:
                ex = explain(u, x, cs[u], rs[x])
                alts = sorted(((fit[(u, j)], names_r[j]) for j in eligible[u] if j != x), reverse=True)
                if alts:
                    best = alts[0]
                    if best[0] > fit[(u, x)]:
                        ex += f"; better-fit role {best[1]} ({best[0]}) was full - seat given to maximise total hires"
                    elif best[0] == fit[(u, x)]:
                        ex += f"; TIE with {best[1]} (same fit {best[0]}): resolved by capacity, then name hash"
                    else:
                        ex += f"; next best {best[1]} (fit {best[0]})"
            elif x is not None:
                ex = "listed as eligible"
            elif eligible[u]:
                ex = "no seat: all eligible roles are full; total fit/hires is maximal without this candidate"
            elif skill_mode:
                blk = sorted({names_r[j] + " needs " + "|".join(sorted(ms[j] - cs[u]))
                              for j in range(len(Rr)) if ms[j] - cs[u]})[:3]
                ex = "not eligible: " + ("missing mandatory skill(s): " + "; ".join(blk) if blk
                                         else f"fewer than {a.min_overlap} required skills for every role")
            else:
                ex = "not eligible for any role"
            nm = lambda k: names_r[k] if k is not None else "UNASSIGNED"
            w.writerow([names_c[u], nm(x), fit[(u, x)] if x is not None else "", ex, nm(g), nm(h),
                        "yes" if h != g else "no"])

    print(f"candidates={n}  roles={len(Rr)}  total capacity={sum(caps)}  eligibility edges={sum(map(len, eligible))}")
    print(f"greedy hires      : {size(gml)}")
    print(f"optimal hires (HK): {size(ml)}   [{dt * 1000:.1f} ms]")
    print(f"total skill fit of assignment: {mfit} (max-cardinality, max-fit)")
    print(f"Greedy Capacity Gap: {size(ml) - size(gml)}")
    ok_cert = covers and not free and len(cl) + len(cr) == size(ml)
    print(f"optimality certificate (vertex cover = matching size): {'OK' if ok_cert else 'FAILED'}")
    if not a.no_ilp:
        print(f"ILP check: {ilp_max(n, caps, eligible)}   max-flow check: {flow_check(n, caps, eligible)}")
    print(f"written: {a.out}")


if __name__ == "__main__":
    main()
