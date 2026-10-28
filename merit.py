"""merit.py - transparent, fair candidate-role selection among maximum matchings.

Hopcroft-Karp returns *a* maximum matching, but many exist and which candidates get
the roles is arbitrary (depends on list order). This module picks, among all
maximum-cardinality assignments, the one with the highest total skill fit, using a
hand-written min-cost max-flow (successive shortest paths, Dijkstra + potentials).
Cost of edge (u, j) = len(required_j) - |skills_u & required_j|, so cost is 0 for a
full skill match. Equal-fit competitors are broken by a seeded hash of the candidate
name (not by file order), so no candidate is favoured by position.
"""
import hashlib
import heapq


def tiebreak(name, seed=0):
    return int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:8], 16)


def merit_assign(n, caps, eligible, fit, names=None, seed=0):
    """fit[u][j] = overlap score (higher better) for eligible pairs, as dict {(u,j): int}.
    Returns (assigned_role list with -1 for none, total_fit)."""
    R = len(caps)
    N = n + R + 2
    s, t = n + R, n + R + 1
    g = [[] for _ in range(N)]          # edge: [to, cap, cost, rev_index]

    def add(a, b, cap, cost):
        g[a].append([b, cap, cost, len(g[b])])
        g[b].append([a, 0, -cost, len(g[a]) - 1])

    top = max(fit.values(), default=0)
    # cost scaled so tie-break noise (< 1) (< 1000 << 2^20) can never override a fit difference
    SC = 1 << 20
    for u in range(n):
        tb = tiebreak(names[u], seed) if names else u
        add(s, u, 1, 0)
        for j in eligible[u]:
            add(u, n + j, 1, (top - fit[(u, j)]) * SC + tb % 1000)
    for j in range(R):
        if caps[j] > 0:
            add(n + j, t, caps[j], 0)
    pot = [0] * N
    INF = float("inf")
    while True:
        dist = [INF] * N
        prev = [None] * N
        dist[s] = 0
        pq = [(0, s)]
        while pq:
            d, v = heapq.heappop(pq)
            if d > dist[v]:
                continue
            for i, (w, c, cost, _) in enumerate(g[v]):
                if c > 0:
                    nd = d + cost + pot[v] - pot[w]
                    if nd < dist[w]:
                        dist[w] = nd
                        prev[w] = (v, i)
                        heapq.heappush(pq, (nd, w))
        if dist[t] == INF:
            break
        for v in range(N):
            if dist[v] < INF:
                pot[v] += dist[v]
        v = t
        while v != s:
            p, i = prev[v]
            e = g[p][i]
            e[1] -= 1
            g[v][e[3]][1] += 1
            v = p
    out, total = [-1] * n, 0
    for u in range(n):
        for w, c, _, _ in g[u]:
            if n <= w < n + R and c == 0:
                # forward edge saturated (reverse edges to s have w == s, excluded)
                out[u] = w - n
                total += fit[(u, w - n)]
    return out, total


def explain(u, j, cand_skills, role_skills):
    got = sorted(cand_skills & role_skills)
    miss = sorted(role_skills - cand_skills)
    return f"matches {len(got)}/{len(role_skills)} required skills ({'|'.join(got)}); missing: {'|'.join(miss) or 'none'}"
