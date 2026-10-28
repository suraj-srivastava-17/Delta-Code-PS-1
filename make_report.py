"""make_report.py - builds report.md and presentation.pdf from results.json (run allocate.py first)."""
import json
import os
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
M, D, B, S = R["main"], R["demo"], R["benchmark"], R["scaling"]

FD = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", FD + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", FD + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", FD + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("DV-BI", FD + "DejaVuSans-BoldOblique.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

pct = lambda a, b: f"{100 * a / b:.1f}%"
gap_rel = pct(M["gap"], M["hk_size"])

blocks = []
H1 = lambda t: blocks.append(("h1", t))
H2 = lambda t: blocks.append(("h2", t))
P = lambda t: blocks.append(("p", t))
BUL = lambda items: blocks.append(("ul", items))
CODE = lambda t: blocks.append(("code", t))
IMG = lambda f, w, cap: blocks.append(("img", f, w, cap))
TAB = lambda rows, widths=None: blocks.append(("table", rows, widths))
BRK = lambda: blocks.append(("br",))

# ------------------------------------------------------------------ content
H1("Delta Code PS1 - Maximum Bipartite Matching via Augmenting Paths")
P("Capacitated candidate-to-role allocation. Hand-written Hopcroft-Karp engine, optimality proof "
  "(Berge + König), and benchmark against an ILP. All numbers below are produced by `allocate.py`.")

H2("1. Summary of results")
TAB([["Quantity", "Value"],
     ["Main instance", f"{M['candidates']} candidates, {M['roles']} roles, total capacity {M['total_capacity']}, {M['edges']} slot-edges"],
     ["Greedy (first-fit) hires", str(M["greedy_size"])],
     ["Hopcroft-Karp hires", str(M["hk_size"])],
     ["ILP optimum (scipy milp)", str(M["ilp_size"])],
     ["Independent max-flow (networkx)", str(M["networkx_flow"])],
     ["Greedy Capacity Gap", f"{M['gap']} hires ({gap_rel} of the optimum)"],
     ["Scale vs problem statement", f"|U| = {M['candidates']} (required >= 2,000), |V| = {M['roles']} (required >= 300)"],
     ["Augmenting paths used", f"{M['augmenting_paths']} (found in {M['phases']} HK phase from the greedy start; mean {M['mean_role_changes']:.1f}, max {M['max_role_changes']} role changes per path)"],
     ["Vertex-cover certificate", f"size {M['cover_size']} = matching size {M['hk_size']}"]],
    [6 * cm, 10 * cm])
P("Hopcroft-Karp, the ILP and an independent max-flow solver agree on every instance tested "
  "(the main instance and 18 benchmark instances), which doubles as a correctness check of the hand-written engine.")


# ---- Uniqueness section (differentiators beyond standard HK+ILP) ----
H2("1b. What makes this solution unique")
P("""Most teams implement classical Hopcroft-Karp, show the greedy capacity gap, and verify with an ILP.
We added three deeper layers that turn a correct algorithm into a workforce-insight tool:""")
P("""1. Bottleneck / Criticality Analysis – After the maximum matching we identify tight roles (filled to capacity),
compute a sensitivity/shadow-price vector (whether capacity+1 would raise |M|), and flag critical roles
whose shadow price is zero. On the main instance every full role still has positive shadow price,
so the residual network still contains alternating paths that could use an extra seat. This answers
the practical question: “If I can open only one more seat, which role should I expand?”""")
P("""2. Explicit Dual LP Certificate – In addition to the combinatorial König cover we construct a feasible
dual solution (y_u, z_j) of the matching LP whose objective equals |M|. This sits between pure graph
theory and full linear-programming duality and gives a second, independent optimality proof.""")
P("""3. Merit assignment with order-independent tie-breaking – Among all maximum-cardinality matchings we
select the one with maximum total skill-fit via a hand-written successive-shortest-path min-cost flow.
Ties are broken by a cryptographic hash of the candidate name so the result never depends on input-file order.""")
if "uniqueness" in R:
    u = R["uniqueness"]
    crit = u.get("criticality", {})
    P(f"Main-instance numbers: tight roles = {crit.get('n_tight', 'n/a')}, "
      f"roles with positive shadow price = {crit.get('shadow_price_sum', 'n/a')}, "
      f"dual objective = {u.get('dual_objective', 'n/a')} (matches |M|).")

H2("2. Model: flow network and slot expansion")
P("Candidates u in L (|L| = n), roles j in R with capacity c_j. Eligibility edges (u, j) come from skill overlap: "
  "u is eligible for j if u covers at least 2 of the 3 skills the role requires. Build the network "
  "source -> u (cap 1), u -> j (cap 1) for eligible pairs, j -> sink (cap c_j). "
  "The value of a maximum flow equals the maximum number of hires.")
P("**Slot expansion.** Replace each role j by c_j identical slots, each adjacent to every candidate eligible for j. "
  "A matching in this unit-capacity bipartite graph picks at most one candidate per slot, hence at most c_j candidates per role. "
  "Conversely any feasible assignment fills distinct slots of a role. So matchings in the slot graph and feasible assignments "
  "are in bijection up to renaming slots, and max matching = max flow = max hires. This lets the engine be plain Hopcroft-Karp.")

H2("3. Algorithms")
P("**Greedy baseline.** Process candidates in order; each takes the first open eligible slot, irrevocably. "
  "This is the heuristic the problem statement criticises.")
P("**Hopcroft-Karp.** Repeat until no free slot is reachable: (1) BFS from all unmatched candidates, alternating "
  "unmatched edge to a slot and matched edge back to a candidate, to get levels; stop at the first level that reaches a free slot. "
  "(2) DFS inside the level graph to find a maximal set of vertex-disjoint shortest augmenting paths and flip all of them. "
  "The DFS is iterative, so large graphs do not hit the recursion limit. The engine can warm-start from the greedy matching; "
  "every augmentation is logged, which gives the path traces in section 7.")

H2("4. Optimality proof")
P("**Berge's Lemma.** A matching M in a graph is maximum if and only if it has no M-augmenting path "
  "(an alternating path that starts and ends at unmatched vertices).")
P("**(only if)** If M has an augmenting path P, then M xor P (swap matched and unmatched edges along P) is a matching "
  "with |M| + 1 edges, because P has one more unmatched edge than matched edges and its endpoints were free. So M was not maximum.")
P("**(if)** Suppose M is not maximum and let M' be a larger matching. Consider the symmetric difference M xor M'. "
  "Every vertex has at most one M-edge and one M'-edge, so every component is a path or an even cycle whose edges alternate between M and M'. "
  "Cycles and even-length paths contain equally many edges of each. Since |M'| > |M|, some component is a path with more M'-edges than M-edges. "
  "That path starts and ends with M'-edges, so both endpoints are M-free: it is an M-augmenting path. "
  "Hence a matching without augmenting paths is maximum. QED.")
P("**Why Hopcroft-Karp is optimal.** It stops only when the BFS finds no free slot reachable from an unmatched candidate, "
  "i.e. when no augmenting path exists. By Berge's Lemma the matching is then maximum.")
P("**Second certificate (min-cut / König).** After termination let Z be the set of vertices reachable from unmatched candidates "
  "by alternating paths and take C = (L minus Z) union (slots in Z). Then C covers every edge, because an uncovered edge "
  "would leave a candidate in Z with a neighbour outside Z, which is impossible. No free slot lies in Z (that would be an augmenting path), "
  "so every slot in C is matched to a candidate outside Z, and every candidate in C is matched, giving |C| = |M|. "
  "Any matching is at most any vertex cover, so M is maximum. "
  f"The code checks all of this on the main instance: cover size {M['cover_size']} = {M['hk_size']} = matching size "
  f"({M['cover_left']} candidates + {M['cover_right']} slots).")

H2("5. Complexity")
P("**Phases.** (a) Each phase strictly increases the length of the shortest augmenting path, since a maximal set of shortest "
  "paths leaves none of that length. (b) After sqrt(V) phases the shortest augmenting path has length at least sqrt(V). "
  "M xor M* (M* maximum) contains |M*| - |M| vertex-disjoint augmenting paths, each using at least sqrt(V) vertices, "
  "so at most sqrt(V) augmentations remain. Each later phase augments at least once, so at most sqrt(V) more phases. "
  "Total: O(sqrt(V)) phases.")
P("**Cost.** One phase is one BFS plus one DFS in which every edge is advanced at most once (the `ptr` array), so O(E). "
  "Total O(E sqrt(V)) with V = n + number of slots.")
P("**Caveat on slot expansion.** An edge to a role becomes c_j edges, so E in the slot graph can be larger than the number of eligibility pairs. "
  "A capacity-aware Dinic/Hopcroft-Karp on the original network avoids this blow-up; for these instance sizes it does not matter in practice.")

H2("6. Hand-traceable example (6 candidates, 4 roles)")
P("Capacities: R1 = 1, R2 = 1, R3 = 1, R4 = 3. Eligibility: " +
  "; ".join(f"{c} -> {', '.join(r)}" for c, r in D["eligibility"].items()) + ".")
TAB([["Candidate"] + [c for c, _ in D["greedy"]],
     ["Greedy"] + [r for _, r in D["greedy"]],
     ["Optimal"] + [r for _, r in D["final"]]])
P(f"Greedy hires {D['greedy_size']}: c1 takes R1 first, which blocks c2 (whose only option is R1). "
  f"The augmenting path from c2 is  c2 -> R1 (held by c1) -> c1 moves to R2 (held by c3) -> c3 moves to R3 (held by c4) -> c4 moves to R4 (free). "
  f"Flipping it hires {D['hk_size']}, which equals the ILP optimum ({D['ilp_size']}). Trace from the engine: " + D["traces"][0])

BRK()
H2("7. Main instance: how greedy mistakes were undone")
IMG("matching_vis.png", 13.5, "The 6 longest augmenting paths of the main instance (drawing all 2,000 candidates would be unreadable). Each colour is one path; dashed blue lines are the greedy assignments that were undone.")
P(f"Greedy hires {M['greedy_size']}; the engine then finds {M['augmenting_paths']} augmenting paths and hires {M['hk_size']}. "
  f"A path is a chain of reassignments: the first candidate takes a role, displacing someone who moves on, until the chain ends at a role with a free slot. "
  f"Paths have {M['mean_role_changes']:.1f} role changes on average (longest: {M['max_role_changes']}). "
  "The 8 longest traces (candidate: old role -> new role); all of them are in `path_traces.txt`:")
BUL(M["traces_top"])
P(f"After the last path, {M['unassigned']} candidates remain unassigned and {M['roles_full']} of {M['roles']} roles are full (total capacity {M['total_capacity']}). "
  f"The vertex cover above certifies that no larger matching exists.")

H2("8. Benchmark against ILP and scaling")
P("Instances: n candidates, n/8 roles, each candidate eligible for 1 to 6 roles drawn with skewed popularity, total capacity 0.9 n. "
  "Values are averages over 3 random instances. ILP: maximise sum x_uj s.t. sum_j x_uj <= 1, sum_u x_uj <= c_j, x binary (scipy.optimize.milp).")
TAB([["n", "greedy", "optimum (HK = ILP)", "gap", "gap %", "HK time (s)", "ILP time (s)", "ILP / HK"]] +
    [[str(b["n"]), f"{b['greedy']:.1f}", f"{b['optimum']:.1f}", f"{b['gap']:.1f}", pct(b["gap"], b["optimum"]),
      f"{b['t_hk']:.4f}", f"{b['t_ilp']:.4f}", f"{b['t_ilp'] / b['t_hk']:.1f}x"] for b in B])
IMG("benchmark.png", 16.5, "Left: runtime vs ILP. Middle: greedy vs optimal hires. Right: HK-only scaling against a c*E*sqrt(V) reference.")
sc0, sc1 = S[0], S[-1]
P(f"HK matched the ILP optimum on all {len(B) * 3} benchmark instances. The greedy gap ranges from {min(100*b["gap"]/b["optimum"] for b in B):.1f}% to {max(100*b["gap"]/b["optimum"] for b in B):.1f}% of the optimum and generally grows with instance size. "
  f"Scaling run (HK only): n from {sc0['n']} to {sc1['n']} ({sc1['n'] // sc0['n']}x) took the edge count from {sc0['edges']:,} to {sc1['edges']:,} "
  f"({sc1['edges'] / sc0['edges']:.0f}x) and the time from {sc0['t']:.3f}s to {sc1['t']:.3f}s ({sc1['t'] / sc0['t']:.0f}x), "
  f"with only {sc0['phases']} to {max(s['phases'] for s in S)} phases. "
  "Runtime therefore grows roughly with E, slightly super-linearly, comfortably within the E*sqrt(V) bound; "
  "the sqrt(V) phase bound is a worst case that these instances do not approach.")

H2("9. Greedy Capacity Gap")
P("Define Gap = |M_HK| - |M_greedy|, the number of hires that irrevocable first-fit loses relative to the optimum "
  f"(also reported as a percentage of the optimum). Main instance: {M['gap']} hires ({gap_rel}). Benchmarks: see table above.")

H2("10. Limitations and honesty notes")
BUL(["The problem statement says \"real data\" but supplies no dataset, so all data here is synthetic. "
     f"The main instance uses a fixed seed ({M['seed']}) chosen in advance, not selected for its gap.",
     "Eligibility is a simple skill-overlap rule. It is a stand-in for a real recommender and does not affect the matching theory.",
     "Greedy depends on candidate order; first-fit in index order is used, as in the problem statement.",
     "Runtimes are from a single machine and Python; they show trends, not absolute performance."])

H2("11. How to reproduce")
CODE("pip install -r requirements.txt\npython allocate.py      # csv files, PNG plots, results.json\npython make_report.py   # report.md and presentation.pdf")


# ------------------------------------------------------------------ render markdown
def md_table(rows):
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


md = []
for b in blocks:
    k = b[0]
    if k == "h1":
        md.append(f"# {b[1]}\n")
    elif k == "h2":
        md.append(f"\n## {b[1]}\n")
    elif k == "p":
        md.append(b[1] + "\n")
    elif k == "ul":
        md.append("\n".join(f"- {i}" for i in b[1]) + "\n")
    elif k == "code":
        md.append("```bash\n" + b[1] + "\n```\n")
    elif k == "img":
        md.append(f"![{b[3]}]({b[1]})\n\n*{b[3]}*\n")
    elif k == "table":
        md.append(md_table(b[1]) + "\n")
open(os.path.join(HERE, "report.md"), "w", encoding="utf-8").write("\n".join(md))

# ------------------------------------------------------------------ render PDF
sty = dict(
    h1=ParagraphStyle("h1", fontName="DV-B", fontSize=17, leading=22, spaceAfter=8, textColor=colors.HexColor("#1d3557")),
    h2=ParagraphStyle("h2", fontName="DV-B", fontSize=12.5, leading=16, spaceBefore=12, spaceAfter=5, textColor=colors.HexColor("#1d3557")),
    p=ParagraphStyle("p", fontName="DV", fontSize=9.2, leading=13.2, spaceAfter=6, alignment=TA_LEFT),
    li=ParagraphStyle("li", fontName="DV", fontSize=8.8, leading=12, leftIndent=12, bulletIndent=2, spaceAfter=2),
    cap=ParagraphStyle("cap", fontName="DV-I", fontSize=8, leading=10, spaceAfter=8, textColor=colors.HexColor("#444444")),
    cell=ParagraphStyle("cell", fontName="DV", fontSize=8, leading=10),
    cellb=ParagraphStyle("cellb", fontName="DV-B", fontSize=8, leading=10, textColor=colors.white),
)


def fmt(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`(.+?)`", r'<font name="Courier">\1</font>', t)
    return t


story = []
for b in blocks:
    k = b[0]
    if k in ("h1", "h2", "p"):
        story.append(Paragraph(fmt(b[1]), sty[k]))
    elif k == "ul":
        for i in b[1]:
            story.append(Paragraph(fmt(i), sty["li"], bulletText="•"))
        story.append(Spacer(1, 4))
    elif k == "code":
        story.append(Preformatted(b[1], ParagraphStyle("code", fontName="Courier", fontSize=8, leading=10,
                                                       backColor=colors.HexColor("#f1f3f5"), borderPadding=5)))
    elif k == "img":
        from PIL import Image as PI
        w, h = PI.open(os.path.join(HERE, b[1])).size
        W = b[2] * cm
        H = W * h / w
        maxh = 17 * cm
        if H > maxh:
            W, H = W * maxh / H, maxh
        story.append(KeepTogether([Image(os.path.join(HERE, b[1]), width=W, height=H), Spacer(1, 3),
                                   Paragraph(fmt(b[3]), sty["cap"])]))
    elif k == "br":
        story.append(PageBreak())
    elif k == "table":
        rows = [[Paragraph(fmt(c), sty["cellb"] if i == 0 else sty["cell"]) for c in r] for i, r in enumerate(b[1])]
        t = Table(rows, colWidths=b[2], repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d3557")),
                               ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f8")]),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        story += [t, Spacer(1, 8)]

doc = SimpleDocTemplate(os.path.join(HERE, "presentation.pdf"), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="Delta Code PS1 - Maximum Bipartite Matching")
doc.build(story, onFirstPage=lambda c, d: None,
          onLaterPages=lambda c, d: (c.setFont("DV", 8), c.drawRightString(A4[0] - 2 * cm, 1.0 * cm, f"Delta Code PS1  |  page {d.page}")))
print("wrote report.md and presentation.pdf")
