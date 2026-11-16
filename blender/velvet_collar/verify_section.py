"""Checks the strap threading for collisions and draws the section diagram.

Run:  python3 verify_section.py [out.png]
Needs shapely + matplotlib (not needed for building the model).
"""
import sys
import numpy as np
from shapely.geometry import LineString, Point, box
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import collar_geometry as g

lay = g.Layout()
P, T = g.P, g.T
print(f"cut length {lay.length:.1f} mm, circumference {lay.C:.0f} mm, "
      f"slider {lay.doubled:.1f} mm from male bar, base kappa {lay.base.kappa}")
print("closure error (mm):", lay.base.px[-1] - lay.base.px[0], lay.base.py[-1] - lay.base.py[0])

# --- hardware cross-sections in section space (z = 0 cut) ---------------------
hw = {}
bw, bt = P["slider_bar_w"] / 2, P["slider_bar_t"] / 2
for nm, x in (("slider_a", lay.xa), ("slider_c", lay.s_sl), ("slider_b", lay.xb)):
    hw[nm] = box(x - bw, g.H_SLIDER - bt, x + bw, g.H_SLIDER + bt).buffer(-0.6).buffer(0.6)
hw["bar_M"] = Point(lay.s_m, g.H_BUCKLE).buffer(P["bar_r"])
hw["bar_F"] = Point(lay.s_f, g.H_BUCKLE).buffer(P["bar_r"])
hw["bar_D"] = Point(lay.s_d, g.H_DRING).buffer(P["bar_r"])
hw["male_base"] = box(4, g.H_BUCKLE - 3.85, 11, g.H_BUCKLE + 3.85)
hw["female_body"] = box(lay.C - 24, g.H_BUCKLE - 3.85, lay.C + 4, g.H_BUCKLE + 3.85)

# --- strap chunks -------------------------------------------------------------
chunk = 2.0
edges = np.arange(0, lay.length + chunk, chunk)
polys = []
for a, b in zip(edges[:-1], edges[1:]):
    m = (lay.ell >= a - 1e-6) & (lay.ell <= b + 1e-6)
    if m.sum() < 2:
        continue
    polys.append(((a + b) / 2, LineString(lay.pts[m]).buffer(T / 2 - 0.02, cap_style="flat")))
bad = 0
for i, (li, pi) in enumerate(polys):
    for nm, h in hw.items():
        if pi.intersects(h) and pi.intersection(h).area > 1e-4:
            bad += 1; print(f"COLLISION strap@{li:.0f} with {nm}: {pi.intersection(h).area:.3f} mm2")
    for lj, pj in polys[i + 1:]:
        if lj - li > 8 and pi.intersects(pj) and pi.intersection(pj).area > 1e-4:
            bad += 1; print(f"COLLISION strap@{li:.0f} with strap@{lj:.0f}: {pi.intersection(pj).area:.3f} mm2")
print("collisions:", bad)

# --- diagram --------------------------------------------------------------------
web = [LineString(lay.pts[(lay.ell >= a - 1e-6) & (lay.ell <= b + 1e-6)]) for a, b in zip(edges[:-1], edges[1:])
       if ((lay.ell >= a - 1e-6) & (lay.ell <= b + 1e-6)).sum() > 1]
fig, axes = plt.subplots(3, 1, figsize=(15, 11))
views = [("Adjustable side: male buckle bar, sewn tail (box-X), slider", lay.s_m - 8, lay.xb + 30),
         ("Slider close-up (bars a, c, b)", lay.xa - 12, lay.xb + 12),
         ("Fixed side: tail end, stitch rows, D-ring bar, female buckle bar", lay.s_e - 22, lay.C + 8)]
off = lay.anorm * (T / 2 - P["vel_t"] / 2)
for ax, (title, x0, x1) in zip(axes, views):
    for _, pol in polys:
        ax.fill(*pol.exterior.xy, color="#cfd8d3", lw=0)
    v = lay.pts + off
    ax.plot(v[:, 0], v[:, 1], color="#2f9c86", lw=2.2, solid_capstyle="butt")
    for nm, h in hw.items():
        ax.fill(*h.exterior.xy, color="#d99a86", ec="#8a4f3f", lw=0.8)
    ax.axhline(0, color="#999", lw=0.6, ls="--")
    ax.set_xlim(x0, x1); ax.set_ylim(-3, 16); ax.set_aspect("equal"); ax.set_title(title, loc="left")
    ax.set_xlabel("distance around collar s (mm)"); ax.set_ylabel("h (mm)")
fig.suptitle("Strap threading, section view  (grey = webbing, teal line = velvet face, rose = metal, dashed = neck side)")
fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "threading_section.png"
fig.savefig(out, dpi=110)
print("saved", out)
sys.exit(1 if bad else 0)
