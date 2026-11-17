"""Renders the step-by-step build animation of the velvet ribbon collar.

    python3 make_tutorial.py --check            # collision-check every animated frame (needs shapely)
    python3 make_tutorial.py --render           # render frames into tutorial/frames/ (resumes)
    python3 make_tutorial.py --render --shots 6,9 [--stride 4]
    python3 assemble_video.py                   # captions + mp4

The collar is assembled open and flat on a table.  Every frame is a valid strap
configuration: the strap is a fixed 660 mm long, and threading is shown either
by sliding the strap along its route (slider loop) or by moving a buckle bar
along the strap so that the strap rolls round it (both buckle halves).
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import collar_geometry as g  # noqa: E402

P, T, RHO, STEP, L1, L2, L3 = g.P, g.T, g.RHO, g.STEP, g.L1, g.L2, g.L3
HS, HM = g.H_SLIDER, g.H_BUCKLE
F = g.Layout(flat=True)                     # the finished collar, laid flat
L = P["cut_length"]
S, XA, XB, X0 = F.s_sl, F.xa, F.xb, F.x_tail1
FPS = 15
D_LOW = P["bar_r"] + 0.05                   # D-ring bar resting on the table
sstep = g._sstep


# =============================================================================
# strap configurations (section space: s along the table, h above it)
# =============================================================================
class Cfg:
    def __init__(self, poly, start=0.0):
        cl = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))])
        a, b = start, start + L
        if cl[-1] < b:
            raise ValueError("route shorter than the strap")
        keep = (cl > a + 1e-6) & (cl < b - 1e-6)
        ends = np.array([[np.interp(x, cl, poly[:, 0]), np.interp(x, cl, poly[:, 1])] for x in (a, b)])
        self.pts = np.vstack([ends[:1], poly[keep], ends[1:]])
        self.ell = np.concatenate([[0.0], cl[keep] - a, [L]])
        d = np.gradient(self.pts, axis=0)
        d /= np.linalg.norm(d, axis=1)[:, None]
        self.anorm = np.stack([-d[:, 1], d[:, 0]], -1)
        self.roll = 0.0                      # rotation about the strap's long axis (straight strap only)
        self.tip = self.pts[-1]

    def section(self, ell):
        s = np.interp(ell, self.ell, self.pts[:, 0]); h = np.interp(ell, self.ell, self.pts[:, 1])
        a_s = np.interp(ell, self.ell, self.anorm[:, 0]); a_h = np.interp(ell, self.ell, self.anorm[:, 1])
        n = np.hypot(a_s, a_h)
        return s, h, a_s / n, a_h / n

    def _roll(self, y, z, pivot=True):
        if not self.roll:
            return y, z
        c, sn = math.cos(self.roll), math.sin(self.roll)
        z0 = L1 if pivot else 0.0
        y2 = c * y - sn * (z - z0); z2 = sn * y + c * (z - z0) + z0
        return y2, z2 + (14.0 * abs(sn) if pivot else 0.0)

    def surface(self, ell, w, off):
        s, h, a_s, a_h = self.section(np.asarray(ell, float))
        x = s + off * a_s; z = h + off * a_h; y = -np.broadcast_to(w, x.shape) + 0.0
        y, z = self._roll(y, z)
        return np.stack([x, y, z], -1)

    def normal(self, ell):
        s, h, a_s, a_h = self.section(np.asarray(ell, float))
        y, z = self._roll(np.zeros_like(a_s), a_h, pivot=False)
        return np.stack([a_s, y, z], -1)


def slider_part(u):
    """Fixed loop round the slider's centre bar. u = 0: slider rests on tail + inner layer; 1: final height."""
    hs = HS - STEP * (1 - u)
    return [("end1", X0, L2, 0, 1),
            ("a_tail", XA, hs, RHO + STEP * u, +1),
            ("c_loop", S, hs, RHO, -1),
            ("a_l1", XA, hs, RHO + STEP * (1 + u), -1)], hs


def cfg_slide(delta):
    """Slider loop: the strap slides along its route; the head (cut end 1) is `delta` short of its final place."""
    E, _ = slider_part(0.0)
    poly, _ = g.belt(E + [("far", -2500.0, L1, 0, 1)])
    return Cfg(poly, delta)


_E0, _ = slider_part(0.0)
_, _m0 = g.belt(_E0 + [("far", -2500.0, L1, 0, 1)])
SLIDE_LEN = _m0["a_l1"][1]                  # route length from the tail end to where the strap leaves bar a


def male_route(b, u):
    E, hs = slider_part(u)
    l3 = hs - RHO
    E += [("M", b, HM, RHO, -1), ("G1", b + 9, L2 + 6, 6, +1),
          ("G2", X0 - 12, L2 + 5, 5, +1), ("G3", X0 - 2, l3 - 5, 5, -1),
          ("a_l2", XA, hs, RHO, +1), ("c_l2", S, hs, RHO + STEP, -1), ("b_l2", XB, hs, RHO, +1),
          ("G4", XB + 6, l3 - 5, 5, -1), ("G5", XB + 20, L1 + 6, 6, +1)]
    return E


def cfg_male(b, u):
    """Male buckle bar at s = b; the strap rolls round it and the outer layer follows the final route."""
    poly, marks = g.belt(male_route(b, u) + [("far", 3000.0, L1, 0, 1)])
    c = Cfg(poly)
    c.marks = marks
    return c


def fixed_elems(d, b, q):
    """D-ring bar at d, female buckle bar at b, tail length q behind the bar (drives the lifts)."""
    lam = L1 + STEP * sstep((q - 19) / 4.0)          # main strap level in front of the D-ring
    mu = L1 + STEP * sstep((q - 2) / 4.0)            # level at the row between D-ring and buckle
    hd = D_LOW + (g.H_DRING - D_LOW) * sstep((q - 6) / 6.0)
    x7 = d + 17.0 - max(q + 2.0, 28.0)
    E = male_route(F.s_m, 1.0)
    E += [("G6", x7 - 12, L1 + 5, 5, +1), ("G7", x7, lam - 5, 5, -1),
          ("G8", d - 9, lam + 4, 4, +1), ("D", d, hd, RHO, -1), ("G9", d + 8.5, mu + 3.5, 3.5, +1)]
    return E, hd


def cfg_ring(d):
    """D-ring slid along the free end to position d (the strap humps over its bar)."""
    E, hd = fixed_elems(d, d + 17, 0.0)
    poly, _ = g.belt(E + [("far", 3000.0, L1, 0, 1)])
    c = Cfg(poly); c.hd = hd
    return c


def cfg_female(b, d=None, q=None):
    """Female buckle bar at b. With q given, b is solved so the tail behind the bar is q long."""
    if q is None:
        E, hd = fixed_elems(d, b, 0.0)
        poly, _ = g.belt(E + [("F", b, HM, RHO, -1), ("far", -3000.0, L1, 0, 1)])
        c = Cfg(poly); c.hd, c.b, c.d = hd, b, d
        return c
    for _ in range(12):
        E, hd = fixed_elems(b - 17, b, q)
        poly, marks = g.belt(E + [("F", b, HM, RHO, -1), ("far", -3000.0, L1, 0, 1)])
        err = marks["F"][1] + q - L
        if abs(err) < 1e-4:
            break
        b -= err
    c = Cfg(poly); c.hd, c.b, c.d = hd, b, b - 17
    return c


# ---- key positions ----------------------------------------------------------
def solve(fn, lo, hi, target):
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if (fn(mid) < target) == (fn(lo) < target):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


DELTA0 = SLIDE_LEN + 30.0                                  # straight strap, head 30 mm short of the slider
HEAD_X = cfg_slide(DELTA0).pts[0, 0]
B_TIP = solve(lambda b: cfg_male(b, 0).marks["M"][0], -700.0, 0.0, L)      # bar exactly under the cut end
B_PRE = solve(lambda b: cfg_male(b, 0).tip[0], B_TIP + 5, 0.0, X0 - 40)    # end 40 mm short of the sewn tail
B_POST = solve(lambda b: cfg_male(b, 1).tip[0], B_PRE, F.s_m, XB + 60)     # end 60 mm past the slider
TIP_FREE = cfg_male(F.s_m, 1).tip[0]                       # free end after the adjustable side is done
Q_END = L - F.marks["F"][1]
B_F0 = cfg_female(TIP_FREE, q=0.0).b                       # female bar exactly under the cut end
D_START = B_F0 - 17.0


# =============================================================================
# shots
# =============================================================================
def ease(t):
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


PARK = dict(slider=(HEAD_X - 250, -62), male=(HEAD_X - 310, -62), female=(HEAD_X - 380, -62), dring=(HEAD_X - 440, -60))


def base_state():
    return dict(cfg=None, roll=0.0, ribbon=(0.0, 0.0), ribbon_on=True, edge=1.0, boxA=0.0, fixed=0.0,
                slider=None, male=None, female=None, dring=None, needle=None)


def st_straight(t, roll=math.pi, ribbon=(0.0, 0.0), edge=0.0):
    s = base_state(); s["cfg"] = cfg_slide(DELTA0); s["roll"] = roll; s["ribbon"] = ribbon; s["edge"] = edge
    return s


def shot_supplies(t):
    return st_straight(t, ribbon=(46.0, -P["web_t"]))


def shot_ribbon(t):
    e = ease(t)
    return st_straight(t, ribbon=(46.0 * (1 - e), -P["web_t"] * (1 - e) + 9.0 * math.sin(math.pi * e)))


def shot_edge(t):
    s = st_straight(t, edge=t); s["needle"] = "edge"
    return s


def shot_flip(t):
    return st_straight(t, roll=math.pi * (1 - ease(t)), edge=1.0)


def shot_slider(t):
    s = base_state(); s["cfg"] = cfg_slide(DELTA0 * (1 - ease(t))); s["slider"] = 0.0
    s["slider_h"] = lerp(P["slider_bar_t"] / 2, HS - STEP, ease(min(1.0, t / 0.15)))   # picked up off the table
    return s


def shot_boxA(t):
    s = base_state(); s["cfg"] = cfg_slide(0.0); s["slider"] = 0.0; s["boxA"] = t; s["needle"] = "boxA"
    return s


def _male(b, u):
    s = base_state(); s["cfg"] = cfg_male(b, u); s["slider"] = u; s["boxA"] = 1.0; s["male"] = b
    return s


def shot_male_a(t):
    return _male(lerp(B_TIP - 30, B_TIP + 40, ease(t)), 0.0)


def shot_male_b(t):
    return _male(lerp(B_TIP + 40, B_PRE, ease(t)), 0.0)


def shot_thread(t):
    u = ease(min(1.0, t / 0.16))                       # the slider is lifted before the end reaches it
    tt = max(0.0, (t - 0.16) / 0.84)
    return _male(lerp(B_PRE, B_POST, tt), u)


def shot_pull(t):
    return _male(lerp(B_POST, F.s_m, ease(t)), 1.0)


def _fixed_common(s):
    s["slider"] = 1.0; s["boxA"] = 1.0; s["male"] = F.s_m
    return s


def shot_dring(t):
    d = lerp(TIP_FREE + 38, D_START, ease(t))
    s = base_state(); s["cfg"] = cfg_ring(d); s["dring"] = (d, s["cfg"].hd)
    return _fixed_common(s)


def shot_female(t):
    s = base_state()
    if t < 0.2:
        c = cfg_female(lerp(B_F0 + 26, B_F0, ease(t / 0.2)), d=D_START)
    else:
        c = cfg_female(B_F0, q=Q_END * ease((t - 0.2) / 0.8))
    s["cfg"] = c; s["dring"] = (c.d, c.hd); s["female"] = c.b
    return _fixed_common(s)


def shot_sew_fixed(t):
    c = cfg_female(F.s_f, q=Q_END)
    s = base_state(); s["cfg"] = c; s["dring"] = (c.d, c.hd); s["female"] = c.b; s["fixed"] = t; s["needle"] = "fixed"
    return _fixed_common(s)


def shot_done(t):
    s = shot_sew_fixed(1.0); s["needle"] = None
    return s


def cam(tx, ty, tz, dist, el, az, lens=50):
    return dict(t=(tx, ty, tz), dist=dist, el=el, az=az, lens=lens)


MID = HEAD_X - 330
# name, state function, frames, camera (start, end or None)
SHOTS = [
    ("01_supplies", shot_supplies, 1, cam(MID, -10, 0, 1050, 52, 0), None),
    ("02_cut", shot_supplies, 1, cam(HEAD_X - 45, 18, 0, 300, 42, -24), None),
    ("03_ribbon", shot_ribbon, 20, cam(HEAD_X - 95, 12, 0, 430, 44, -22), None),
    ("04_edge", shot_edge, 45, cam(HEAD_X - 75, 0, 0, 330, 55, -18), None),
    ("05_flip", shot_flip, 16, cam(HEAD_X - 130, 0, 6, 500, 36, -20), None),
    ("06_slider", shot_slider, 45, cam(S - 10, 0, 6, 175, 17, 16), None),
    ("07_boxx", shot_boxA, 30, cam(X0 + 16, 0, 3, 165, 52, 12), None),
    ("08_male", shot_male_a, 28, cam(B_TIP - 12, 0, 4, 215, 27, 22), cam(B_TIP + 48, 0, 4, 215, 27, 22)),
    ("09_fold", shot_male_b, 30, cam(B_TIP + 110, 0, 4, 420, 30, 18), cam(B_PRE + 150, 0, 4, 520, 30, 18)),
    ("10_thread", shot_thread, 52, cam(S - 2, 0, 7, 195, 17, 16), None),
    ("11_pull", shot_pull, 24, cam((F.s_m + S) / 2 + 20, 0, 5, 400, 32, 14), None),
    ("12_dring", shot_dring, 26, cam(D_START + 28, 0, 5, 250, 30, -20), None),
    ("13_female", shot_female, 48, cam(B_F0 - 5, 0, 5, 230, 19, -16), cam(F.s_f + 2, 0, 5, 215, 19, -16)),
    ("14_sew", shot_sew_fixed, 40, cam(F.s_f - 42, 0, 3, 235, 56, -12), None),
    ("15_flat", shot_done, 1, cam((F.s_m + F.s_f) / 2, 0, 0, 760, 40, 0), None),
]


def shot_state(i, k):
    name, fn, n, c0, c1 = SHOTS[i]
    t = 0.0 if n == 1 else k / (n - 1)
    st = fn(t)
    st["cfg"].roll = st["roll"]
    c = dict(c0)
    if c1:
        e = ease(t)
        c = dict(t=tuple(lerp(a, b, e) for a, b in zip(c0["t"], c1["t"])), dist=lerp(c0["dist"], c1["dist"], e),
                 el=lerp(c0["el"], c1["el"], e), az=lerp(c0["az"], c1["az"], e), lens=c0["lens"])
    st["cam"] = c
    return st


# =============================================================================
# 2D collision check of every frame
# =============================================================================
def check():
    from shapely.geometry import LineString, Point, box
    bw, bt = P["slider_bar_w"] / 2, P["slider_bar_t"] / 2
    worst = 0.0; bad = 0; total = 0
    for i, (name, fn, n, _, _) in enumerate(SHOTS):
        shot_bad = 0
        for k in range(n):
            st = shot_state(i, k); c = st["cfg"]; total += 1
            hw = {}
            if st["slider"] is not None:
                hs = st.get("slider_h", HS - STEP * (1 - st["slider"]))
                for nm, x in (("slider_a", XA), ("slider_c", S), ("slider_b", XB)):
                    hw[nm] = box(x - bw, hs - bt, x + bw, hs + bt).buffer(-0.6).buffer(0.6)
            if st["male"] is not None:
                b = st["male"]
                hw["bar_M"] = Point(b, HM).buffer(P["bar_r"]); hw["male_body"] = box(b - 11.85, HM - 3.85, b - 5, HM + 3.85)
            if st["female"] is not None:
                b = st["female"]
                hw["bar_F"] = Point(b, HM).buffer(P["bar_r"]); hw["female_body"] = box(b + 5, HM - 3.85, b + 33, HM + 3.85)
            if st["dring"] is not None:
                hw["bar_D"] = Point(st["dring"][0], st["dring"][1]).buffer(P["bar_r"])
            hw["table"] = box(-3000, -50, 3000, -0.001)
            edges = np.arange(0, L + 2.0, 2.0); polys = []
            for a, b2 in zip(edges[:-1], edges[1:]):
                m = (c.ell >= a - 1e-6) & (c.ell <= b2 + 1e-6)
                if m.sum() > 1:
                    polys.append(((a + b2) / 2, LineString(c.pts[m]).buffer(T / 2 - 0.02, cap_style="flat")))
            hits = []
            for j, (li, pi) in enumerate(polys):
                for nm, h in hw.items():
                    a = pi.intersection(h).area if pi.intersects(h) else 0
                    if a > 1e-3:
                        hits.append((a, f"strap@{li:.0f} x {nm}"))
                for lj, pj in polys[j + 1:]:
                    if lj - li > 8 and pi.intersects(pj):
                        a = pi.intersection(pj).area
                        if a > 1e-3:
                            hits.append((a, f"strap@{li:.0f} x strap@{lj:.0f}"))
            if hits:
                shot_bad += 1; bad += 1
                a, what = max(hits); worst = max(worst, a)
                if shot_bad <= 3:
                    print(f"  {name} frame {k}: {len(hits)} overlaps, worst {a:.3f} mm2 ({what})")
        print(f"{name}: {n} frames, {shot_bad} with collisions")
    print(f"frames checked {total}, with collisions {bad}, worst overlap {worst:.3f} mm2")
    return bad


# =============================================================================
# Blender side
# =============================================================================
def render(argv):
    import bpy
    from mathutils import Matrix, Vector
    import build_collar as B
    MM = B.MM
    out_dir = os.path.join(HERE, "tutorial", "frames")
    os.makedirs(out_dir, exist_ok=True)
    only = argv[argv.index("--shots") + 1].split(",") if "--shots" in argv else None
    stride = int(argv[argv.index("--stride") + 1]) if "--stride" in argv else 1
    samples = int(argv[argv.index("--samples") + 1]) if "--samples" in argv else 14

    sc = B.reset_scene()
    velvet = B.mat_velvet("Velvet_Ribbon", B.VELVET_RGB)
    # the strap deforms, so drive the velvet texture from strap UVs instead of object space
    nt = velvet.node_tree
    tc = next(n for n in nt.nodes if n.type == "TEX_COORD")
    for n in nt.nodes:
        if n.type == "TEX_NOISE":
            n.inputs["Scale"].default_value *= 0.1
            nt.links.new(tc.outputs["UV"], n.inputs["Vector"])
    mats = dict(velvet=velvet, web=B.mat_webbing("Webbing", B.WEB_RGB),
                thread=B.mat_simple("Thread", (0.86, 0.90, 0.84), rough=0.6, sheen=0.5),   # pale thread so the seams read on video
                metal=B.mat_simple("Rose_Gold", B.METAL_RGB, rough=0.22, metallic=1.0),
                steel=B.mat_simple("Steel", (0.75, 0.76, 0.78), rough=0.25, metallic=1.0))
    c_strap, c_hw, c_studio = (B.collection(n) for n in ("Collar_Strap", "Collar_Hardware", "Studio"))

    # ---- table with a 10 mm grid, lights, camera ---------------------------
    me = bpy.data.meshes.new("Table")
    me.from_pydata([(-4, -2, 0), (4, -2, 0), (4, 2, 0), (-4, 2, 0)], [], [(0, 1, 2, 3)])
    table = bpy.data.objects.new("Table", me); c_studio.objects.link(table)
    tm, tnt, tb = B.principled("Cutting_Mat")
    B.setv(tb, "Roughness", 0.6)
    tcn = tnt.nodes.new("ShaderNodeTexCoord")
    grid = tnt.nodes.new("ShaderNodeTexBrick"); grid.offset = 0.0
    grid.inputs["Scale"].default_value = 100.0; grid.inputs["Mortar Size"].default_value = 0.012
    grid.inputs["Brick Width"].default_value = 1.0; grid.inputs["Row Height"].default_value = 1.0
    grid.inputs["Color1"].default_value = grid.inputs["Color2"].default_value = (0.50, 0.49, 0.47, 1)
    grid.inputs["Mortar"].default_value = (0.40, 0.39, 0.38, 1)
    tnt.links.new(tcn.outputs["Object"], grid.inputs["Vector"]); tnt.links.new(grid.outputs["Color"], tb.inputs["Base Color"])
    me.materials.append(tm)
    world = bpy.data.worlds.new("Studio"); sc.world = world; world.use_nodes = True
    bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs[0].default_value = (1, 1, 1, 1); bg.inputs[1].default_value = 0.38
    lights = []
    for nm, off, size, power in (("Key", (-0.25, 0.30, 0.55), 0.5, 2.6), ("Fill", (0.40, 0.35, 0.30), 0.6, 0.6),
                                 ("Rim", (0.05, -0.45, 0.35), 0.5, 1.0)):
        ld = bpy.data.lights.new(nm, "AREA"); ld.size = size; ld.energy = power
        lo = bpy.data.objects.new("Light_" + nm, ld); c_studio.objects.link(lo); lights.append((lo, Vector(off)))
    cd = bpy.data.cameras.new("Cam"); cd.clip_start = 0.01; cd.clip_end = 20
    camo = bpy.data.objects.new("Cam", cd); c_studio.objects.link(camo); sc.camera = camo

    # ---- hardware (built once, moved every frame) ---------------------------
    slider = B.build_slider(c_hw, mats["metal"])
    dring = B.build_d_ring(c_hw, mats["metal"])
    fem, male = B.build_buckle(c_hw, mats["metal"])
    needle = B.solid("Needle", lambda bm: B.bm_cyl_z(bm, 0, 0, 0.45, 0.0, 36.0, 12), c_studio)
    needle.data.materials.append(mats["steel"])
    R = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))      # local x->X, y->Z, z->-Y

    def place(ob, x, y, z, rot=None):
        m = Matrix.Translation((x * MM, y * MM, z * MM)) @ R
        ob.matrix_world = m @ rot if rot else m

    webbing = bpy.data.objects.new("Webbing_25mm", bpy.data.meshes.new("w")); c_strap.objects.link(webbing)
    ribbon = bpy.data.objects.new("Velvet_Ribbon_22mm", bpy.data.meshes.new("v")); c_strap.objects.link(ribbon)
    thread = bpy.data.objects.new("Stitching", bpy.data.meshes.new("s")); c_strap.objects.link(thread)

    def set_mesh(ob, verts, faces, mat, uv=None, sharp=40.0):
        me = bpy.data.meshes.new(ob.name)
        me.from_pydata((np.asarray(verts) * MM).tolist(), [], faces)
        if uv is not None:
            layer = me.uv_layers.new(name="UVMap")
            idx = np.zeros(len(me.loops), dtype=np.int32); me.loops.foreach_get("vertex_index", idx)
            layer.data.foreach_set("uv", np.asarray(uv, np.float32)[idx].ravel())
        B.finish_mesh(me, sharp)
        me.materials.append(mat)
        old = ob.data; ob.data = me; bpy.data.meshes.remove(old)

    def sweep(c, l0, l1, o0, o1, w0, w1, rad, dy=0.0, dz=0.0):
        ells = c.ell[(c.ell > l0 + 1e-6) & (c.ell < l1 - 1e-6)]
        ells = np.concatenate([[l0], ells, [l1]])
        prof = g.rounded_rect(o0, o1, w0, w1, rad); K = len(prof); n = len(ells)
        verts = np.concatenate([c.surface(ells, w, o)[:, None, :] for o, w in prof], axis=1).reshape(-1, 3)
        verts = verts + np.array([0.0, dy, dz])
        uv = np.concatenate([np.stack([ells / 100.0, np.full_like(ells, w / 100.0)], -1)[:, None, :] for o, w in prof], axis=1).reshape(-1, 2)
        faces = [(i * K + k, i * K + (k + 1) % K, (i + 1) * K + (k + 1) % K, (i + 1) * K + k)
                 for i in range(n - 1) for k in range(K)]
        faces.append(tuple(range(K))[::-1]); faces.append(tuple((n - 1) * K + k for k in range(K)))
        return verts, faces, uv

    # ---- stitches in strap coordinates (l0, w0, l1, w1, face), grouped ------
    def rows(l0, w0, l1, w1, face, pitch=3.0, gap=0.45):
        n = max(1, int(round(math.hypot(l1 - l0, w1 - w0) / pitch))); out = []
        for i in range(n):
            a = (i + gap / pitch / 2) / n; b = (i + 1 - gap / pitch / 2) / n
            out.append((l0 + (l1 - l0) * a, w0 + (w1 - w0) * a, l0 + (l1 - l0) * b, w0 + (w1 - w0) * b, face))
        return out

    def box_x(branches, s0, s1, hw=8.5):
        out = []
        for i in range(6):
            for br in branches:
                a, b = F.ell_on(br, s0), F.ell_on(br, s1)
                q = ((a, -hw, b, -hw), (b, -hw, b, hw), (b, hw, a, hw), (a, hw, a, -hw), (a, -hw, b, hw), (a, hw, b, -hw))[i]
                out.append(rows(*q, +1))
        # interleave the two layers so both faces are sewn together
        return [s for pair in zip(*[out[i::len(branches)] for i in range(len(branches))]) for grp in zip(*pair) for s in grp] \
            if len(set(map(len, out))) == 1 else [s for grp in out for s in grp]

    e = P["vel_w"] / 2 - 1.5
    edge = []
    for seg in ((4.0, -e, L - 4.0, -e), (L - 4.0, -e, L - 4.0, e), (L - 4.0, e, 4.0, e), (4.0, e, 4.0, -e)):
        top, bot = rows(*seg, +1), rows(*seg, -1)
        edge += [s for pair in zip(top, bot) for s in pair]
    groups = dict(
        edge=edge,
        boxA=box_x(("tail_slider", "inner"), X0 + 5.0, X0 + 22.0),
        fixed=[s for sx in (F.s_row_bd,) for br in ("main_bd", "tail_fixed") for s in rows(F.ell_on(br, sx), -10, F.ell_on(br, sx), 10, +1)]
        + [s for sx in F.s_rows_d for br in ("main_fixed", "tail_fixed") for s in rows(F.ell_on(br, sx), -10, F.ell_on(br, sx), 10, +1)]
        + box_x(("main_fixed", "tail_fixed"), F.s_e + 5.0, F.s_e + 22.0),
    )
    groups = {k: np.array(v) for k, v in groups.items()}
    prof_h = np.array([-0.30, -0.10, -0.08, -0.10, -0.30]); tt = np.linspace(0, 1, 5)
    ang = np.linspace(0, 2 * math.pi, 6, endpoint=False)

    def stitch_mesh(c, st):
        sel = []
        for k in ("edge", "boxA", "fixed"):
            n = int(round(st[k] * len(groups[k])))
            if n:
                sel.append(groups[k][:n])
        if not sel:
            return np.zeros((0, 3)), [], None
        a = np.concatenate(sel); N = len(a)
        ell = a[:, 0:1] + (a[:, 2:3] - a[:, 0:1]) * tt; w = a[:, 1:2] + (a[:, 3:4] - a[:, 1:2]) * tt
        face = a[:, 4:5]
        off = face * (T / 2 + prof_h)
        pts = c.surface(ell.ravel(), w.ravel(), off.ravel()).reshape(N, 5, 3)
        nrm = c.normal(ell.ravel()).reshape(N, 5, 3) * face[:, :, None]
        tan = np.gradient(pts, axis=1); tan /= np.linalg.norm(tan, axis=2)[:, :, None]
        bi = np.cross(tan, nrm); bi /= np.linalg.norm(bi, axis=2)[:, :, None]
        ring = np.cos(ang)[None, None, :, None] * nrm[:, :, None, :] + np.sin(ang)[None, None, :, None] * bi[:, :, None, :]
        verts = (pts[:, :, None, :] + 0.19 * ring).reshape(-1, 3)
        f1 = [(i * 6 + k, i * 6 + (k + 1) % 6, (i + 1) * 6 + (k + 1) % 6, (i + 1) * 6 + k) for i in range(4) for k in range(6)]
        f1 += [tuple(range(6))[::-1], tuple(24 + k for k in range(6))]
        faces = [tuple(v + 30 * j for v in f) for j in range(N) for f in f1]
        return verts, faces, pts[-1, 2]

    def apply(st, frame_k):
        c = st["cfg"]
        hw_, hv = P["web_w"] / 2, P["vel_w"] / 2
        v, f, uv = sweep(c, 0.0, L, -T / 2, -T / 2 + P["web_t"], -hw_, hw_, 0.35)
        set_mesh(webbing, v, f, mats["web"], uv)
        dy, dz = st["ribbon"]
        v, f, uv = sweep(c, 2.0, L - 2.0, T / 2 - P["vel_t"] + 0.01, T / 2, -hv, hv, 0.3, dy, dz)
        set_mesh(ribbon, v, f, mats["velvet"], uv)
        v, f, last = stitch_mesh(c, st)
        set_mesh(thread, v if len(v) else np.zeros((3, 3)), f, mats["thread"], sharp=80)
        # hardware
        if st["slider"] is None:
            place(slider, PARK["slider"][0], PARK["slider"][1], 1.3)
        else:
            place(slider, S, 0, st.get("slider_h", HS - STEP * (1 - st["slider"])))
        if st["male"] is None:
            place(male, PARK["male"][0], PARK["male"][1], 3.85)
        else:
            place(male, st["male"] - P["s_bar_m"], 0, HM)
        if st["female"] is None:
            place(fem, PARK["female"][0], PARK["female"][1], 3.85)
        else:
            place(fem, st["female"] - P["s_bar_f"], 0, HM)
        if st["dring"] is None:
            place(dring, PARK["dring"][0], PARK["dring"][1], 1.5, Matrix.Rotation(math.radians(90), 4, "Z"))
        else:
            place(dring, st["dring"][0], 0, st["dring"][1], Matrix.Rotation(math.radians(78), 4, "Z"))
        if st["needle"] and last is not None and 0 < st[st["needle"]] < 1:
            needle.matrix_world = Matrix.Translation((last[0] * MM, last[1] * MM, (last[2] + (0.5 if frame_k % 2 else 5.0)) * MM))
            needle.hide_render = False
        else:
            needle.hide_render = True
        # camera + lights follow the target
        cm = st["cam"]; tgt = Vector(cm["t"]) * MM
        el, az = math.radians(cm["el"]), math.radians(cm["az"])
        camo.location = tgt + cm["dist"] * MM * Vector((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)))
        camo.rotation_euler = (tgt - camo.location).to_track_quat("-Z", "Y").to_euler()
        cd.lens = cm["lens"]
        for lo, off in lights:
            lo.location = tgt + off
            lo.rotation_euler = (tgt - lo.location).to_track_quat("-Z", "Y").to_euler()

    B.setup_render(samples)
    sc.render.resolution_x, sc.render.resolution_y = 1280, 720
    sc.render.image_settings.file_format = "JPEG"; sc.render.image_settings.quality = 95
    for i, (name, fn, n, _, _) in enumerate(SHOTS):
        if only and name[:2] not in only:
            continue
        for k in list(range(0, n, stride)) + ([n - 1] if (n - 1) % stride else []):
            path = os.path.join(out_dir, f"{name}_{k:03d}.jpg")
            if os.path.exists(path):
                continue
            apply(shot_state(i, k), k)
            bpy.context.view_layer.update()
            sc.render.filepath = path
            bpy.ops.render.render(write_still=True)
            print("FRAME", name, k, flush=True)
    if "--save" in argv:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "tutorial", "tutorial_last_frame.blend"), compress=True)


def render_final(argv):
    """Turntable of the finished, buckled collar (from velvet_ribbon_dog_collar.blend)."""
    import bpy
    from mathutils import Vector
    samples = int(argv[argv.index("--samples") + 1]) if "--samples" in argv else 14
    out_dir = os.path.join(HERE, "tutorial", "frames"); os.makedirs(out_dir, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "velvet_ribbon_dog_collar.blend"))
    sc = bpy.context.scene; camo = bpy.data.objects["Cam_Hero"]; sc.camera = camo
    sc.cycles.samples = samples; sc.render.resolution_x, sc.render.resolution_y = 1280, 720
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "JPEG"; sc.render.image_settings.quality = 95
    tgt = Vector((0.0, -0.005, 0.012)); n = 48
    for k in range(n):
        path = os.path.join(out_dir, f"16_final_{k:03d}.jpg")
        if os.path.exists(path):
            continue
        az = math.radians(lerp(-38.0, 38.0, ease(k / (n - 1)))); el = math.radians(36.0)
        camo.location = tgt + 0.86 * Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
        camo.rotation_euler = (tgt - camo.location).to_track_quat("-Z", "Y").to_euler()
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        print("FRAME 16_final", k, flush=True)


if __name__ == "__main__":
    argv = sys.argv[1:]
    print(f"key positions: head {HEAD_X:.1f}, slider {S:.1f}, male bar tip {B_TIP:.1f} -> {F.s_m:.1f}, "
          f"free end {TIP_FREE:.1f}, female bar {B_F0:.1f} -> {F.s_f:.1f}, tail {Q_END:.1f}")
    if "--check" in argv:
        sys.exit(1 if check() else 0)
    if "--render" in argv:
        render(argv)
    if "--final" in argv:
        render_final(argv)
