"""Pure-numpy geometry for the velvet ribbon dog collar (no Blender needed).

Everything is in millimetres.  The strap is designed in a flat "section" space
(s = distance around the collar, h = height above the neck-side plane) and then
wrapped onto a closed base curve that is flat wherever rigid hardware sits.

Strap construction (one continuous piece, 26 in / 660 mm cut length):
  cut end 1 -> tail sewn back (box-X) -> loop round the slider's centre bar
  -> inner layer to the male buckle bar -> round the bar -> outer layer back
  -> under/over/under the three slider bars -> all the way round the neck
  -> over the D-ring bar -> round the female buckle bar -> tail back under the
  D-ring bar on the inside -> cut end 2 (rows + box-X).
The velvet ribbon is sewn to face "A" of the webbing; face A is the left side
of the direction of travel along that path.
"""
import math
import numpy as np

# ----------------------------------------------------------------------------
# Parameters (mm)
# ----------------------------------------------------------------------------
P = dict(
    cut_length=660.0,        # 26 in webbing + ribbon (16-20 in size, starting point)
    circumference=410.0,     # neck setting modelled (about 16 in, tight end of the range)
    web_w=25.0, web_t=1.4,   # 1 in webbing
    vel_w=22.0, vel_t=0.8,   # 7/8 in velvet ribbon, centred on the webbing
    layer_gap=0.15,          # air between stacked layers
    bar_gap=0.25,            # air between strap and a metal bar
    bar_r=1.5,               # round bars: buckle bars and D-ring wire (3 mm)
    # buckle (local x along s, origin at the buckle centre)
    s_bar_m=16.0,            # male strap bar
    s_bar_f=-29.0,           # female strap bar
    d_ring_from_bar=17.0,    # D-ring bar sits this far behind the female bar
    fixed_tail=95.0,         # return tail behind the female bar (about 4 in pulled through)
    # slider (tri-glide)
    slider_pitch=10.5,       # bar centre to bar centre
    slider_bar_w=3.0, slider_bar_t=2.6,
    slider_tail=38.0,        # tail behind the slider's first bar (about 2 in in total)
)
T = P["web_t"] + P["vel_t"]          # full strap thickness
RHO = P["bar_r"] + P["bar_gap"] + T / 2   # centreline radius of a tight wrap
STEP = T + P["layer_gap"]            # centreline spacing of stacked layers
L1 = T / 2 + 0.1                     # inner layer level
L2 = L1 + STEP                       # second layer
L3 = L2 + STEP                       # third layer
H_SLIDER = L3 + RHO                  # slider bar centres
H_BUCKLE = L1 + RHO                  # buckle bar centres (= buckle mid-plane)
H_DRING = H_BUCKLE                   # D-ring bar centre (sandwiched between layers)


# ----------------------------------------------------------------------------
# Base curve: closed, flat at the buckle and at the slider
# ----------------------------------------------------------------------------
def _sstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


class BaseCurve:
    def __init__(self, C, s_slider, ds=0.25, ramp=8.0, hinge=34.0):
        self.C = C
        n = int(round(C / ds))
        self.s = np.linspace(0.0, C, n + 1)
        # flats (rigid hardware) and a soft "hinge" of extra bend at each flat end
        f1a, f1b = C - 34.0, 21.0                 # buckle flat (wraps through s=0)
        f2a, f2b = s_slider - 15.0, s_slider + 15.0
        s = self.s
        flat = lambda a, b: _sstep((s - a) / ramp + 1.0) * (1 - _sstep((s - b) / ramp))
        arc = 1.0 - flat(f2a, f2b) - (1 - _sstep((s - f1b) / ramp)) - _sstep((s - f1a) / ramp + 1.0)

        def bump(c0, sign):
            t = np.clip(sign * (s - c0) / hinge, 0.0, 1.0)
            return (1 - np.cos(2 * math.pi * t)) / hinge
        W = [arc, bump(f1b, 1) + bump(f1a, -1), bump(f2b, 1) + bump(f2a, -1)]

        def integrate(k):
            kap = sum(ki * wi for ki, wi in zip(k, W))
            th = np.concatenate([[0], np.cumsum(0.5 * (kap[1:] + kap[:-1]) * ds)])
            cx, cy = np.cos(th), np.sin(th)
            x = np.concatenate([[0], np.cumsum(0.5 * (cx[1:] + cx[:-1]) * ds)])
            y = np.concatenate([[0], np.cumsum(0.5 * (cy[1:] + cy[:-1]) * ds)])
            return th, x, y

        def resid(k):
            th, x, y = integrate(k)
            return np.array([(th[-1] - 2 * math.pi) * 50.0, x[-1], y[-1]])

        R = C / (2 * math.pi)
        k = np.array([1 / R, math.asin(27.5 / R), math.asin(15 / R)])
        for _ in range(40):
            r = resid(k)
            if np.abs(r).max() < 1e-6:
                break
            J = np.zeros((3, 3))
            for j in range(3):
                dk = np.zeros(3); dk[j] = 1e-6
                J[:, j] = (resid(k + dk) - r) / 1e-6
            k = k - np.linalg.solve(J, r)
        self.kappa = k
        th, x, y = integrate(k)
        # mirror X so that the fixed (D-ring) side ends up on world +X
        self.px, self.py = -x, y
        self.tx, self.ty = -np.cos(th), np.sin(th)

    def frame(self, s):
        s = np.mod(s, self.C)
        px = np.interp(s, self.s, self.px); py = np.interp(s, self.s, self.py)
        tx = np.interp(s, self.s, self.tx); ty = np.interp(s, self.s, self.ty)
        n = np.hypot(tx, ty)
        tx, ty = tx / n, ty / n
        return px, py, tx, ty, -ty, tx          # point, tangent, outward normal

    def world(self, s, h, w):
        """(s, h, w) -> world xyz (mm). w is across the strap (world Z)."""
        px, py, tx, ty, nx, ny = self.frame(s)
        return np.stack([px + h * nx, py + h * ny, np.broadcast_to(w, np.shape(px)) + 0.0], -1)

    def normal3(self, s, a_s, a_h):
        """Section-space vector (a_s, a_h) -> world direction."""
        px, py, tx, ty, nx, ny = self.frame(s)
        return np.stack([a_s * tx + a_h * nx, a_s * ty + a_h * ny, np.zeros_like(px)], -1)


# ----------------------------------------------------------------------------
# Belt path: tangent lines between circles ("pulleys"); r = 0 gives a point
# ----------------------------------------------------------------------------
def belt(elements, arc_step=0.2, line_step=0.6):
    """elements: list of (name, cx, cy, r, d) with d=+1 ccw (centre on the left), -1 cw."""
    tang = []
    for (_, x1, y1, r1, d1), (_, x2, y2, r2, d2) in zip(elements[:-1], elements[1:]):
        a, b = d1 * r1, d2 * r2
        v = np.array([x2 - x1, y2 - y1]); D = np.linalg.norm(v)
        c = (b - a) / D
        if abs(c) > 1:
            raise ValueError("pulleys overlap: cannot route strap")
        phi = math.acos(c); vh = v / D
        n = np.array([vh[0] * math.cos(phi) - vh[1] * math.sin(phi),
                      vh[0] * math.sin(phi) + vh[1] * math.cos(phi)])
        tang.append((np.array([x1, y1]) - a * n, np.array([x2, y2]) - b * n))
    pts, marks, ell = [], {}, 0.0

    def add(seg):
        nonlocal ell
        for p in seg:
            if pts:
                dl = np.linalg.norm(p - pts[-1])
                if dl < 1e-9:
                    continue
                ell += dl
            pts.append(p)

    for i, (name, cx, cy, r, d) in enumerate(elements):
        l_in = ell
        if 0 < i < len(elements) - 1 and r > 0:
            p_in, p_out = tang[i - 1][1], tang[i][0]
            a0 = math.atan2(p_in[1] - cy, p_in[0] - cx)
            a1 = math.atan2(p_out[1] - cy, p_out[0] - cx)
            sw = (a1 - a0) % (2 * math.pi) if d > 0 else -((a0 - a1) % (2 * math.pi))
            if abs(sw) > 2 * math.pi - 1e-6:
                sw = 0.0
            n = max(2, int(abs(sw) * r / arc_step) + 1)
            ang = a0 + sw * np.linspace(0, 1, n)
            add(np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], -1))
        elif i == 0:
            add([tang[0][0]])
        marks[name] = (l_in, ell)
        if i < len(elements) - 1:
            p0, p1 = tang[i]
            n = max(2, int(np.linalg.norm(p1 - p0) / line_step) + 1)
            add(p0 + (p1 - p0) * np.linspace(0, 1, n)[:, None])
            if i == len(elements) - 2:
                marks[elements[-1][0]] = (ell, ell)
    return np.array(pts), marks


class Layout:
    """Solves the slider position for the cut length and exposes the strap."""

    def __init__(self):
        C = P["circumference"]
        self.C = C
        self.s_m = P["s_bar_m"]
        self.s_f = C + P["s_bar_f"]
        self.s_d = self.s_f - P["d_ring_from_bar"]
        self.s_e = self.s_f - P["fixed_tail"]
        s_sl = self.s_m + 100.0
        for _ in range(12):
            self._build(s_sl)
            err = self.length - P["cut_length"]
            if abs(err) < 0.02:
                break
            s_sl -= err
        self.doubled = s_sl - self.s_m

    def _build(self, s_sl):
        self.s_sl = s_sl
        p = P["slider_pitch"]
        xa, xb = s_sl - p, s_sl + p
        self.xa, self.xb = xa, xb
        x0 = xa - P["slider_tail"]
        self.x_tail1 = x0
        hs, hm = H_SLIDER, H_BUCKLE
        sM, sF, sD, sE = self.s_m, self.s_f, self.s_d, self.s_e
        self.s_row_bd = sD + 8.5            # stitch row between buckle and D-ring
        self.s_rows_d = (sD - 9.0, sD - 12.5)   # rows behind the D-ring
        E = [
            ("end1", x0, L2, 0, 1),
            ("a_tail", xa, hs, RHO + STEP, +1),      # tail passes under bar a
            ("c_loop", s_sl, hs, RHO, -1),           # fixed loop round the centre bar
            ("a_l1", xa, hs, RHO + 2 * STEP, -1),    # inner layer leaves under bar a
            ("M", sM, hm, RHO, -1),                  # round the male buckle bar
            ("G1", sM + 9, L2 + 6, 6, +1),
            ("G2", x0 - 12, L2 + 5, 5, +1),          # step up over the sewn tail
            ("G3", x0 - 2, L3 - 5, 5, -1),
            ("a_l2", xa, hs, RHO, +1),               # under bar a
            ("c_l2", s_sl, hs, RHO + STEP, -1),      # over the centre bar (on top of the loop)
            ("b_l2", xb, hs, RHO, +1),               # under bar b
            ("G4", xb + 6, L3 - 5, 5, -1),
            ("G5", xb + 20, L1 + 6, 6, +1),
            ("G6", sE - 14, L1 + 5, 5, +1),          # step up over the fixed tail
            ("G7", sE - 2, L2 - 5, 5, -1),
            ("G8", sD - 9, L2 + 4, 4, +1),
            ("D", sD, H_DRING, RHO, -1),             # over the D-ring bar
            ("G9", self.s_row_bd, L2 + 3.5, 3.5, +1),
            ("F", sF, hm, RHO, -1),                  # round the female buckle bar
            ("end2", sE, L1, 0, 1),
        ]
        self.elements = E
        self.pts, self.marks = belt(E)
        self.base = BaseCurve(self.C, s_sl)
        d = np.gradient(self.pts, axis=0)
        d /= np.linalg.norm(d, axis=1)[:, None]
        self.tan = d
        self.anorm = np.stack([-d[:, 1], d[:, 0]], -1)     # face A = left of travel
        self.ell2d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(self.pts, axis=0), axis=1))])
        w3 = self.base.world(self.pts[:, 0], self.pts[:, 1], 0.0)
        self.ell = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(w3, axis=0), axis=1))])
        self.length = self.ell[-1]
        # marks were measured in section space: convert to true strap length
        self.marks = {k: (float(np.interp(a, self.ell2d, self.ell)), float(np.interp(b, self.ell2d, self.ell)))
                      for k, (a, b) in self.marks.items()}
        m = self.marks
        self.branches = dict(
            tail_slider=(0.0, m["a_tail"][0]),
            inner=(m["a_l1"][1], m["M"][0]),
            main_fixed=(m["G7"][1], m["G8"][1]),
            main_bd=(m["G9"][0], m["G9"][1]),
            tail_fixed=(m["F"][1], self.length),
        )

    # -- strap evaluation -----------------------------------------------------
    def section(self, ell):
        """strap length -> section point (s,h) and unit A-normal (a_s,a_h)."""
        s = np.interp(ell, self.ell, self.pts[:, 0]); h = np.interp(ell, self.ell, self.pts[:, 1])
        a_s = np.interp(ell, self.ell, self.anorm[:, 0]); a_h = np.interp(ell, self.ell, self.anorm[:, 1])
        n = np.hypot(a_s, a_h)
        return s, h, a_s / n, a_h / n

    def surface(self, ell, w, off):
        """world point at strap length ell, across-position w, offset off along the A-normal."""
        s, h, a_s, a_h = self.section(np.asarray(ell, float))
        return self.base.world(s + off * a_s, h + off * a_h, w)

    def surface_normal(self, ell):
        s, h, a_s, a_h = self.section(np.asarray(ell, float))
        return self.base.normal3(s, a_s, a_h)

    def ell_on(self, branch, s):
        a, b = self.branches[branch]
        m = (self.ell >= a) & (self.ell <= b)
        ss, ll = self.pts[m, 0], self.ell[m]
        o = np.argsort(ss)
        return float(np.interp(s, ss[o], ll[o]))


def rounded_rect(o0, o1, w0, w1, r, seg=3):
    """Closed profile in (offset, width), counter-clockwise."""
    out = []
    for cx, cy, a0 in ((o1 - r, w1 - r, 0), (o0 + r, w1 - r, 90), (o0 + r, w0 + r, 180), (o1 - r, w0 + r, 270)):
        for k in range(seg + 1):
            a = math.radians(a0 + 90 * k / seg)
            out.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return out
