"""Builds the velvet ribbon dog collar scene in Blender.

Run either way:
    blender -b -P build_collar.py -- [--render] [--samples 96]
    python3 build_collar.py [--render]          (with the `bpy` pip module)

Output: velvet_ribbon_dog_collar.blend (+ renders/*.png with --render).
Scene units are metres (real-world scale); all design numbers below are mm.
"""
import math
import os
import sys

import numpy as np
import bpy
import bmesh
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import collar_geometry as g  # noqa: E402

P, T = g.P, g.T
MM = 0.001
ENGRAVING = "Collarly"                 # text engraved on the buckle face
Z_STRAP = 16.5                         # strap centre height: collar stands on its hardware
VELVET_RGB = (0.115, 0.44, 0.36)
WEB_RGB = (0.16, 0.50, 0.40)
THREAD_RGB = (0.10, 0.36, 0.30)
METAL_RGB = (0.94, 0.60, 0.50)         # rose gold

lay = g.Layout()


# =============================================================================
# helpers
# =============================================================================
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.unit_settings.length_unit = "MILLIMETERS"
    return sc


def collection(name):
    c = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
    return c


def finish_mesh(me, sharp_deg=40.0):
    bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me); bm.free()
    for p in me.polygons:
        p.use_smooth = True
    try:
        me.set_sharp_from_angle(angle=math.radians(sharp_deg))
    except Exception:
        pass
    me.update()


def mesh_object(name, verts_mm, faces, coll, mat=None, uv=None, sharp_deg=40.0, world_mm=None):
    me = bpy.data.meshes.new(name)
    v = np.asarray(verts_mm, float) * MM
    me.from_pydata(v.tolist(), [], faces)
    if uv is not None:
        layer = me.uv_layers.new(name="UVMap")
        uv = np.asarray(uv, float)
        for poly in me.polygons:
            for li in poly.loop_indices:
                layer.data[li].uv = uv[me.loops[li].vertex_index]
    finish_mesh(me, sharp_deg)
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    if mat:
        me.materials.append(mat)
    if world_mm is not None:
        ob.matrix_world = world_mm
    return ob


def apply_modifiers(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    old = ob.data
    ob.modifiers.clear()
    ob.data = me
    me.name = ob.name
    bpy.data.meshes.remove(old)


def frame_matrix(s, h):
    """Rigid frame on the collar: x = along strap (+s), y = outward, z = up. Translation in metres."""
    px, py, tx, ty, nx, ny = lay.base.frame(np.array([s]))
    o = Vector(((px[0] + h * nx[0]) * MM, (py[0] + h * ny[0]) * MM, Z_STRAP * MM))
    m = Matrix(((tx[0], nx[0], 0, o.x), (ty[0], ny[0], 0, o.y), (0, 0, 1, o.z), (0, 0, 0, 1)))
    return m


# ---- bmesh primitives (mm, local) -------------------------------------------
def bm_box(bm, x0, x1, y0, y1, z0, z1):
    vs = [bm.verts.new((x, y, z)) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    for idx in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
        bm.faces.new([vs[i] for i in idx])


def bm_cyl_z(bm, cx, cy, r, z0, z1, n=40):
    lo = [bm.verts.new((cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n), z0)) for i in range(n)]
    hi = [bm.verts.new((v.co.x, v.co.y, z1)) for v in lo]
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((lo[i], lo[j], hi[j], hi[i]))
    bm.faces.new(lo[::-1]); bm.faces.new(hi)


def bm_prism_y(bm, poly_xz, y0, y1):
    lo = [bm.verts.new((x, y0, z)) for x, z in poly_xz]
    hi = [bm.verts.new((x, y1, z)) for x, z in poly_xz]
    n = len(lo)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((lo[i], lo[j], hi[j], hi[i]))
    bm.faces.new(lo[::-1]); bm.faces.new(hi)


def solid(name, build, coll):
    """Object from a bmesh builder working in mm."""
    bm = bmesh.new(); build(bm)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.scale(bm, vec=(MM, MM, MM), verts=bm.verts)
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me); coll.objects.link(ob)
    return ob


def boolean(target, cutters, op):
    for c in cutters:
        m = target.modifiers.new("bool", "BOOLEAN")
        m.operation = op; m.object = c
        try:
            m.solver = "EXACT"
        except Exception:
            pass
        apply_modifiers(target)
        bpy.data.objects.remove(c, do_unlink=True)


def bevel(ob, width_mm, segments=3, angle=30.0):
    m = ob.modifiers.new("bevel", "BEVEL")
    m.width = width_mm * MM; m.segments = segments
    m.limit_method = "ANGLE"; m.angle_limit = math.radians(angle)
    apply_modifiers(ob)
    finish_mesh(ob.data, 35.0)


def tube(points, normals, binormals, radius, sides, closed):
    """Sweeps a circle along 3D points. Returns verts, faces."""
    points = np.asarray(points); n = len(points)
    ang = np.linspace(0, 2 * math.pi, sides, endpoint=False)
    ring = np.cos(ang)[None, :, None] * np.asarray(normals)[:, None, :] + np.sin(ang)[None, :, None] * np.asarray(binormals)[:, None, :]
    verts = (points[:, None, :] + radius * ring).reshape(-1, 3)
    faces = []
    for i in range(n if closed else n - 1):
        j = (i + 1) % n
        for k in range(sides):
            l = (k + 1) % sides
            faces.append((i * sides + k, i * sides + l, j * sides + l, j * sides + k))
    if not closed:
        faces.append(tuple(range(sides))[::-1])
        faces.append(tuple((n - 1) * sides + k for k in range(sides)))
    return verts, faces


# =============================================================================
# materials
# =============================================================================
def principled(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if b is None:
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None) or nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(b.outputs[0], out.inputs[0])
    return m, nt, b


def setv(b, key, val):
    if key in b.inputs:
        b.inputs[key].default_value = val


def mat_velvet(name, rgb, pile=1.0):
    m, nt, b = principled(name)
    setv(b, "Roughness", 0.9)
    setv(b, "Sheen Weight", 1.0 * pile); setv(b, "Sheen Roughness", 0.35)
    setv(b, "Sheen Tint", (min(1, rgb[0] * 2.2 + 0.25), min(1, rgb[1] * 1.7 + 0.2), min(1, rgb[2] * 1.8 + 0.2), 1))
    setv(b, "Specular IOR Level", 0.15)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n1 = nt.nodes.new("ShaderNodeTexNoise"); n1.inputs["Scale"].default_value = 2600.0; n1.inputs["Detail"].default_value = 3.0
    n2 = nt.nodes.new("ShaderNodeTexNoise"); n2.inputs["Scale"].default_value = 90.0; n2.inputs["Detail"].default_value = 4.0
    nt.links.new(tc.outputs["Object"], n1.inputs["Vector"]); nt.links.new(tc.outputs["Object"], n2.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.3; ramp.color_ramp.elements[1].position = 0.75
    ramp.color_ramp.elements[0].color = (rgb[0] * 0.86, rgb[1] * 0.86, rgb[2] * 0.86, 1)
    ramp.color_ramp.elements[1].color = (rgb[0] * 1.08, rgb[1] * 1.08, rgb[2] * 1.08, 1)
    nt.links.new(n2.outputs[0], ramp.inputs[0]); nt.links.new(ramp.outputs[0], b.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.35; bump.inputs["Distance"].default_value = 0.00015
    nt.links.new(n1.outputs[0], bump.inputs["Height"]); nt.links.new(bump.outputs[0], b.inputs["Normal"])
    return m


def mat_webbing(name, rgb):
    """Woven webbing: bump from two crossed wave textures driven by the strap UVs (UV unit = 100 mm)."""
    m, nt, b = principled(name)
    setv(b, "Base Color", (*rgb, 1)); setv(b, "Roughness", 0.75)
    setv(b, "Sheen Weight", 0.3); setv(b, "Specular IOR Level", 0.25)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    waves = []
    for axis, scale in (("X", 70.0), ("Y", 110.0)):
        w = nt.nodes.new("ShaderNodeTexWave"); w.bands_direction = axis; w.wave_profile = "SIN"
        w.inputs["Scale"].default_value = scale
        nt.links.new(tc.outputs["UV"], w.inputs["Vector"]); waves.append(w)
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"
    nt.links.new(waves[0].outputs["Fac"], mul.inputs[0]); nt.links.new(waves[1].outputs["Fac"], mul.inputs[1])
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.6; bump.inputs["Distance"].default_value = 0.0002
    nt.links.new(mul.outputs[0], bump.inputs["Height"]); nt.links.new(bump.outputs[0], b.inputs["Normal"])
    return m


def mat_simple(name, rgb, rough=0.6, metallic=0.0, sheen=0.0):
    m, nt, b = principled(name)
    setv(b, "Base Color", (*rgb, 1)); setv(b, "Roughness", rough); setv(b, "Metallic", metallic)
    setv(b, "Sheen Weight", sheen)
    return m


# =============================================================================
# strap: webbing + velvet ribbon + stitching
# =============================================================================
def sweep_strap(l0, l1, o0, o1, w0, w1, rad):
    ells = lay.ell[(lay.ell > l0 + 1e-6) & (lay.ell < l1 - 1e-6)]
    ells = np.concatenate([[l0], ells, [l1]])
    prof = g.rounded_rect(o0, o1, w0, w1, rad)
    K = len(prof)
    verts = np.concatenate([lay.surface(ells, w + Z_STRAP, o)[:, None, :] for o, w in prof], axis=1).reshape(-1, 3)
    uv = np.concatenate([np.stack([ells / 100.0, np.full_like(ells, w / 100.0)], -1)[:, None, :] for o, w in prof], axis=1).reshape(-1, 2)
    n = len(ells)
    faces = [(i * K + k, i * K + (k + 1) % K, (i + 1) * K + (k + 1) % K, (i + 1) * K + k)
             for i in range(n - 1) for k in range(K)]
    faces.append(tuple(range(K))[::-1]); faces.append(tuple((n - 1) * K + k for k in range(K)))
    return verts, faces, uv


class Stitching:
    """Collects sewing-thread stitches lying on the strap surfaces."""
    R, SIDES = 0.19, 6
    PROFILE = np.array([-0.30, -0.10, -0.08, -0.10, -0.30])     # centre depth relative to the surface

    def __init__(self):
        self.verts, self.faces, self.n = [], [], 0

    def stitch(self, l0, w0, l1, w1, face):
        t = np.linspace(0, 1, len(self.PROFILE))
        ell = l0 + (l1 - l0) * t; w = w0 + (w1 - w0) * t
        off = face * (T / 2 + self.PROFILE)
        pts = lay.surface(ell, w + Z_STRAP, off)
        nrm = lay.surface_normal(ell) * face
        tan = np.gradient(pts, axis=0); tan /= np.linalg.norm(tan, axis=1)[:, None]
        bi = np.cross(tan, nrm); bi /= np.linalg.norm(bi, axis=1)[:, None]
        v, f = tube(pts, nrm, bi, self.R, self.SIDES, closed=False)
        self.faces += [tuple(i + self.n for i in q) for q in f]
        self.verts.append(v); self.n += len(v)

    def row(self, l0, w0, l1, w1, face, pitch=3.0, gap=0.45):
        length = math.hypot(l1 - l0, w1 - w0)
        n = max(1, int(round(length / pitch)))
        for i in range(n):
            a = (i + gap / pitch / 2) / n; b = (i + 1 - gap / pitch / 2) / n
            self.stitch(l0 + (l1 - l0) * a, w0 + (w1 - w0) * a, l0 + (l1 - l0) * b, w0 + (w1 - w0) * b, face)

    def cross_row(self, branches, s, half_w=10.0):
        for br in branches:
            l = lay.ell_on(br, s)
            self.row(l, -half_w, l, half_w, +1)

    def box_x(self, branches, s0, s1, half_w=8.5):
        for br in branches:
            a, b = lay.ell_on(br, s0), lay.ell_on(br, s1)
            for q in ((a, -half_w, b, -half_w), (b, -half_w, b, half_w), (b, half_w, a, half_w), (a, half_w, a, -half_w),
                      (a, -half_w, b, half_w), (a, half_w, b, -half_w)):
                self.row(*q, +1)


def build_strap(coll, mats):
    L = lay.length
    hw, hv = P["web_w"] / 2, P["vel_w"] / 2
    v, f, uv = sweep_strap(0.0, L, -T / 2, -T / 2 + P["web_t"], -hw, hw, 0.35)
    mesh_object("Webbing_25mm", v, f, coll, mats["web"], uv)
    # velvet ribbon: stops 2 mm short of each heat-sealed webbing end
    v, f, uv = sweep_strap(2.0, L - 2.0, T / 2 - P["vel_t"] + 0.01, T / 2, -hv, hv, 0.3)
    mesh_object("Velvet_Ribbon_22mm", v, f, coll, mats["velvet"], uv)

    st = Stitching()
    e = hv - 1.5                                   # edge-stitch 1.5 mm inside the ribbon edge
    for face in (+1, -1):                          # needle thread on the velvet, bobbin thread on the webbing
        for w in (-e, e):
            st.row(4.0, w, L - 4.0, w, face)
        for l in (4.0, L - 4.0):
            st.row(l, -e, l, e, face)
    # slider fixed loop: square with an X through the tail and the inner layer
    st.box_x(("tail_slider", "inner"), lay.x_tail1 + 5.0, lay.x_tail1 + 22.0)
    # fixed end: one row between buckle and D-ring, two rows behind the D-ring, box-X at the tail end
    st.cross_row(("main_bd", "tail_fixed"), lay.s_row_bd)
    for s in lay.s_rows_d:
        st.cross_row(("main_fixed", "tail_fixed"), s)
    st.box_x(("main_fixed", "tail_fixed"), lay.s_e + 5.0, lay.s_e + 22.0)
    mesh_object("Stitching", np.concatenate(st.verts), st.faces, coll, mats["thread"], sharp_deg=80)


# =============================================================================
# hardware
# =============================================================================
def build_slider(coll, mat):
    bw = P["slider_bar_w"] / 2; p = P["slider_pitch"]; ht = P["slider_bar_t"] / 2
    ob = solid("Slider_TriGlide", lambda bm: bm_box(bm, -p - bw, p + bw, -ht, ht, -16.5, 16.5), coll)
    cut = [solid("cut", lambda bm, x0=x0, x1=x1: bm_box(bm, x0, x1, -5, 5, -13.5, 13.5), coll)
           for x0, x1 in ((-p + bw, -bw), (bw, p - bw))]
    boolean(ob, cut, "DIFFERENCE")
    bevel(ob, 0.95, 4)
    ob.data.materials.clear(); ob.data.materials.append(mat)
    ob.matrix_world = frame_matrix(lay.s_sl, g.H_SLIDER)
    return ob


def d_ring_path():
    pts = [(z, 0.0) for z in np.linspace(-12, 12, 40)]
    pts += [(12 + 3 * math.cos(a), 3 + 3 * math.sin(a)) for a in np.linspace(-math.pi / 2, 0, 10)[1:]]
    pts += [(15 * math.cos(a), 5 + 15 * math.sin(a)) for a in np.linspace(0, math.pi, 64)]
    pts += [(-12 + 3 * math.cos(a), 3 + 3 * math.sin(a)) for a in np.linspace(math.pi, 1.5 * math.pi, 10)[:-1]]
    return np.array(pts)


def build_d_ring(coll, mat, tilt_deg=38.0):
    zu = d_ring_path()                                         # (z, u): bar along z, D rises along u
    pts = np.stack([np.zeros(len(zu)), zu[:, 1], zu[:, 0]], -1)
    d = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    d /= np.linalg.norm(d, axis=1)[:, None]
    bi = np.tile([1.0, 0.0, 0.0], (len(pts), 1))
    nr = np.cross(bi, d)
    v, f = tube(pts, nr, bi, P["bar_r"], 20, closed=True)
    ob = mesh_object("D_Ring_Welded", v, f, coll, mat, sharp_deg=60)
    ob.matrix_world = frame_matrix(lay.s_d, g.H_DRING) @ Matrix.Rotation(math.radians(tilt_deg), 4, "Z")
    return ob


def build_buckle(coll, mat):
    r = P["bar_r"]; xf, xm = P["s_bar_f"], P["s_bar_m"]
    # ---- female half: housing + strap bar --------------------------------
    def female(bm):
        bm_box(bm, -24, 4, -3.85, 3.85, -15.5, 15.5)
    fem = solid("Buckle_Female", female, coll)
    boolean(fem, [solid("c", lambda bm: bm_box(bm, -22.4, 5, -2.25, 2.25, -13.5, 13.5), coll),
                  solid("c", lambda bm: bm_box(bm, -17.0, -7.6, -2.25, 2.25, -17, 17), coll)], "DIFFERENCE")
    boolean(fem, [solid("c", lambda bm: bm_cyl_z(bm, xf, 0, r, -15.5, 15.5), coll),
                  solid("c", lambda bm: bm_box(bm, xf, -23.5, -r, r, 13.25, 15.5), coll),
                  solid("c", lambda bm: bm_box(bm, xf, -23.5, -r, r, -15.5, -13.25), coll)], "UNION")
    bevel(fem, 0.45, 3)
    # engraving on the outer face (reads left-to-right from outside the collar)
    try:
        cu = bpy.data.curves.new("engraving", "FONT")
        cu.body = ENGRAVING; cu.size = 5.0 * MM; cu.extrude = 0.3 * MM
        cu.align_x = "CENTER"; cu.align_y = "CENTER"
        txt = bpy.data.objects.new("engraving", cu); coll.objects.link(txt)
        txt.matrix_world = Matrix(((-1, 0, 0, -10.0 * MM), (0, 0, 1, 3.85 * MM), (0, 1, 0, 0), (0, 0, 0, 1)))
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(txt.evaluated_get(dg))
        me.transform(txt.matrix_world)
        cutter = bpy.data.objects.new("cut", me); coll.objects.link(cutter)
        bpy.data.objects.remove(txt, do_unlink=True)
        nv = len(fem.data.vertices)
        boolean(fem, [cutter], "DIFFERENCE")
        finish_mesh(fem.data, 35.0)
        print("engraving verts added:", len(fem.data.vertices) - nv)
    except Exception as ex:                                     # keep the build usable without a font
        print("engraving skipped:", ex)

    # ---- male half: base + strap bar + guide + two spring arms with hooks --
    male = solid("Buckle_Male", lambda bm: bm_box(bm, 4.15, 11, -3.85, 3.85, -15.5, 15.5), coll)
    parts = [solid("c", lambda bm: bm_cyl_z(bm, xm, 0, r, -15.5, 15.5), coll),
             solid("c", lambda bm: bm_box(bm, 10.5, xm, -r, r, 13.25, 15.5), coll),
             solid("c", lambda bm: bm_box(bm, 10.5, xm, -r, r, -15.5, -13.25), coll),
             solid("c", lambda bm: bm_box(bm, -21, 4.5, -2.0, 2.0, -3.5, 3.5), coll)]
    for sg in (1, -1):
        arm = [(4.5, sg * 9.2), (-17.5, sg * 9.2), (-17.5, sg * 12.0), (-16.6, sg * 12.9), (-13.6, sg * 15.5),
               (-8.0, sg * 15.5), (-8.0, sg * 12.9), (4.5, sg * 12.9)]
        parts.append(solid("c", lambda bm, a=arm: bm_prism_y(bm, a, -2.0, 2.0), coll))
    boolean(male, parts, "UNION")
    bevel(male, 0.4, 3)
    M = frame_matrix(0.0, g.H_BUCKLE)
    for ob in (fem, male):
        ob.data.materials.clear(); ob.data.materials.append(mat)
        ob.matrix_world = M
    return fem, male


# =============================================================================
# bow tie (detachable; elastic loop on the back slides over the collar)
# =============================================================================
BOW = dict(half_len=48.0, fabric_h=55.0, knot_h=28.0, sheet_t=1.1, g_center=0.65, g_end=6.5,
           knot_half_w=6.5)


def zigzag(H, A, R, n):
    """Pleated cross-section (z, y): rounded zig-zag sampled at n equal fabric steps. Returns pts, normals, length."""
    hh = H / 2
    V = np.array([(-hh, 0), (-0.8 * hh, A), (-0.4 * hh, -A), (0, A), (0.4 * hh, -A), (0.8 * hh, A), (hh, 0)], float)
    if A < 1e-4:
        z = np.linspace(-hh, hh, n)
        return np.stack([z, np.zeros(n)], -1), np.tile([0.0, 1.0], (n, 1)), H, 1e9
    out = [V[0]]; r_min = 1e9
    for i in range(1, len(V) - 1):
        d1 = V[i] - V[i - 1]; l1 = np.linalg.norm(d1); d1 /= l1
        d2 = V[i + 1] - V[i]; l2 = np.linalg.norm(d2); d2 /= l2
        phi = math.acos(np.clip(d1 @ d2, -1, 1))
        lim1 = (0.9 if i == 1 else 0.45) * l1              # end segments carry one fillet only
        lim2 = (0.9 if i == len(V) - 2 else 0.45) * l2
        t = min(R * math.tan(phi / 2), lim1, lim2)
        r = t / math.tan(phi / 2); r_min = min(r_min, r)
        sg = np.sign(d1[0] * d2[1] - d1[1] * d2[0])
        c = V[i] - d1 * t + sg * r * np.array([-d1[1], d1[0]])
        a0 = math.atan2(*(V[i] - d1 * t - c)[::-1])
        for k in np.linspace(0, 1, 9):
            a = a0 + sg * phi * k
            out.append(c + r * np.array([math.cos(a), math.sin(a)]))
    out.append(V[-1])
    out = np.array(out)
    cl = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(out, axis=0), axis=1))])
    u = np.linspace(0, cl[-1], n)
    pts = np.stack([np.interp(u, cl, out[:, 0]), np.interp(u, cl, out[:, 1])], -1)
    d = np.gradient(pts, axis=0); d /= np.linalg.norm(d, axis=1)[:, None]
    nrm = np.stack([-d[:, 1], d[:, 0]], -1)                 # (z, y) normal pointing to +y on flat fabric
    return pts, nrm, cl[-1], r_min


def bow_column(x, nv):
    """Mid-surface of the bow at position x: pleats sized so the fabric keeps its 55 mm height."""
    B = BOW
    th = np.clip((abs(x) - 7.0) / 27.0, 0, 1)
    H = B["knot_h"] + (B["fabric_h"] - B["knot_h"]) * g._sstep(th)
    tg = np.clip((abs(x) - 9.0) / (B["half_len"] - 9.0), 0, 1)
    gap = B["g_center"] + (B["g_end"] - B["g_center"]) * g._sstep(tg)
    R = max(1.6, gap + 1.5)
    lo, hi = 0.0, 20.0
    if B["fabric_h"] - H < 1e-3:
        A = 0.0
    else:
        for _ in range(40):
            A = 0.5 * (lo + hi)
            if zigzag(H, A, R, 40)[2] < B["fabric_h"]:
                lo = A
            else:
                hi = A
    pts, nrm, _, r_min = zigzag(H, A, R, nv)
    gap = max(B["g_center"], min(gap, r_min - 1.2))        # layers may not fold tighter than the pleat radius
    return pts, nrm, gap


def build_bow(coll, mats):
    B = BOW; nv = 81; Lx = B["half_len"]
    cols = []                                                # each: (nv,3) points in bow space (x, y, z)
    xs = np.linspace(-Lx, Lx, 161)

    def layer(x, sign):
        pts, nrm, gap = bow_column(x, nv)
        return np.stack([np.full(nv, x), pts[:, 1] + sign * gap * nrm[:, 1], pts[:, 0] + sign * gap * nrm[:, 0]], -1)

    def cap(sign_x):
        pts, nrm, gap = bow_column(sign_x * Lx, nv)
        a0 = math.pi / 2 if sign_x > 0 else -math.pi / 2
        res = []
        for a in np.linspace(a0, a0 - math.pi, 18)[1:-1]:
            res.append(np.stack([np.full(nv, sign_x * Lx + gap * math.cos(a)), pts[:, 1] + gap * math.sin(a), pts[:, 0]], -1))
        return res

    cols += [layer(x, +1) for x in xs]
    cols += cap(+1)
    cols += [layer(x, -1) for x in xs[::-1]]
    cols += cap(-1)
    nu = len(cols)
    verts = np.concatenate(cols)
    faces = [(i * nv + k, ((i + 1) % nu) * nv + k, ((i + 1) % nu) * nv + k + 1, i * nv + k + 1)
             for i in range(nu) for k in range(nv - 1)]
    body = mesh_object("Bow_Body", verts, faces, coll, mats["velvet_bow"], sharp_deg=60)
    m = body.modifiers.new("solid", "SOLIDIFY"); m.thickness = B["sheet_t"] * MM; m.offset = 0.0
    apply_modifiers(body)
    bevel(body, 0.35, 2, angle=60)

    # knot band wrapped round the pinched centre
    pts0, _, _ = bow_column(0.0, nv)
    hy = np.abs(pts0[:, 1]).max() + B["g_center"] + B["sheet_t"] / 2 + 0.2 + 0.7
    hz = B["knot_h"] / 2 + B["g_center"] + B["sheet_t"] / 2 + 0.3 + 0.7
    loop = np.array(g.rounded_rect(-hy, hy, -hz, hz, 4.2, seg=8))      # (y, z)
    xk = np.linspace(-B["knot_half_w"], B["knot_half_w"], 9)
    ctr = loop.mean(axis=0); out = loop - ctr; out /= np.linalg.norm(out, axis=1)[:, None]
    kv = []
    for x in xk:
        bulge = 0.7 * (1 - (x / B["knot_half_w"]) ** 2)
        q = loop + out * bulge
        kv.append(np.stack([np.full(len(q), x), q[:, 0], q[:, 1]], -1))
    K = len(loop); kv = np.concatenate(kv)
    kf = [(i * K + k, i * K + (k + 1) % K, (i + 1) * K + (k + 1) % K, (i + 1) * K + k)
          for i in range(len(xk) - 1) for k in range(K)]
    knot = mesh_object("Bow_Knot_Band", kv, kf, coll, mats["velvet_bow"], sharp_deg=60)
    m = knot.modifiers.new("solid", "SOLIDIFY"); m.thickness = 1.4 * MM; m.offset = 0.0
    apply_modifiers(knot); bevel(knot, 0.4, 2, angle=60)

    # elastic loop sewn to the back of the knot (shown relaxed and flat)
    yf = -(hy + 0.7 + 0.7 + 0.15 + 0.45); rr = 0.65
    el = [(yf, z) for z in np.linspace(-13.5, 13.5, 24)]
    el += [(yf - rr + rr * math.cos(a), 13.5 + rr * math.sin(a)) for a in np.linspace(0, math.pi, 10)[1:-1]]
    el += [(yf - 2 * rr, z) for z in np.linspace(13.5, -13.5, 24)]
    el += [(yf - rr + rr * math.cos(a), -13.5 + rr * math.sin(a)) for a in np.linspace(math.pi, 2 * math.pi, 10)[1:-1]]
    el = np.array(el); K = len(el)
    ev = np.concatenate([np.stack([np.full(K, x), el[:, 0], el[:, 1]], -1) for x in (-5.0, 5.0)])
    ef = [(k, (k + 1) % K, K + (k + 1) % K, K + k) for k in range(K)]
    ela = mesh_object("Bow_Elastic_Loop", ev, ef, coll, mats["elastic"], sharp_deg=60)
    m = ela.modifiers.new("solid", "SOLIDIFY"); m.thickness = 0.9 * MM; m.offset = 0.0
    apply_modifiers(ela); finish_mesh(ela.data, 40)

    # ---- pose: lying face-up on the floor in front of the collar, settled on knot + one loop
    parts = (body, knot, ela)
    allv = np.concatenate([np.array([v.co[:] for v in o.data.vertices]) for o in parts]) / MM
    base = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)   # bow x->X, bow front (y)->Z, bow top (z)->-Y

    def lows(alpha):
        ry = np.array([[math.cos(alpha), 0, math.sin(alpha)], [0, 1, 0], [-math.sin(alpha), 0, math.cos(alpha)]])
        w = allv @ (ry @ base).T
        return w[np.abs(allv[:, 0]) < 12, 2].min(), w[allv[:, 0] > 25, 2].min(), ry
    lo, hi = 0.0, math.radians(20)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        c, e, _ = lows(mid)
        if e > c:
            lo = mid
        else:
            hi = mid
    c, e, ry = lows(hi)
    rot = ry @ base
    emp = bpy.data.objects.new("Bow_Tie", None); coll.objects.link(emp)
    emp.empty_display_size = 0.02
    M = Matrix.Identity(4)
    for i in range(3):
        for j in range(3):
            M[i][j] = rot[i, j]
    M.translation = Vector((0.0, -78.0 * MM, -min(c, e) * MM))
    emp.matrix_world = M
    for o in parts:
        o.parent = emp
    return emp


# =============================================================================
# studio
# =============================================================================
def look_at(ob, target):
    d = Vector(target) - ob.location
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def build_studio(coll):
    sc = bpy.context.scene
    me = bpy.data.meshes.new("Floor"); s = 3.0
    me.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
    floor = bpy.data.objects.new("Floor", me); coll.objects.link(floor)
    me.materials.append(mat_simple("Floor_White", (0.9, 0.9, 0.9), rough=0.55))
    world = bpy.data.worlds.new("Studio"); sc.world = world; world.use_nodes = True
    bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs[0].default_value = (1, 1, 1, 1); bg.inputs[1].default_value = 0.45
    cy = float(lay.base.py.mean()) * MM
    for name, loc, size, power in (("Key", (-0.35, -0.45, 0.6), 0.5, 3.0), ("Fill", (0.5, -0.25, 0.35), 0.6, 1.0),
                                   ("Rim", (0.1, 0.6, 0.45), 0.5, 1.5)):
        ld = bpy.data.lights.new(name, "AREA"); ld.size = size; ld.energy = power
        lo = bpy.data.objects.new("Light_" + name, ld); coll.objects.link(lo)
        lo.location = loc; look_at(lo, (0, cy, 0.01))
    cams = {}
    px, py = lay.base.frame(np.array([lay.s_sl, lay.s_d]))[:2]
    for name, loc, tgt, lens in (
            ("Cam_Hero", (0.0, -0.52, 0.40), (0.0, 0.0, 0.012), 85),
            ("Cam_Top", (0.0, cy - 0.02, 0.62), (0.0, cy - 0.02, 0.0), 85),
            ("Cam_Buckle", (0.06, -0.20, 0.10), (0.012, 0.0, 0.016), 85),
            ("Cam_Slider", (px[0] * MM - 0.20, py[0] * MM - 0.05, 0.10), (px[0] * MM, py[0] * MM, 0.016), 85),
            ("Cam_Inside", (-0.03, cy + 0.02, 0.17), (px[1] * MM - 0.02, 0.012, 0.012), 60)):
        cd = bpy.data.cameras.new(name); cd.lens = lens; cd.clip_start = 0.01; cd.clip_end = 20
        co = bpy.data.objects.new(name, cd); coll.objects.link(co)
        co.location = loc; look_at(co, tgt); cams[name] = co
    sc.camera = cams["Cam_Hero"]
    return cams


def setup_render(samples, scale=100):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"; sc.cycles.samples = samples
    try:
        sc.cycles.use_denoising = True
    except Exception:
        pass
    sc.render.resolution_x, sc.render.resolution_y = 1600, 1280
    sc.render.resolution_percentage = scale
    sc.view_settings.view_transform = "Standard"
    sc.render.image_settings.file_format = "PNG"


# =============================================================================
def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    samples = int(argv[argv.index("--samples") + 1]) if "--samples" in argv else 96
    reset_scene()
    mats = dict(
        velvet=mat_velvet("Velvet_Ribbon", VELVET_RGB),
        velvet_bow=mat_velvet("Velvet_Bow", VELVET_RGB),
        web=mat_webbing("Webbing", WEB_RGB),
        thread=mat_simple("Thread", THREAD_RGB, rough=0.6, sheen=0.5),
        metal=mat_simple("Rose_Gold", METAL_RGB, rough=0.2, metallic=1.0),
        elastic=mat_simple("Elastic", WEB_RGB, rough=0.8, sheen=0.4),
    )
    c_strap, c_hw, c_bow, c_studio = (collection(n) for n in ("Collar_Strap", "Collar_Hardware", "Bow_Tie", "Studio"))
    build_strap(c_strap, mats)
    build_slider(c_hw, mats["metal"])
    build_d_ring(c_hw, mats["metal"])
    build_buckle(c_hw, mats["metal"])
    build_bow(c_bow, mats)
    cams = build_studio(c_studio)
    setup_render(samples, int(argv[argv.index("--scale") + 1]) if "--scale" in argv else 100)
    sc = bpy.context.scene
    sc["cut_length_mm"] = round(lay.length, 1)
    sc["neck_circumference_mm"] = lay.C
    sc["slider_to_male_bar_mm"] = round(lay.doubled, 1)
    bpy.context.view_layer.update()
    out = os.path.join(HERE, "velvet_ribbon_dog_collar.blend")
    bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
    print("saved", out)
    for o in bpy.data.objects:
        if o.type == "MESH":
            print(f"  {o.name:22s} verts={len(o.data.vertices)}")
    if "--render" in argv:
        only = argv[argv.index("--only") + 1].split(",") if "--only" in argv else list(cams)
        os.makedirs(os.path.join(HERE, "renders"), exist_ok=True)
        for name in only:
            sc.camera = cams[name]
            sc.render.filepath = os.path.join(HERE, "renders", name.replace("Cam_", "").lower() + ".png")
            bpy.ops.render.render(write_still=True)
        sc.camera = cams["Cam_Hero"]


if __name__ == "__main__":
    main()
