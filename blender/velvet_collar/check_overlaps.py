"""Reports surface intersections between parts of the saved .blend, and of each part with itself
(0 everywhere = nothing interpenetrates).
Stitching is skipped: thread is sewn into the strap on purpose."""
import os, sys, itertools
import bpy, bmesh
from mathutils.bvhtree import BVHTree
here = os.path.dirname(os.path.abspath(__file__))
bpy.ops.wm.open_mainfile(filepath=os.path.join(here, "velvet_ribbon_dog_collar.blend"))
trees = {}; selfbad = 0
for o in bpy.data.objects:
    if o.type != "MESH" or o.name == "Stitching":
        continue
    bm = bmesh.new(); bm.from_mesh(o.data); bm.transform(o.matrix_world); bm.normal_update()
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.faces.ensure_lookup_table()
    t = trees[o.name] = BVHTree.FromBMesh(bm, epsilon=0.0)
    selfhits = sum(1 for i, j in t.overlap(t) if i < j and not
                   ({v.index for v in bm.faces[i].verts} & {v.index for v in bm.faces[j].verts}))
    if selfhits:
        selfbad += 1; print(f"SELF-INTERSECTION {o.name}: {selfhits} triangle pairs")
    zmin = min(v.co.z for v in bm.verts)
    print(f"{o.name:22s} lowest point z = {zmin*1000:7.3f} mm")
bad = 0
for a, b in itertools.combinations(sorted(trees), 2):
    if "Floor" in (a, b):
        continue                      # resting contact; see lowest-point list above
    n = len(trees[a].overlap(trees[b]))
    if n:
        bad += 1; print(f"INTERSECTION {a} x {b}: {n} triangle pairs")
print("intersecting pairs:", bad, "| self-intersecting parts:", selfbad)
sys.exit(1 if bad or selfbad else 0)
