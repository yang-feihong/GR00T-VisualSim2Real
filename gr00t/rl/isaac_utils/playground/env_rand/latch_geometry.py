"""Embedded DoorMan bolt and strike geometry, aligned with lab30 c32e252."""

from pxr import Gf, UsdGeom, UsdPhysics

LATCH_MODEL = "embedded_bevel_strike_v1"


def build_strike_pocket(
    stage, frame_path, half_width, half_wall_length, half_wall_height, side, height
):
    """Split the jamb around a 24 mm opening and a 22 mm deep strike pocket."""
    frame = stage.DefinePrim(frame_path, "Xform")
    UsdPhysics.CollisionAPI(frame).GetCollisionEnabledAttr().Set(False)
    UsdGeom.Xformable(frame).MakeMatrixXform().Set(Gf.Matrix4d(1))
    x0, x1 = -0.07, 0.03
    u0, u1 = half_width, half_width + 2 * half_wall_length
    z0, z1 = 0.0, 2 * half_wall_height
    hx0, hx1, hz0, hz1 = -0.012, 0.012, height - 0.012, height + 0.012
    boxes = [
        ("below", x0, x1, u0, u1, z0, hz0),
        ("above", x0, x1, u0, u1, hz1, z1),
        ("negative_face", x0, hx0, u0, u1, hz0, hz1),
        ("positive_face", hx1, x1, u0, u1, hz0, hz1),
        ("pocket_back", hx0, hx1, u0 + 0.022, u1, hz0, hz1),
    ]
    for name, xa, xb, ua, ub, za, zb in boxes:
        if not (xb > xa and ub > ua and zb > za):
            raise ValueError("Strike pocket does not fit inside the generated jamb")
        cube = UsdGeom.Cube.Define(stage, frame_path + "/" + name)
        cube.AddTranslateOp().Set(Gf.Vec3d((xa + xb) / 2, side * (ua + ub) / 2, (za + zb) / 2))
        cube.AddScaleOp().Set(Gf.Vec3f((xb - xa) / 2, (ub - ua) / 2, (zb - za) / 2))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def build_latch_bolt(stage, path, side, opening):
    """Keep 30 mm embedded in the panel and a 20 mm bevel in the strike."""
    a = 0.008
    outline = [
        (-opening * a, -0.030),
        (opening * a, -0.030),
        (opening * a, 0.0),
        (-opening * a, 0.020),
    ]
    points = [Gf.Vec3f(x, side * u, z) for z in [-a, a] for x, u in outline]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    faces = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    if side * opening < 0:
        faces = [list(reversed(face)) for face in faces]
    mesh.CreateFaceVertexIndicesAttr([vertex for face in faces for vertex in face])
    mesh.CreateSubdivisionSchemeAttr().Set("none")
    mesh.GetPurposeAttr().Set("default")
    UsdPhysics.MassAPI.Apply(mesh.GetPrim()).CreateMassAttr().Set(0.1)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("convexHull")
