import argparse
import json
import math
import os
import uuid
import xml.etree.ElementTree as ElementTree

from pxr import Gf, Usd, UsdGeom, UsdPhysics
from b2z1_mesh import read_dae

from gr00t.rl.utils.b2z1_asset import (
    BOX_COLOR, BOX_MOUNT_XYZ, BOX_MOUNT_YAW,
    ROBOT_RESOURCES, ROBOT_URDF, ROBOT_USD, read_arm_mount,
)


def write_urdf(source, mount_xyz, mount_yaw, mesh, colliders):
    robot = ElementTree.parse(source).getroot()
    for element in robot.findall(".//mesh"):
        filename = element.get("filename")
        if "://" not in filename:
            element.set("filename", os.path.relpath((source.parent / filename).resolve(), ROBOT_URDF.parent))
    mount = robot.find("joint[@name='z1_mount_joint']/origin")
    mount.set("xyz", " ".join(str(value) for value in mount_xyz))
    mount.set("rpy", f"0 0 {mount_yaw}")
    base = robot.find("link[@name='base_link']")
    visual = ElementTree.SubElement(base, "visual", {"name": "back_box"})
    ElementTree.SubElement(visual, "origin", {
        "xyz": " ".join(str(value) for value in BOX_MOUNT_XYZ),
        "rpy": f"0 0 {BOX_MOUNT_YAW}",
    })
    geometry = ElementTree.SubElement(visual, "geometry")
    ElementTree.SubElement(geometry, "mesh", {"filename": os.path.relpath(mesh, ROBOT_URDF.parent)})
    material = ElementTree.SubElement(visual, "material", {"name": "back_box_silver"})
    ElementTree.SubElement(material, "color", {"rgba": " ".join(str(value) for value in (*BOX_COLOR, 1))})
    box_rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(BOX_MOUNT_YAW))
    for collider in colliders:
        center = Gf.Vec3d(*BOX_MOUNT_XYZ) + box_rotation.TransformDir(Gf.Vec3d(*collider["center_m"]))
        quaternion = collider["quaternion_wxyz"]
        rotation = box_rotation.GetQuat() * Gf.Quatd(quaternion[0], Gf.Vec3d(*quaternion[1:]))
        real = rotation.GetReal()
        imaginary_x, imaginary_y, imaginary_z = rotation.GetImaginary()
        roll = math.atan2(2 * (real * imaginary_x + imaginary_y * imaginary_z),
                          1 - 2 * (imaginary_x ** 2 + imaginary_y ** 2))
        pitch = math.asin(max(-1, min(1, 2 * (real * imaginary_y - imaginary_z * imaginary_x))))
        yaw = math.atan2(2 * (real * imaginary_z + imaginary_x * imaginary_y),
                         1 - 2 * (imaginary_y ** 2 + imaginary_z ** 2))
        collision = ElementTree.SubElement(base, "collision", {"name": collider["name"]})
        ElementTree.SubElement(collision, "origin", {
            "xyz": " ".join(str(value) for value in center),
            "rpy": f"{roll} {pitch} {yaw}",
        })
        shape = ElementTree.SubElement(collision, "geometry")
        ElementTree.SubElement(shape, "box", {"size": " ".join(str(value) for value in collider["size_m"])})
    ElementTree.indent(robot, space="  ")
    ElementTree.ElementTree(robot).write(ROBOT_URDF, encoding="utf-8", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-mount-xyz", type=float, nargs=3)
    args = parser.parse_args()
    source = ROBOT_USD.parent / "b2z1_mount_90_x0p22_y0_z0p17.usd"
    original = Usd.Stage.Open(str(source))
    root_path = original.GetDefaultPrim().GetPath()
    joint_path = root_path.AppendPath("joints/z1_mount_joint")
    joint = UsdPhysics.FixedJoint(original.GetPrimAtPath(joint_path))
    original_position = joint.GetLocalPos0Attr().Get()
    original_xyz, original_yaw = read_arm_mount(source)
    if args.arm_mount_xyz is None:
        mount_xyz, mount_yaw = read_arm_mount()
    else:
        mount_xyz, mount_yaw = tuple(args.arm_mount_xyz), original_yaw
    if abs(math.remainder(mount_yaw - original_yaw, 2 * math.pi)) > 1e-6:
        raise ValueError("This builder preserves the original mounting rotation")
    delta = Gf.Vec3d(*mount_xyz) - Gf.Vec3d(*original_xyz)
    mesh_path = ROBOT_RESOURCES / "meshes/back_box.dae"
    points, faces = read_dae(mesh_path)
    colliders = json.loads((ROBOT_RESOURCES / "cad/back_box_colliders.json").read_text())["colliders"]
    temporary = ROBOT_USD.with_name(f".b2z1-build-{uuid.uuid4().hex}.usda")
    try:
        stage = Usd.Stage.CreateNew(str(temporary))
        root = UsdGeom.Xform.Define(stage, root_path)
        root.GetPrim().GetReferences().AddReference(os.path.relpath(source, temporary.parent))
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(original))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(original))
        arm_links = [f"link{index:02d}" for index in range(7)] + ["gripper_link", "gripperMover"]
        for name in arm_links:
            path = root_path.AppendChild(name)
            translation = original.GetPrimAtPath(path).GetAttribute("xformOp:translate").Get()
            stage.GetPrimAtPath(path).GetAttribute("xformOp:translate").Set(Gf.Vec3d(translation) + delta)
        stage.GetPrimAtPath(joint_path).GetAttribute("physics:localPos0").Set(
            Gf.Vec3f(Gf.Vec3d(original_position) + delta)
        )
        stage.GetPrimAtPath(root_path.AppendPath("base_link/visuals")).SetInstanceable(False)
        mesh = UsdGeom.Mesh.Define(stage, root_path.AppendPath("base_link/visuals/back_box"))
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
        mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*BOX_COLOR)])
        mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(mesh.GetPointsAttr().Get()))
        mesh.AddTranslateOp().Set(Gf.Vec3d(*BOX_MOUNT_XYZ))
        mesh.AddRotateZOp().Set(math.degrees(BOX_MOUNT_YAW))
        collision_root = UsdGeom.Xform.Define(stage, root_path.AppendPath("base_link/back_box_collisions"))
        collision_root.AddTranslateOp().Set(Gf.Vec3d(*BOX_MOUNT_XYZ))
        collision_root.AddRotateZOp().Set(math.degrees(BOX_MOUNT_YAW))
        collision_root.CreatePurposeAttr(UsdGeom.Tokens.guide)
        for collider in colliders:
            cube = UsdGeom.Cube.Define(stage, collision_root.GetPath().AppendChild(collider["name"]))
            cube.CreateSizeAttr(1)
            cube.AddTranslateOp().Set(Gf.Vec3d(*collider["center_m"]))
            quaternion = collider["quaternion_wxyz"]
            cube.AddOrientOp().Set(Gf.Quatf(quaternion[0], Gf.Vec3f(*quaternion[1:])))
            cube.AddScaleOp().Set(Gf.Vec3f(*collider["size_m"]))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
        if not stage.GetRootLayer().Export(str(ROBOT_USD)):
            raise RuntimeError(f"Cannot save {ROBOT_USD}")
    finally:
        temporary.unlink(missing_ok=True)
    source_urdf = ROBOT_RESOURCES / "urdf/b2z1_base.urdf"
    write_urdf(source_urdf, mount_xyz, original_yaw, mesh_path, colliders)
    report = {
        "source_usd": source.name,
        "arm_mount_xyz_m": mount_xyz,
        "box_mount_xyz_m": BOX_MOUNT_XYZ,
        "box_yaw_rad": BOX_MOUNT_YAW,
        "box_appearance": "silver-white approximation; source STEP contains no color",
        "physics": "original robot mass, inertia, colliders, and joint settings preserved; box colliders belong to base_link",
        "box_collider_count": len(colliders),
        "urdf": str(ROBOT_URDF.relative_to(ROBOT_RESOURCES)),
    }
    (ROBOT_RESOURCES / "cad/assembly.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
