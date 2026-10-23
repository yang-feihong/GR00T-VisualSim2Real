from pathlib import Path
from functools import lru_cache
import math


RL_ROOT = Path(__file__).resolve().parents[1]
ROBOT_RESOURCES = RL_ROOT / "third_party/visual_wholebody_lowlevel/resources/robots/b2z1"
ROBOT_USD = RL_ROOT / "data/robots/b2z1_lab30/b2z1.usd"
ROBOT_URDF = ROBOT_RESOURCES / "urdf/b2z1.urdf"

def set_robot_usd(usd_path):
    global ROBOT_USD
    resolved = Path(usd_path).expanduser().resolve()
    if resolved != ROBOT_USD:
        ROBOT_USD = resolved
        read_arm_mount.cache_clear()


@lru_cache(maxsize=8)
def read_arm_mount(usd_path=None):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    usd_path = ROBOT_USD if usd_path is None else usd_path
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError(f"Cannot open robot USD: {usd_path}")
    mounts = [prim for prim in stage.Traverse()
              if prim.GetName() == "z1_mount_joint" and prim.IsA(UsdPhysics.FixedJoint)]
    if len(mounts) != 1:
        raise ValueError(f"Expected one z1_mount_joint in {usd_path}, found {len(mounts)}")
    joint = UsdPhysics.FixedJoint(mounts[0])
    parents = joint.GetBody0Rel().GetTargets()
    children = joint.GetBody1Rel().GetTargets()
    if len(parents) != 1 or len(children) != 1 or parents[0].name != "base_link" or children[0].name != "link00":
        raise ValueError(f"Unexpected arm mounting bodies in {usd_path}")
    frames = []
    for position, orientation in (
        (joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
        (joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()),
    ):
        frame = Gf.Matrix4d(1.0)
        frame.SetRotate(Gf.Quatd(orientation))
        frame.SetTranslateOnly(Gf.Vec3d(position))
        frames.append(frame)
    transform = frames[1].GetInverse() * frames[0]
    if not Gf.IsClose(transform.TransformDir(Gf.Vec3d(0, 0, 1)), Gf.Vec3d(0, 0, 1), 1e-6):
        raise ValueError("Arm kinematics currently require a yaw-only mounting rotation")
    units = UsdGeom.GetStageMetersPerUnit(stage)
    translation = tuple(float(value) * units for value in transform.ExtractTranslation())
    forward = transform.TransformDir(Gf.Vec3d(1, 0, 0))
    return translation, math.atan2(forward[1], forward[0])


BOX_MOUNT_XYZ = (-0.05, 0.0, 0.0930129411816597)
BOX_MOUNT_YAW = 1.5707963267948966
BOX_COLOR = (0.75, 0.76, 0.78)
