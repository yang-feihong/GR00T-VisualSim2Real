import xml.etree.ElementTree as ElementTree

import numpy as np

from gr00t.rl.utils.b2z1_asset import (
    BOX_MOUNT_XYZ, ROBOT_RESOURCES, ROBOT_URDF, ROBOT_USD, read_arm_mount,
)


def test_production_assembly_preserves_joint_and_physics_contract():
    original = ElementTree.parse(
        ROBOT_RESOURCES / "urdf/b2z1_base.urdf"
    ).getroot()
    robot = ElementTree.parse(ROBOT_URDF).getroot()
    source_joints = {joint.get("name"): joint for joint in original.findall("joint")}
    joints = {joint.get("name"): joint for joint in robot.findall("joint")}
    assert source_joints.keys() == joints.keys()
    for name, joint in joints.items():
        source = source_joints[name]
        assert joint.attrib == source.attrib
        for tag in ("parent", "child", "axis", "limit", "dynamics"):
            source_element, element = source.find(tag), joint.find(tag)
            assert (None if element is None else element.attrib) == (
                None if source_element is None else source_element.attrib
            )
        if name != "z1_mount_joint":
            assert joint.find("origin").attrib == source.find("origin").attrib
    for source in original.findall("link"):
        link = robot.find(f"link[@name='{source.get('name')}']")
        inertial, original_inertial = link.find("inertial"), source.find("inertial")
        assert (None if inertial is None else ElementTree.tostring(inertial)) == (
            None if original_inertial is None else ElementTree.tostring(original_inertial)
        )
        original_collisions = [collision for collision in link.findall("collision")
                               if not collision.get("name", "").startswith("back_box_")]
        assert len(original_collisions) == len(source.findall("collision"))


def test_box_has_wall_and_floor_colliders():
    robot = ElementTree.parse(ROBOT_URDF).getroot()
    colliders = [collision for collision in robot.findall("link[@name='base_link']/collision")
                 if collision.get("name", "").startswith("back_box_")]
    assert len(colliders) == 63
    for collider in colliders:
        size = np.fromstring(collider.find("geometry/box").get("size"), sep=" ")
        assert np.all(size > 0)


def test_production_mounts_and_mesh_resources():
    robot = ElementTree.parse(ROBOT_URDF).getroot()
    mount = robot.find("joint[@name='z1_mount_joint']/origin")
    np.testing.assert_allclose(np.fromstring(mount.get("xyz"), sep=" "), read_arm_mount()[0])
    box = robot.find("link[@name='base_link']/visual[@name='back_box']/origin")
    np.testing.assert_allclose(np.fromstring(box.get("xyz"), sep=" "), BOX_MOUNT_XYZ)
    for mesh in robot.findall(".//mesh"):
        assert (ROBOT_URDF.parent / mesh.get("filename")).is_file(), mesh.get("filename")
    assert ROBOT_USD.is_file()
