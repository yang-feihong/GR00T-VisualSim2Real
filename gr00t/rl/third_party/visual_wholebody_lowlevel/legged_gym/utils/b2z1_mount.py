import os
import re
import math


VALID_MOUNT_DEGREES = (0, 90, 180, 270)
MOUNT_URDF_SPECS = {
    "b2z1": {
        "source_urdf_rel_path": os.path.join(
            "resources", "robots", "b2z1", "urdf", "b2z1_isaacsim_mesh_axis_fixed.urdf"
        ),
        "generated_urdf_dir_rel_path": os.path.join("resources", "robots", "b2z1", "urdf", "generated"),
        "generated_filename_prefix": "b2z1_mount",
        "mount_joint_name": "z1_mount_joint",
        "default_xyz": ["0.2", "0", "0.09"],
    },
}


def normalize_mount_xyz(mount_xyz):
    if len(mount_xyz) != 3:
        raise ValueError(f"mount_xyz must contain exactly 3 values, got {mount_xyz!r}")
    return tuple(float(value) for value in mount_xyz)


def _format_mount_xyz_token(value):
    value = float(value)
    if abs(value) < 1e-12:
        value = 0.0
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def _is_relative_resource_path(path):
    return not (
        os.path.isabs(path)
        or "://" in path
        or path.startswith("package://")
    )


def _rebase_urdf_resource_paths(line, source_dir, output_dir):
    def replace(match):
        original_path = match.group(1)
        if not _is_relative_resource_path(original_path):
            return match.group(0)

        source_abs_path = os.path.normpath(os.path.join(source_dir, original_path))
        rebased_path = os.path.relpath(source_abs_path, output_dir).replace(os.sep, "/")
        return match.group(0).replace(original_path, rebased_path, 1)

    return re.sub(r'filename="([^"]+)"', replace, line)


def normalize_mount_deg(mount_deg):
    mount_deg = int(round(float(mount_deg))) % 360
    if mount_deg not in VALID_MOUNT_DEGREES:
        raise ValueError(f"Unsupported mount_deg={mount_deg}. Expected one of {VALID_MOUNT_DEGREES}.")
    return mount_deg


def mount_deg_to_rad(mount_deg):
    return math.radians(normalize_mount_deg(mount_deg))


def _get_mount_urdf_spec(generator_name):
    try:
        return MOUNT_URDF_SPECS[generator_name]
    except KeyError as exc:
        supported = ", ".join(sorted(MOUNT_URDF_SPECS.keys()))
        raise ValueError(f"Unsupported mount_urdf_generator={generator_name!r}. Supported values: {supported}.") from exc


def get_generated_mount_urdf_rel_path(generator_name, mount_deg, mount_xyz, variant_token=""):
    mount_deg = normalize_mount_deg(mount_deg)
    mount_xyz = normalize_mount_xyz(mount_xyz)
    spec = _get_mount_urdf_spec(generator_name)
    xyz_suffix = "_".join(
        f"{axis}{_format_mount_xyz_token(value)}"
        for axis, value in zip(("x", "y", "z"), mount_xyz)
    )
    variant_suffix = f"_{variant_token}" if variant_token else ""
    filename = f"{spec['generated_filename_prefix']}_{mount_deg}_{xyz_suffix}{variant_suffix}.urdf"
    return os.path.join(spec["generated_urdf_dir_rel_path"], filename)


def ensure_mount_urdf(
    root_dir,
    generator_name,
    mount_deg,
    mount_xyz,
    source_urdf_rel_path=None,
    variant_token="",
    keep_collision_links=None,
):
    mount_deg = normalize_mount_deg(mount_deg)
    mount_xyz = normalize_mount_xyz(mount_xyz)
    spec = _get_mount_urdf_spec(generator_name)
    source_path = os.path.join(root_dir, source_urdf_rel_path or spec["source_urdf_rel_path"])
    output_rel_path = get_generated_mount_urdf_rel_path(generator_name, mount_deg, mount_xyz, variant_token)
    output_path = os.path.join(root_dir, output_rel_path)
    source_dir = os.path.dirname(source_path)
    output_dir = os.path.dirname(output_path)

    os.makedirs(output_dir, exist_ok=True)

    with open(source_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_base_static_joint = False
    replaced_origin = False
    current_link_name = None
    dropping_collision = False
    yaw_rad = mount_deg_to_rad(mount_deg)
    rewritten_lines = []

    for line in lines:
        line = _rebase_urdf_resource_paths(line, source_dir, output_dir)
        link_match = re.search(r'<link\s+name="([^"]+)"', line)
        if link_match:
            current_link_name = link_match.group(1)

        if keep_collision_links is not None and current_link_name not in keep_collision_links:
            if "<collision" in line:
                if "</collision>" not in line and "/>" not in line:
                    dropping_collision = True
                continue
            if dropping_collision:
                if "</collision>" in line:
                    dropping_collision = False
                continue

        if f'<joint name="{spec["mount_joint_name"]}"' in line:
            in_base_static_joint = True
            rewritten_lines.append(line)
            continue

        if in_base_static_joint and "<origin" in line:
            indent = re.match(r"\s*", line).group(0)
            xyz_string = " ".join(f"{value:g}" for value in mount_xyz)
            rewritten_lines.append(f'{indent}<origin rpy="0 0 {yaw_rad:.16g}" xyz="{xyz_string}" />\n')
            replaced_origin = True
            continue

        if in_base_static_joint and "</joint>" in line:
            in_base_static_joint = False

        rewritten_lines.append(line)
        if "</link>" in line:
            current_link_name = None

    if not replaced_origin:
        raise RuntimeError(f'Failed to rewrite mount joint "{spec["mount_joint_name"]}" origin in {source_path}')

    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(rewritten_lines)

    return output_rel_path
