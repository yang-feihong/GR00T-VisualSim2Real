from __future__ import annotations

import argparse

import torch


RECORD_FIELDS = ("requested_torque", "applied_torque", "joint_pos", "joint_vel")
LOGICAL_JOINT_METADATA = {
    "initial_joint_pos": ("initial_joint_pos",),
    "initial_joint_vel": ("initial_joint_vel",),
    "default_joint_pos": ("default_joint_pos",),
}
ARTICULATION_JOINT_METADATA = {
    "joint_armature": ("joint_armature",),
    "joint_friction": ("joint_friction",),
    "joint_pos_limits": ("joint_pos_limits", "articulation_joint_pos_limits"),
    "joint_vel_limits": ("joint_vel_limits", "articulation_joint_vel_limits"),
    "joint_effort_limits": ("joint_effort_limits", "articulation_joint_effort_limits"),
}
BODY_METADATA = (
    "initial_body_pos",
    "initial_body_quat",
    "initial_body_lin_vel",
    "initial_body_ang_vel",
    "body_mass",
    "body_inertia",
)


def max_error(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, int]:
    if lhs.shape != rhs.shape:
        raise ValueError(f"shape mismatch: {tuple(lhs.shape)} != {tuple(rhs.shape)}")
    error = (lhs - rhs).abs().flatten()
    index = int(error.argmax())
    return float(error[index]), index


def name_reorder(reference_names, candidate_names, label):
    missing = set(reference_names) - set(candidate_names)
    extra = set(candidate_names) - set(reference_names)
    if missing or extra:
        raise RuntimeError(
            f"{label} names differ: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    candidate_index = {name: index for index, name in enumerate(candidate_names)}
    return torch.tensor([candidate_index[name] for name in reference_names], dtype=torch.long)


def lookup(metadata, aliases):
    for key in aliases:
        if key in metadata:
            return metadata[key]
    raise RuntimeError(f"Trace metadata is missing all aliases: {aliases}")


def reorder_named(value, order, item_count):
    if value.shape[-1] == item_count:
        return value.index_select(-1, order)
    if value.ndim >= 2 and value.shape[-2] == item_count:
        return value.index_select(-2, order)
    raise RuntimeError(
        f"Cannot find named dimension of length {item_count} in shape {tuple(value.shape)}"
    )


def check_field(label, lhs, rhs, atol):
    error, index = max_error(lhs, rhs)
    if error > atol:
        raise RuntimeError(f"Initial metadata diverges at {label}[{index}]: max error {error:.9g}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare deterministic B2Z1 physics traces.")
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument(
        "--allow-intended-position-limit-override",
        action="store_true",
        help="Skip raw USD position-limit equality when the candidate applies physical B2Z1 limits.",
    )
    args = parser.parse_args()

    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    joint_order = name_reorder(reference["dof_names"], candidate["dof_names"], "DOF")
    lhs_meta = reference["metadata"]
    rhs_meta = candidate["metadata"]
    body_order = name_reorder(lhs_meta["body_names"], rhs_meta["body_names"], "body")

    check_field(
        "initial_root_state",
        lhs_meta["initial_root_state"],
        rhs_meta["initial_root_state"],
        args.atol,
    )
    for label, aliases in LOGICAL_JOINT_METADATA.items():
        lhs = lookup(lhs_meta, aliases)
        rhs = lookup(rhs_meta, aliases)
        rhs = reorder_named(rhs, joint_order, len(candidate["dof_names"]))
        check_field(label, lhs, rhs, args.atol)
    lhs_articulation_names = lhs_meta.get("articulation_dof_names", reference["dof_names"])
    rhs_articulation_names = rhs_meta.get("articulation_dof_names", candidate["dof_names"])
    articulation_order = name_reorder(
        lhs_articulation_names, rhs_articulation_names, "articulation DOF"
    )
    for label, aliases in ARTICULATION_JOINT_METADATA.items():
        if label == "joint_pos_limits" and args.allow_intended_position_limit_override:
            continue
        lhs = lookup(lhs_meta, aliases)
        rhs = lookup(rhs_meta, aliases)
        rhs = reorder_named(rhs, articulation_order, len(rhs_articulation_names))
        check_field(label, lhs, rhs, args.atol)
    for label in BODY_METADATA:
        rhs = reorder_named(rhs_meta[label], body_order, len(rhs_meta["body_names"]))
        check_field(label, lhs_meta[label], rhs, args.atol)

    for group_name, field_names in (
        ("policy_alignment", ("proprio", "observation", "action")),
        ("ik_alignment", ("goal_position", "goal_orientation_wxyz", "arm_target")),
    ):
        if group_name not in reference or group_name not in candidate:
            continue
        for field_name in field_names:
            check_field(
                f"{group_name}.{field_name}",
                reference[group_name][field_name],
                candidate[group_name][field_name],
                args.atol,
            )

    first_divergence = None
    for step, (lhs_record, rhs_record) in enumerate(
        zip(reference["records"], candidate["records"], strict=True)
    ):
        summaries = []
        for field in RECORD_FIELDS:
            rhs = rhs_record[field].index_select(-1, joint_order)
            error, index = max_error(lhs_record[field], rhs)
            summaries.append(f"{field}={error:.9g}@{index}")
            if first_divergence is None and error > args.atol:
                first_divergence = (step, field, index, error)
        error, index = max_error(lhs_record["root_state"], rhs_record["root_state"])
        summaries.append(f"root_state={error:.9g}@{index}")
        if first_divergence is None and error > args.atol:
            first_divergence = (step, "root_state", index, error)
        print(f"step {step}: {' '.join(summaries)}")

    if first_divergence is None:
        print(f"No divergence above atol={args.atol:.9g}")
        return

    step, field, index, error = first_divergence
    raise SystemExit(
        f"First divergence: step={step} field={field} index={index} "
        f"error={error:.9g} (atol={args.atol:.9g})"
    )


if __name__ == "__main__":
    main()
