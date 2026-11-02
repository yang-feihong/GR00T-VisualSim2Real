"""Lab30 gain and gravity-feedforward contracts without a simulator launch."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import yaml
from legged_gym.utils.arm_gravity import ArmGravityCompensation
from legged_gym.utils.joint_friction import restore_urdf_friction_efforts

from gr00t.rl.utils.z1_gains import Z1_DAMPING, Z1_STIFFNESS

LOWLEVEL = Path(__file__).parents[1] / "third_party/visual_wholebody_lowlevel/legged_gym"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_default_orientation_correction_does_not_flip_small_error(device):
    from legged_gym.utils.constrained_ik import batched_constrained_dls_delta
    from legged_gym.utils.ee_steady_inverse_gpu import _orientation_error

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    tree = ast.parse((LOWLEVEL / "envs/manip_loco/b2z1_config.py").read_text())
    gain = next(
        ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "teleop_ik_orientation_gain" for t in node.targets)
    )
    angles = torch.tensor([-0.2, -0.07, 0.07, 0.2], device=device, dtype=torch.float64)
    target = torch.zeros((4, 4), device=device, dtype=torch.float64)
    target[:, 0] = torch.cos(angles / 2)
    target[:, 3] = torch.sin(angles / 2)
    rotation = torch.eye(3, device=device, dtype=torch.float64).expand(4, -1, -1)
    residual = torch.zeros((4, 6), device=device, dtype=torch.float64)
    residual[:, 3:] = gain * _orientation_error(target, rotation)
    jacobian = torch.eye(6, device=device, dtype=torch.float64).expand(4, -1, -1)
    joint = torch.zeros_like(residual)
    limits = torch.tensor([[-3.0, 3.0]] * 6, device=device, dtype=torch.float64)
    step = batched_constrained_dls_delta(jacobian, residual, joint, limits, 0.05)
    remaining = angles - step[:, 5]
    assert torch.all(remaining * angles > 0)
    assert torch.all(remaining.abs() < angles.abs() * 0.01)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fixed", [False, True])
def test_compensation_sign_layout_limits_and_buffer_reuse(device, fixed):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    offset = 0 if fixed else 6
    forces = torch.arange(2 * (9 + offset), device=device, dtype=torch.float32).reshape(2, -1) - 8
    robot = NS(
        root_physx_view=NS(get_gravity_compensation_forces=lambda: forces),
        is_fixed_base=fixed,
        num_instances=2,
        num_joints=9,
    )
    ids = torch.tensor([1, 4, 7], device=device)
    controller = ArmGravityCompensation(robot, ids, torch.full((9,), 10.0, device=device))
    torques = torch.full((2, 9), 123.0, device=device)
    controller.apply(torques, True)
    controller.apply(torques, True)
    torch.testing.assert_close(torques[:, ids], forces[:, ids + offset].clamp(-10, 10))
    forces.add_(2.0)
    controller.apply(torques, True)
    torch.testing.assert_close(torques[:, ids], forces[:, ids + offset].clamp(-10, 10))
    controller.apply(torques, False)
    torch.testing.assert_close(torques[:, ids], torch.zeros((2, 3), device=device))
    torch.testing.assert_close(
        torques[:, [0, 2, 3, 5, 6, 8]], torch.full((2, 6), 123.0, device=device)
    )


@pytest.mark.parametrize("forces", [torch.zeros(1, 9), torch.full((1, 15), float("nan"))])
def test_bad_gravity_force_layout_fails_early(forces):
    robot = NS(
        root_physx_view=NS(get_gravity_compensation_forces=lambda: forces),
        is_fixed_base=False,
        num_instances=1,
        num_joints=9,
    )
    with pytest.raises(RuntimeError, match="gravity force layout"):
        ArmGravityCompensation(robot, torch.tensor([3, 4, 5]), torch.ones(9))


def test_imported_friction_is_converted_once_and_only_for_arm():
    legacy = torch.tensor([[0.3, 1.0, 0.0, 2.0, 0.5], [0.3, 1.0, 0.0, 2.0, 0.5]])
    original = legacy.clone()
    props = torch.full((2, 5, 3), 0.1)
    ids = torch.tensor([1, 3, 4])

    def set_legacy(values, rows):
        torch.testing.assert_close(rows, torch.arange(2, dtype=torch.int32))
        legacy.copy_(values)

    def set_efforts(static, dynamic, viscous, joint_ids):
        props[:, joint_ids, 0] = static
        props[:, joint_ids, 1] = dynamic
        props[:, joint_ids, 2] = viscous

    robot = NS(
        device="cpu",
        num_instances=2,
        root_physx_view=NS(
            get_dof_friction_coefficients=lambda: legacy,
            get_dof_friction_properties=lambda: props,
            set_dof_friction_coefficients=set_legacy,
        ),
        write_joint_friction_coefficient_to_sim=set_efforts,
    )
    restore_urdf_friction_efforts(robot, ids)
    torch.testing.assert_close(legacy[:, ids], torch.zeros((2, 3)))
    torch.testing.assert_close(legacy[:, [0, 2]], original[:, [0, 2]])
    torch.testing.assert_close(props[:, ids, 0], original[:, ids] * 0.5)
    torch.testing.assert_close(props[:, ids, 1], original[:, ids] * 0.5)
    torch.testing.assert_close(props[:, ids, 2], torch.full((2, 3), 0.1))
    torch.testing.assert_close(props[:, [0, 2]], torch.full((2, 2, 3), 0.1))
    before = props.clone()
    restore_urdf_friction_efforts(robot, ids)
    torch.testing.assert_close(props, before)


@pytest.mark.parametrize(
    "source",
    [
        LOWLEVEL / "envs/base/legged_robot_config.py",
        Path(__file__).parents[1] / "simulator/isaacsim/isaacsim.py",
    ],
)
def test_actual_actuator_gains_match_lab30_in_both_backends(source):
    tree = ast.parse(source.read_text())
    arm = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and any(
            k.arg == "stiffness"
            and isinstance(k.value, ast.Call)
            and ast.unparse(k.value) == "dict(Z1_STIFFNESS)"
            for k in n.keywords
        )
    )
    values = {
        k.arg: eval(
            compile(ast.Expression(k.value), "gains", "eval"),
            {
                "__builtins__": {},
                "dict": dict,
                "Z1_STIFFNESS": Z1_STIFFNESS,
                "Z1_DAMPING": Z1_DAMPING,
            },
        )
        for k in arm.keywords
        if k.arg in ("stiffness", "damping")
    }
    assert [values["stiffness"][f"joint{i}"] for i in range(1, 7)] == pytest.approx(
        [v / 30 * 25.6 for v in (264, 328, 264, 264, 264, 264)]
    )
    assert [values["damping"][f"joint{i}"] for i in range(1, 7)] == [1.5, 3, 1.5, 1.5, 1.5, 1.5]
    assert values["stiffness"]["jointGripper"] == pytest.approx(100 / 30 * 25.6)
    assert values["damping"]["jointGripper"] == 1


def test_substep_compensation_wiring():
    tree = ast.parse((LOWLEVEL / "envs/manip_loco/manip_loco.py").read_text())
    method = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_apply_action"
    )
    calls = [
        n.func.attr
        for stmt in method.body
        for n in ast.walk(stmt)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]
    assert (
        calls.index("_compute_torques")
        < calls.index("apply")
        < calls.index("set_joint_effort_target")
    )


def test_yaml_and_lowlevel_control_defaults_share_drive_gains():
    control = yaml.safe_load(
        (Path(__file__).parents[1] / "config/robot/b2z1/b2z1.yaml").read_text()
    )["robot"]["control"]
    tree = ast.parse((LOWLEVEL / "envs/manip_loco/b2z1_config.py").read_text())
    cfg = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "ControlCfg"
    )
    namespace = {
        "__builtins__": {},
        "Z1_STIFFNESS": Z1_STIFFNESS,
        "Z1_DAMPING": Z1_DAMPING,
        "B2_stiffness": 360,
        "B2_damping": 5,
    }
    lowlevel = {
        k.arg: eval(compile(ast.Expression(k.value), "defaults", "eval"), namespace)
        for k in cfg.keywords
        if k.arg in ("stiffness", "damping")
    }
    for values in (control, lowlevel):
        for key, expected in (("stiffness", Z1_STIFFNESS), ("damping", Z1_DAMPING)):
            assert {name: values[key][name] for name in expected} == pytest.approx(expected)
    assert control["stiffness"]["hip"] == 360
    assert lowlevel["stiffness"]["FL_hip_joint"] == 360


def test_checkpoint_arm_defaults_cannot_restore_stale_gains_but_legs_are_preserved():
    source = LOWLEVEL / "utils/helpers.py"
    method = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_apply_checkpoint_policy_control_args"
    )
    namespace = {"Z1_STIFFNESS": Z1_STIFFNESS, "Z1_DAMPING": Z1_DAMPING}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    cfg = NS(policy_joint_names=["FL_hip_joint"], default_joint_angles={}, control=NS())
    metadata = {
        "env_cfg": {
            "env": {},
            "init_state": {"default_joint_angles": {"FL_hip_joint": 0.1}},
            "control": {
                "decimation": 4,
                "action_scale": [1],
                "stiffness": {"FL_hip_joint": 371, "joint1": 360},
                "damping": {"FL_hip_joint": 4.2, "joint1": 40},
            },
            "normalization": {
                "clip_actions": 1,
                "clip_observations": 1,
                "obs_scales": dict.fromkeys(("lin_vel", "ang_vel", "dof_pos", "dof_vel"), 1),
            },
        }
    }
    namespace["_apply_checkpoint_policy_control_args"](cfg, metadata)
    assert cfg.control.stiffness["FL_hip_joint"] == 371
    assert cfg.control.damping["FL_hip_joint"] == 4.2
    for name, value in Z1_STIFFNESS.items():
        assert cfg.control.stiffness[name] == value
        assert cfg.control.damping[name] == Z1_DAMPING[name]
