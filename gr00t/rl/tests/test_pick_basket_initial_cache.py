"""Admitted standard paths are installed once, with device-side state validation."""

import math
from types import SimpleNamespace

import pytest
import torch

from gr00t.rl.auto_open.motion_targets import PaddedPathBank
from gr00t.rl.pick import task_initialization, tensor_planning
from gr00t.rl.pick.basket_controller import BasketPickController
from gr00t.rl.pick.controller import PickController
from gr00t.rl.pick.scene import _planning_failure_reason
from gr00t.rl.pick.stages import BasketStage
from gr00t.rl.pick.tensor_fsm import TensorPickFSM
from gr00t.rl.pick.tensor_planning import TensorPickPlanningCoordinator
from gr00t.rl.tests.test_pick_task_initialization import batch_fixture
from gr00t.rl.tests.test_pick_tensor_fsm import batch
from gr00t.rl.tests.test_pick_workflow import fake_controller


def forbidden(*args, **kwargs):
    raise AssertionError(
        "Initial cache installation must not replan or read tensor scalars on the host"
    )


@pytest.mark.parametrize(
    "stage,reason",
    [
        ("CLEAR_FOR_REPOSITION", "no_open_jaw_clear_inverse_path"),
        ("REPOSITION_OBJECT", "no_closed_nudge_inverse_path"),
        ("REOPEN_AFTER_REPOSITION", "post_nudge_no_safe_grasp_candidate"),
    ],
)
def test_batch_planning_failure_summary_uses_stage_diagnostics_not_stale_configuration(
    stage, reason
):
    controller = SimpleNamespace(
        reason=None, planning_diagnostics={stage.lower(): {"failure": reason}}
    )
    assert _planning_failure_reason(controller, stage) == reason
    controller.planning_diagnostics.clear()
    assert _planning_failure_reason(controller, stage) == "stage_entry_planning_failed"


@pytest.mark.parametrize("needs_nudge", [False, True])
@pytest.mark.parametrize("stale_cache", [False, True, "probe_fallback"])
def test_shared_admission_cache_reaches_actual_controller_and_fsm_without_replanning(
    monkeypatch, tmp_path, needs_nudge, stale_cache
):
    runner, episodes, _, _, _ = batch_fixture(monkeypatch)
    admission = task_initialization.basket_grasp_access_batch
    calls = []

    def plan(state, *args, **kwargs):
        calls.append(len(state["root_states"]))
        accepted, report, path = admission(state, *args, **kwargs)
        if needs_nudge:
            report["direct_path_available"][1] = False
            report["initial_reposition_required"][1] = True
        return accepted, report, path

    monkeypatch.setattr(task_initialization, "basket_grasp_access_batch", plan)
    state, views, collision, checks, geometry = task_initialization.admit_pick_batch(
        runner, episodes, [object(), object()], tmp_path
    )
    controls = []
    for row, episode in enumerate(episodes):
        cache = geometry[row].initial_grasp_plan
        if cache is not None:
            cache["diagnostics"]["candidate_cost"] = torch.arange(row + 2).float()[None]
            cache["diagnostics"]["candidate_source_indices"] = torch.arange(row + 2)[None]
        episode["base_yaw"] = 0.0
        control = BasketPickController(
            views[row],
            episode,
            geometry[row],
            collision=collision,
            object_support_points=geometry[row].object_support_points,
            initialize=False,
        )
        control.current_stage = (
            BasketStage.OPEN_FOR_REPOSITION
            if episode.get("basket_initial_reposition_required")
            else BasketStage.MOVE_TO_GRASP
        )
        control.stage.fill_(int(control.current_stage))
        controls.append(control)
    fsm = TensorPickFSM(runner.env, controls)
    planner = TensorPickPlanningCoordinator(controls)
    assert planner.basket_geometry is geometry[0].basket_geometry_factory
    refreshed = []

    def refresh(local, *args, **kwargs):
        refreshed.append(len(local["root_states"]))
        assert not kwargs["realized_body"].any() and not kwargs["prefer_realized_body"].any()
        p = (local["ee_pos"] + 0.03)[:, None].expand(-1, 4096, -1)
        q = local["ee_orn"][:, None].expand(-1, 4096, -1)
        valid = not (stale_cache == "probe_fallback" and "candidate_orientations" in kwargs)
        return (
            (p, q, torch.tensor([3])),
            torch.tensor([valid]),
            {
                "selected": torch.tensor([0 if valid else -1]),
                "entry_orientation_world_wxyz": local["ee_orn"],
            },
        )

    if stale_cache:
        state["dof_pos"][0, 0] += 0.0001
        monkeypatch.setattr(tensor_planning, "plan_basket_grasp_batch", refresh)
    else:
        monkeypatch.setattr(planner, "plan_basket_grasp", forbidden)
    planner.install_initial_basket_plans(fsm, state)
    assert calls == [2] and all(check["accepted"] for check in checks)
    assert fsm.arm_path.length.tolist() == [3, 1 if needs_nudge else 3]
    direct = torch.tensor([True, not needs_nudge])
    expected = state["ee_pos"] + 0.02
    if stale_cache:
        expected[0] += 0.01
    torch.testing.assert_close(fsm.arm_path.positions[direct, 2], expected[direct])
    torch.testing.assert_close(fsm.awaiting_entry_refresh, direct)
    assert refreshed == ([1, 1] if stale_cache == "probe_fallback" else [1] if stale_cache else [])
    assert fsm.nudge_attempts[0] == 0
    assert fsm.stage[0] == int(BasketStage.MOVE_TO_GRASP)
    assert not fsm.grasp_retries.any()
    planner.finish_diagnostics()
    if stale_cache:
        assert controls[0].planning_diagnostics["grasp"]["initial_admission_cache_refreshed"]
    else:
        assert controls[0].planning_diagnostics["grasp"]["reused_initial_admission_plan"]
        diagnostic = controls[0].planning_diagnostics["grasp"]
        assert diagnostic["candidate_cost"][:2] == [0, 1]
        if not needs_nudge:
            assert math.isnan(diagnostic["candidate_cost"][2])
            assert diagnostic["candidate_source_indices"] == [0, 1, -1]
    if needs_nudge:
        assert (
            controls[1].planning_diagnostics["reposition_attempts"][0]["trigger"]
            == "initial_layout_requires_reachable_grasp"
        )
        assert fsm.nudge_attempts.tolist() == [0, 1]
    else:
        assert controls[1].planning_diagnostics["grasp"]["reused_initial_admission_plan"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_initial_orientation_probe_fallback_only_replans_failed_rows(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU initial refresh")
    planner = TensorPickPlanningCoordinator.__new__(TensorPickPlanningCoordinator)
    planner.grasp_metadata = True
    planner.basket_geometry = SimpleNamespace(update=lambda state: "geometry")
    fsm = SimpleNamespace(
        device=device,
        stage_entry_from=torch.full((3,), int(BasketStage.APPROACH_OBJECT), device=device),
        grasp_retries=torch.zeros(3, device=device, dtype=torch.long),
        tried_grasp_mask=torch.zeros(3, 8, device=device, dtype=torch.bool),
        basket_final_entry=torch.zeros(3, device=device, dtype=torch.bool),
        basket_entry_depth_fraction=torch.full((3,), torch.nan, device=device),
    )
    calls = []

    def plan(fsm, state, rows, geometry, final_entry, **kwargs):
        calls.append((rows.clone(), kwargs))
        return rows != 1

    monkeypatch.setattr(planner, "_plan_basket_grasp_group", plan)
    rows = torch.tensor([2, 0, 1], device=device)
    orientations = torch.eye(4, device=device)[:3]
    planner.plan_basket_grasp(
        fsm, {}, rows, initial_refresh=True, initial_orientations=orientations
    )
    assert len(calls) == 2 and calls[0][1]["probe_only"]
    torch.testing.assert_close(calls[0][0], rows)
    torch.testing.assert_close(calls[0][1]["candidate_orientations"], orientations[:, None])
    torch.testing.assert_close(calls[1][0], rows.new_tensor([1]))
    assert not fsm.grasp_retries.any() and not fsm.tried_grasp_mask.any()


@pytest.mark.parametrize("basket", [False, True])
def test_configuration_only_construction_never_plans_rejected_or_preplanned_rows(
    monkeypatch, basket
):
    original, state = fake_controller(monkeypatch)
    for cls in (PickController, BasketPickController):
        for name in ("prepare_approach", "transition", "begin_reposition"):
            assert not hasattr(cls, name)
    cls = BasketPickController if basket else PickController
    geometry = SimpleNamespace(basket_model=object()) if basket else None
    control = cls(
        original.env, original.episode, geometry, collision=original.collision, initialize=False
    )
    assert control.current_stage == BasketStage.APPROACH_OBJECT
    assert control.approach_gaze_valid and control.paths is None
    assert control.arm_path.length.tolist() == [1]
    torch.testing.assert_close(control.arm_path.positions[:, 0], state["ee_pos"])


def test_initial_cache_routes_direct_nudge_and_rejected_rows_without_replanning(monkeypatch):
    env, controls, state = batch(monkeypatch, count=3)
    state["distractor_states"] = torch.zeros(3, 0, 13)
    model = SimpleNamespace(boxes=torch.tensor([[[0.0, 0, 0.2]] * 8]))
    for control in controls:
        control.basket, control.lift_margin = model, 0.1
        control.object_support_points = torch.zeros(8, 3)
        control.geometry = SimpleNamespace()
        control.initial_grasp_plan = None
    controls[0].current_stage = BasketStage.MOVE_TO_GRASP
    controls[1].current_stage = BasketStage.OPEN_FOR_REPOSITION
    p = state["ee_pos"][:1, None] + torch.tensor(
        [[[0.0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [0.02, 0, 0]]]
    )
    q = state["ee_orn"][:1, None].expand(-1, 4, -1)
    controls[0].initial_grasp_plan = {
        "paths": {"approach": (p, q, torch.tensor([3]))},
        "state": {
            key: state[key][:1].clone()
            for key in ("root_states", "dof_pos", "object_state", "distractor_states")
        },
        "diagnostics": {"selected": torch.tensor([0])},
    }
    fsm = TensorPickFSM(env, controls)
    fsm.fail(fsm.rows[2:], 10)
    planner = TensorPickPlanningCoordinator(controls)
    monkeypatch.setattr(planner, "plan_basket_grasp", forbidden)
    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        planner.install_initial_basket_plans(fsm, state)
    assert fsm.awaiting_entry_refresh.tolist() == [True, False, False]
    assert fsm.arm_path.length.tolist()[:2] == [3, 1]
    assert fsm.nudge_attempts.tolist() == [0, 1, 0]
    assert fsm.failure_code.tolist() == [0, 0, 10]
    torch.testing.assert_close(fsm.arm_path.positions[0, :3], p[0, :3])
    assert controls[0].initial_grasp_plan is None
    planner.finish_diagnostics()
    assert controls[0].planning_diagnostics["grasp"]["reused_initial_admission_plan"]
    assert (
        controls[1].planning_diagnostics["reposition_attempts"][0]["trigger"]
        == "initial_layout_requires_reachable_grasp"
    )


@pytest.mark.parametrize("changed", [False, True])
def test_changed_cache_is_not_consumed_and_missing_plans_are_rejected_before_control(
    monkeypatch, changed
):
    env, controls, state = batch(monkeypatch, count=1)
    state["distractor_states"] = torch.zeros(1, 0, 13)
    controls[0].current_stage = BasketStage.MOVE_TO_GRASP
    controls[0].geometry = SimpleNamespace()
    controls[0].initial_grasp_plan = None
    planner = TensorPickPlanningCoordinator(controls)
    if changed:
        controls[0].initial_grasp_plan = {
            "state": {
                key: state[key].clone()
                for key in ("root_states", "dof_pos", "object_state", "distractor_states")
            }
        }
        state["dof_pos"] += 0.001
    fsm = SimpleNamespace(
        device="cpu",
        payload_mask=torch.ones(1, 1, dtype=torch.bool),
        stage_entry_from=torch.tensor([int(BasketStage.MOVE_TO_GRASP)]),
        tried_grasp_mask=torch.ones(1, 8, dtype=torch.bool),
    )
    if not changed:
        with pytest.raises(ValueError, match="missing its verified"):
            planner.install_initial_basket_plans(fsm, state)
    else:
        replanned = []
        monkeypatch.setattr(
            planner,
            "plan_basket_grasp",
            lambda f, s, rows, **kw: replanned.append((rows.clone(), kw)),
        )
        planner.install_initial_basket_plans(fsm, state)
        assert len(replanned) == 1 and replanned[0][0].tolist() == [0]
        assert replanned[0][1]["initial_refresh"]
        assert fsm.stage_entry_from.tolist() == [int(BasketStage.APPROACH_OBJECT)]
        assert not fsm.tried_grasp_mask.any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU initial cache")
def test_gpu_cache_validation_ignores_absent_slots_without_host_reads(monkeypatch):
    root = torch.zeros(2, 13, device="cuda")
    root[:, 3], root[1, 0] = 1, 32
    props = root[:, None].expand(-1, 2, -1).clone()
    props[1, 1] = torch.nan
    state = {
        "root_states": root,
        "object_state": root.clone(),
        "dof_pos": root.new_zeros(2, 6),
        "distractor_states": props,
    }
    q = root[:, None, 3:7].expand(-1, 5, -1)
    p = root[:, None, :3] + root.new_tensor(
        [[[0.2, 0, 0.7], [0.3, 0, 0.8], [0.4, 0, 0.9], [0.4, 0, 0.9], [0.4, 0, 0.9]]]
    )
    controls = [
        SimpleNamespace(
            current_stage=BasketStage.MOVE_TO_GRASP,
            basket=object(),
            geometry=SimpleNamespace(),
            initial_grasp_plan={
                "state": {key: value[row : row + 1].clone() for key, value in state.items()},
                "paths": {
                    "approach": (
                        p[row : row + 1],
                        q[row : row + 1],
                        torch.tensor([3], device="cuda"),
                    )
                },
                "diagnostics": {
                    "selected": torch.tensor([row], device="cuda"),
                    "selected_grasp_depth_fraction": root.new_tensor([0.15 + row * 0.2]),
                },
            },
        )
        for row in range(2)
    ]
    fsm = SimpleNamespace(
        device="cuda",
        arm_path=PaddedPathBank(2, 16, "cuda"),
        payload_mask=torch.tensor([[True, True, True], [True, True, False]], device="cuda"),
        awaiting_entry_refresh=torch.zeros(2, device="cuda", dtype=torch.bool),
        basket_entry_orientation=root.new_zeros(2, 4),
        basket_entry_object_orientation=root.new_zeros(2, 4),
        basket_entry_depth_fraction=root.new_full((2,), torch.nan),
        tried_grasp_orientations=root.new_zeros(2, 8, 4),
        tried_grasp_mask=torch.zeros(2, 8, device="cuda", dtype=torch.bool),
    )
    planner = TensorPickPlanningCoordinator(controls)
    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist", "__int__", "__float__"):
            scope.setattr(torch.Tensor, name, forbidden)
        planner.install_initial_basket_plans(fsm, state)
    torch.testing.assert_close(fsm.arm_path.positions[:, :3], p[:, :3])
    assert fsm.awaiting_entry_refresh.all() and fsm.tried_grasp_mask[:, 0].all()
    torch.testing.assert_close(fsm.basket_entry_depth_fraction, root.new_tensor([0.15, 0.35]))
