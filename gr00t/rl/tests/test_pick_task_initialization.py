"""Scene admission uses the same batch stepper and preserves scene-only plans."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gr00t.rl.pick import task_initialization as initial
from gr00t.rl.pick.basket import NoValidBasketLayout
from gr00t.rl.tests.test_pick_workflow import fake_controller


def fixture(monkeypatch, *, layout_valid):
    class Validator:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, data, preferred, report, rows, joints):
            return preferred, report

    monkeypatch.setattr(initial, "BasketLayoutPathValidator", Validator)
    config, state = fake_controller(monkeypatch)
    state["distractor_states"] = torch.zeros(1, 0, 13)
    state["contact_forces_w"] = torch.zeros(1, 1, 3)
    env = config.env
    env.gripper_control_options = {}
    env.num_actions = 18
    env.scene = SimpleNamespace(env_origins=torch.zeros(1, 3))
    env.contact_sensor = SimpleNamespace(body_names=["base_link"])
    env._clip_teleop_targets = lambda: None
    env._update_effective_teleop_inputs = lambda **kwargs: None
    resets, calls = [], []
    env._reset_idx = lambda rows: resets.append(rows.tolist())
    env.get_observations = lambda: torch.zeros(env.num_envs, 18)
    env.admission_reset_events = []
    env.obs_history_buf = torch.zeros(1, 15, 18)
    env.scene.write_data_to_sim = lambda: env.admission_reset_events.append("write")
    env.sim = SimpleNamespace(forward=lambda: env.admission_reset_events.append("forward"))
    env.synchronize_rgb_cameras_after_reset = lambda: env.admission_reset_events.append("cameras")

    def reset():
        env.admission_reset_events.append("reset")
        env._reset_idx(torch.arange(env.num_envs))
        env.admission_reset_events.append("forward")
        return {"policy": env.get_observations()}, {}

    env.reset = reset

    def step():
        calls.append("physics")
        return {"state": state, "low_level_action": torch.zeros(1, 18)}

    runner = SimpleNamespace(env=env, step_once=step)
    basket = SimpleNamespace(boxes=torch.zeros(1, 8, 3))
    monkeypatch.setattr(initial.BasketGeometry, "from_robot_usd", lambda **kwargs: basket)
    monkeypatch.setattr(
        initial, "read_arm_joint_limits", lambda *args: np.zeros((6, 2), dtype=np.float64)
    )
    monkeypatch.setattr(initial, "BasketArmCollisionSamples", lambda value: value)
    monkeypatch.setattr(
        initial,
        "basket_object_checks_batch",
        lambda basket, state, *args: (
            {"layout_valid": torch.full((len(state["root_states"]),), layout_valid)},
            {},
        ),
    )

    def metadata(episodes, slots, device):
        count = len(episodes)
        return SimpleNamespace(
            hulls=(
                torch.zeros(count, slots, 1, 3),
                torch.ones(count, slots, 1, dtype=torch.bool),
                torch.ones(count, slots, dtype=torch.bool),
            ),
            sizes=torch.full((count, 3), 0.04),
            payload_sizes=torch.full((count, slots, 3), 0.04),
            support=torch.zeros(count, 1, 3),
            support_mask=torch.ones(count, 1, dtype=torch.bool),
            supports=[torch.zeros(1, 3) for _ in episodes],
            centers=torch.zeros(count, 3),
            lift_margin=torch.full((count,), 0.1),
        )

    monkeypatch.setattr(initial, "BasketAdmissionInputs", metadata)

    class Geometry:
        basket_model = basket

        def select(self, rows):
            return Geometry()

    class GeometryFactory:
        def __init__(self, controllers):
            pass

        def update(self, state):
            return Geometry()

        def with_target(self, state):
            return Geometry(), Geometry()

    monkeypatch.setattr(initial, "BasketGeometryBatch", GeometryFactory)

    def collision(env):
        assert len(calls) >= 25
        return object()

    monkeypatch.setattr(initial, "ArmCollisionSamples", collision)

    def access(state, *args, **kwargs):
        assert args[11].dtype == state["root_states"].dtype
        count = len(state["root_states"])
        path = (
            (state["ee_pos"] + 0.02)[:, None].expand(-1, 4096, -1),
            state["ee_orn"][:, None].expand(-1, 4096, -1),
            torch.full((count,), 3),
        )
        return (
            torch.ones(count, dtype=torch.bool),
            {
                "direct_path_available": torch.ones(count, dtype=torch.bool),
                "initial_reposition_required": torch.zeros(count, dtype=torch.bool),
                "direct_full_path.selected": torch.zeros(count, dtype=torch.long),
                "direct_full_path.entry_orientation_world_wxyz": state["ee_orn"],
                "direct_full_path.required_object_lift_m": 0.1,
            },
            path,
        )

    monkeypatch.setattr(initial, "basket_grasp_access_batch", access)
    episode = {
        "seed": 1700,
        "task_type": "pick_basket",
        "scene_id": "room",
        "base_xy": [0, 0],
        "station_xy": [0, 0],
        "initial_arm_joints": [0] * 6,
        "object_position": [0.65, 0, 0.03],
        "support_z": 0.0,
        "object_yaw": 0.0,
        "asset": {"name": "object", "usd_path": "unused", "size_m": [0.04] * 3},
        "distractors": [],
        "basket_sampling": {},
    }
    return runner, episode, calls, resets


def test_basket_payload_can_settle_past_eighty_steps_without_resampling(monkeypatch, tmp_path):
    runner, episode, calls, resets = fixture(monkeypatch, layout_valid=True)
    state = runner.env.get_control_state()
    original = runner.step_once

    def step():
        result = original()
        state["object_state"][:, 7] = 0.10 if len(calls) <= 82 else 0.0
        if len(calls) <= 82:
            state["object_state"][:, 0] += 0.002
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError("A supported payload still settling must not be resampled at step 80")

    runner.step_once = step
    monkeypatch.setattr(initial, "resample_basket_payloads", forbidden)
    _, _, _, checks, _ = initial.admit_pick_batch(runner, [episode], [object()], tmp_path)
    assert checks[0]["accepted"] and checks[0]["attempt"] == 0
    assert len(calls) == 96 and resets == [[0]]
    history = json.loads((tmp_path / "seed_1700/initial_attempts.json").read_text())
    assert len(history) == 1 and history[0]["checks"]["all_objects_stable_15_frames"]
    snapshot = history[0]["initial_acceptance_values"]
    assert snapshot["actual_joint_positions_rad"] == state["dof_pos"][0].tolist()
    assert snapshot["ee_position_world_m"] == state["ee_pos"][0].tolist()
    assert snapshot["ee_orientation_world_wxyz"] == state["ee_orn"][0].tolist()
    assert snapshot["initial_arm_joint_target_rad"] == episode["initial_arm_joints"]


def test_admission_preserves_static_room_and_reuses_the_validated_grasp(monkeypatch, tmp_path):
    runner, episode, calls, resets = fixture(monkeypatch, layout_valid=True)
    room = object()
    _, _, _, checks, planning = initial.admit_pick_batch(runner, [episode], [room], tmp_path)
    assert checks[0]["accepted"]
    assert runner.env.pick_initial_admitted.tolist() == [True]
    assert len(calls) == 32 and resets == [[0]]
    assert runner.env.admission_reset_events == ["reset", "forward"]
    assert runner.obs.shape == (1, 18)
    assert planning[0].basket_static_scene is room
    path = planning[0].initial_grasp_plan["paths"]["approach"]
    assert path[2].tolist() == [3]
    torch.testing.assert_close(path[0][0, 2], runner.env.get_control_state()["ee_pos"][0] + 0.02)
    assert not hasattr(initial, "basket_grasp_access")


def test_layout_hint_is_rotated_to_world_and_revalidated(monkeypatch, tmp_path):
    runner, episode, _, _ = fixture(monkeypatch, layout_valid=True)
    episode["base_yaw"] = -np.pi / 2
    physics_step = runner.step_once

    def turn_during_warmup():
        result = physics_step()
        result["state"]["root_states"][:, 3:7] = torch.tensor([2**-0.5, 0, 0, 2**-0.5])
        return result

    runner.step_once = turn_during_warmup
    episode["basket_sampling"] = {
        "reachable_proposals": 1,
        "reachability_fallback": False,
        "grasp_seed_orientation_local_wxyz": [2**-0.5, 0, 2**-0.5, 0],
    }
    original, calls = initial.basket_grasp_access_batch, []

    def access(state, *args, **kwargs):
        calls.append(kwargs)
        return original(state, *args, **kwargs)

    monkeypatch.setattr(initial, "basket_grasp_access_batch", access)
    _, _, _, checks, _ = initial.admit_pick_batch(runner, [episode], [object()], tmp_path)
    assert len(calls) == 1 and calls[0]["direct_only"]
    torch.testing.assert_close(
        calls[0]["candidate_orientations"][0, 0], torch.tensor([0.5, -0.5, 0.5, 0.5])
    )
    assert checks[0]["cached_grasp_revalidation"]["direct_path_available"]


def test_basket_admission_uses_pose_window_and_keeps_raw_twists(monkeypatch, tmp_path):
    runner, episode, _, _ = fixture(monkeypatch, layout_valid=True)
    state = runner.env.get_control_state()
    state["root_states"][:, 7] = 0.04
    state["root_states"][:, 12] = 0.2
    displacement = state["object_state"][:, :3] - state["root_states"][:, :3]
    state["object_state"][:, 7:10] = state["root_states"][:, 7:10] + torch.cross(
        state["root_states"][:, 10:13], displacement, dim=-1
    )
    state["object_state"][:, 10:13] = state["root_states"][:, 10:13]
    _, _, _, checks, _ = initial.admit_pick_batch(
        runner, [episode], [object()], tmp_path, attempts=1
    )
    assert checks[0]["accepted"]
    assert checks[0]["checks"]["all_objects_stable_15_frames"]
    values = checks[0]["initial_acceptance_values"]
    assert values["payload_angular_speeds_world_rad_s"][0] == pytest.approx(0.2)
    assert values["payload_angular_speeds_relative_to_base_rad_s"][0] == pytest.approx(0)
    assert values["payload_linear_speeds_relative_to_base_m_s"][0] == pytest.approx(0)
    assert values["object_stable_frames"] == 15
    assert values["payload_position_spans_relative_to_base_m"][0] < 0.001


def test_basket_pose_window_rejects_real_sliding(monkeypatch, tmp_path):
    runner, episode, _, _ = fixture(monkeypatch, layout_valid=True)
    state = runner.env.get_control_state()
    original = runner.step_once

    def step():
        result = original()
        state["object_state"][:, 0] += 0.002
        return result

    runner.step_once = step
    _, _, _, checks, _ = initial.admit_pick_batch(
        runner, [episode], [object()], tmp_path, attempts=1
    )
    assert not checks[0]["accepted"]
    values = checks[0]["initial_acceptance_values"]
    assert values["object_stable_frames"] == 0
    assert values["payload_position_spans_relative_to_base_m"][0] > 0.001


@pytest.mark.parametrize("known_rejection", [True, False])
def test_resample_rejections_do_not_reset_physics_or_hide_unknown_errors(
    monkeypatch, tmp_path, known_rejection
):
    runner, episode, calls, resets = fixture(monkeypatch, layout_valid=False)

    def resample(*args, **kwargs):
        if known_rejection:
            raise NoValidBasketLayout("layout exhausted", {"object_count": 4})
        raise RuntimeError("unexpected reset error")

    monkeypatch.setattr(initial, "resample_basket_payloads", resample)
    if not known_rejection:
        with pytest.raises(RuntimeError, match="unexpected reset error"):
            initial.admit_pick_batch(runner, [episode], [object()], tmp_path, attempts=3)
        return
    _, _, _, checks, _ = initial.admit_pick_batch(
        runner, [episode], [object()], tmp_path, attempts=3
    )
    assert not checks[0]["accepted"]
    assert len(calls) == 32 and resets == [[0]]
    history = json.loads((tmp_path / "seed_1700/initial_attempts.json").read_text())
    assert [item["attempt"] for item in history] == [0, 1, 2]
    assert history[-1]["details"] == {"object_count": 4}


def batch_fixture(monkeypatch):
    runner, episode, calls, resets = fixture(monkeypatch, layout_valid=True)
    env = runner.env
    original = env.get_control_state()
    state = {key: value.expand(2, *value.shape[1:]).clone() for key, value in original.items()}
    env.num_envs = 2
    for key, value in vars(env).copy().items():
        if (
            key not in ("arm_joint_ids", "gripper_joint_ids", "dof_pos_limits")
            and torch.is_tensor(value)
            and value.ndim
            and value.shape[0] == 1
        ):
            setattr(env, key, value.expand(2, *value.shape[1:]).clone())
    env.scene.env_origins = torch.zeros(2, 3)
    env.get_control_state = lambda: state
    env.get_observations = lambda: torch.zeros(2, 18)

    def step():
        calls.append("physics")
        return {"state": state, "low_level_action": torch.zeros(2, 18)}

    runner.step_once = step
    other = copy.deepcopy(episode)
    other["seed"] = 1701
    return runner, [episode, other], state, calls, resets


def test_external_admission_submits_support_checks_once_and_keeps_failure_metrics(
    monkeypatch, tmp_path
):
    runner, episodes, state, calls, resets = batch_fixture(monkeypatch)
    state["root_states"][:, 0] = -1
    for row, episode in enumerate(episodes):
        episode["task_type"] = "pick_external"
        episode["scene_usd_path"] = "room.usd"
        episode["support_type"] = "floor" if row == 0 else "table"
    queries = []
    robot_updates, visibility_queries = [], []

    class Room:
        device = torch.device("cpu")

        def visible(self, origins, ends):
            visibility_queries.append(origins.shape)
            return torch.ones(origins.shape[:-1], dtype=torch.bool)

        def select(self, rows):
            return self

        def rays(self, origins, ends):
            queries.append(origins.shape)
            hits = origins.clone()
            hits[..., 2] = 0
            normals = torch.zeros_like(origins)
            normals[..., 2] = 1
            self.last_hit_types = torch.zeros(origins.shape[:-1], dtype=torch.long)
            return hits, normals, origins[..., 2]

    room = Room()
    room.source = room
    monkeypatch.setattr(initial.BatchedSceneGeometry, "from_views", lambda rooms: room)

    class RobotVisibility:
        def __init__(self, env):
            assert len(calls) == 32

        def update(self, bodies):
            robot_updates.append(bodies.shape)
            return room

    runner.env.robot = SimpleNamespace(
        data=SimpleNamespace(body_state_w=state["root_states"][:, None])
    )
    monkeypatch.setattr(initial, "RobotOcclusionGeometryBatch", RobotVisibility)
    _, _, _, checks, plans = initial.admit_pick_batch(
        runner, episodes, [room, room], tmp_path, attempts=1
    )
    assert len(calls) == 32 and resets == [[0, 1]]
    assert queries == [torch.Size([2, 1, 3])]
    assert visibility_queries == [torch.Size([2, 3]), torch.Size([2, 3])]
    assert robot_updates == [torch.Size([2, 1, 13])]
    assert checks[0]["accepted"] and not checks[1]["accepted"]
    assert not checks[1]["checks"]["object_support_labels_match"]
    for check in checks:
        assert check["actual_support_type_ids"] == [[0]]
        assert check["object_support_error_m"] == [[0.0]]
        assert check["all_object_speeds_m_s"] == [[0.0]]
        assert check["all_object_angular_speeds_rad_s"] == [[0.0]]
        assert check["distractor_states"] == [[]]
        values = check["initial_acceptance_values"]
        assert values["nonfoot_contact_force_n"] == 0
        assert values["target_support_error_m"] == 0
        assert values["base_stable_frames"] >= 25
        assert values["root_state_world"][0] == -1
    assert plans == [room, room]
    assert plans[0].external_geometry_factory.rooms is room
    assert checks[0]["initial_visibility"]["robot_not_occluding"]
    assert not hasattr(initial, "scene_object_checks")


def test_ready_row_is_planned_before_last_partner_and_all_rows_keep_their_gate(
    monkeypatch, tmp_path
):
    runner, episodes, state, calls, resets = batch_fixture(monkeypatch)
    stability = initial.update_robot_stability

    def delayed(*args, **kwargs):
        stable, steps, diagnostics = stability(*args, **kwargs)
        if len(calls) < 150:
            stable[1] = False
        return stable, steps, diagnostics

    monkeypatch.setattr(initial, "update_robot_stability", delayed)
    _, _, _, checks, _ = initial.admit_pick_batch(
        runner, episodes, [object(), object()], tmp_path, attempts=1
    )
    assert len(calls) == 160
    assert resets == [[0, 1]]
    assert all(check["accepted"] for check in checks)
    assert [check["settle_steps"] for check in checks] == [32, 160]
    assert [check["shared_physics_step"] for check in checks] == [32, 160]
    values = [check["initial_acceptance_values"] for check in checks]
    assert [value["warmup_pose_stable_first_step"] for value in values] == [25, 150]
    assert [value["warmup_pose_stable_frames_total"] for value in values] == [8, 11]
    assert [value["warmup_elapsed_physics_steps"] for value in values] == [32, 160]


def test_failed_ready_row_retries_while_partner_is_still_settling(monkeypatch, tmp_path):
    runner, episodes, state, calls, resets = batch_fixture(monkeypatch)
    stability = initial.update_robot_stability

    def delayed(*args, **kwargs):
        stable, steps, detail = stability(*args, **kwargs)
        if len(calls) < 200:
            stable[1] = False
        return stable, steps, detail

    monkeypatch.setattr(initial, "update_robot_stability", delayed)
    monkeypatch.setattr(
        initial,
        "basket_object_checks_batch",
        lambda *args: (
            {"layout_valid": torch.tensor([episodes[0].get("round", 0) >= 2, True])},
            {},
        ),
    )
    retries = []

    def resample(episodes, rows, basket, attempt, **kwargs):
        retries.append(len(calls))
        return {
            row: {**episodes[row], "round": int(attempt[index])} for index, row in enumerate(rows)
        }

    monkeypatch.setattr(initial, "resample_basket_payloads", resample)
    _, _, _, checks, _ = initial.admit_pick_batch(
        runner, episodes, [object(), object()], tmp_path, attempts=3
    )
    assert retries == [32, 64]
    assert resets == [[0, 1], [0], [0]]
    assert all(check["accepted"] for check in checks)
    assert [check["attempt"] for check in checks] == [2, 0]
    assert [check["shared_physics_step"] for check in checks] == [96, 224]
    assert len(calls) == 224


def test_unstable_row_exhausts_only_the_original_physical_and_attempt_budgets(
    monkeypatch, tmp_path
):
    runner, episode, calls, resets = fixture(monkeypatch, layout_valid=True)
    episodes = [episode]
    state = runner.env.get_control_state()
    state["root_states"][:, 2] = 0.4
    monkeypatch.setattr(
        initial,
        "resample_basket_payloads",
        lambda episodes, rows, basket, attempt, **kwargs: {
            row: dict(episodes[row]) for row in rows
        },
    )
    _, _, _, checks, _ = initial.admit_pick_batch(
        runner, episodes, [object()], tmp_path, attempts=2
    )
    assert len(calls) == 600 and resets == [[0], [0]]
    assert not checks[0]["accepted"] and checks[0]["attempt"] == 1
    assert checks[0]["settle_steps"] == 300 and checks[0]["shared_physics_step"] == 600
    assert not checks[0]["checks"]["robot_stable_25_frames"]
    history = json.loads((tmp_path / "seed_1700/initial_attempts.json").read_text())
    assert [item["attempt"] for item in history] == [0, 1]


@pytest.mark.parametrize("component,delta", [(0, 0.01), (0, 0.0002)])
@pytest.mark.parametrize("cached_rejected", [False, True])
def test_accepted_rows_are_not_reset_but_revalidated_if_waiting_changes_their_pose(
    monkeypatch, tmp_path, component, delta, cached_rejected
):
    runner, episodes, state, calls, resets = batch_fixture(monkeypatch)
    step = runner.step_once

    def drifting_second_attempt():
        if len(calls) >= 32:
            state["root_states"][0, component] = delta
        return step()

    runner.step_once = drifting_second_attempt
    monkeypatch.setattr(
        initial,
        "basket_object_checks_batch",
        lambda basket, state, *args: (
            {
                "layout_valid": torch.tensor(
                    [
                        episode["seed"] == 1700 or episode.get("resampled", False)
                        for episode in episodes
                    ]
                )
            },
            {},
        ),
    )
    monkeypatch.setattr(
        initial,
        "resample_basket_payloads",
        lambda episodes, rows, basket, attempt, **kwargs: {
            row: {**episodes[row], "resampled": True} for row in rows
        },
    )
    calls_access = []
    access = initial.basket_grasp_access_batch

    def planned(local, *args, **kwargs):
        calls_access.append(len(local["root_states"]))
        accepted, report, path = access(local, *args, **kwargs)
        if cached_rejected and kwargs.get("direct_only", False):
            accepted.fill_(False)
            report["direct_path_available"].fill_(False)
        return accepted, report, path

    monkeypatch.setattr(initial, "basket_grasp_access_batch", planned)
    _, _, _, checks, plans = initial.admit_pick_batch(
        runner, episodes, [object(), object()], tmp_path
    )
    assert all(check["accepted"] for check in checks)
    assert resets == [[0, 1], [1]]
    assert len(calls) == 64
    history = json.loads((tmp_path / "seed_1700/initial_attempts.json").read_text())
    assert len(history) == (2 if delta > 0.001 else 1)
    expected_access = [1, 1]
    if delta > 0.001:
        expected_access += [1, 1] if cached_rejected else [1]
    assert calls_access == expected_access
    if delta > 0.001:
        assert (
            checks[0]["cached_grasp_revalidation"]["direct_path_available"] is not cached_rejected
        )
        torch.testing.assert_close(
            plans[0].initial_grasp_plan["state"]["root_states"], state["root_states"][:1]
        )
    else:
        assert (
            plans[0].initial_grasp_plan["state"]["root_states"][0, 0] != state["root_states"][0, 0]
        )


def test_physical_basket_checks_are_submitted_once_for_the_complete_shared_batch(
    monkeypatch, tmp_path
):
    runner, episodes, state, calls, resets = batch_fixture(monkeypatch)
    check = initial.basket_object_checks_batch
    sizes = []

    def checked(basket, full_state, *args):
        assert full_state is state
        sizes.append(len(full_state["root_states"]))
        return check(basket, full_state, *args)

    monkeypatch.setattr(initial, "basket_object_checks_batch", checked)
    _, _, _, checks, _ = initial.admit_pick_batch(runner, episodes, [object(), object()], tmp_path)
    assert sizes == [2]
    assert all(row["accepted"] for row in checks)
    assert len(calls) == 32
    assert not hasattr(initial, "basket_object_checks")


def test_layout_retry_keeps_stable_robot_but_restarts_payload_stability(monkeypatch, tmp_path):
    runner, episodes, state, calls, resets = batch_fixture(monkeypatch)
    payload_resets = []
    runner.env.reset_payload_rows = lambda rows: payload_resets.append(rows.clone())
    monkeypatch.setattr(
        initial,
        "basket_object_checks_batch",
        lambda basket, state, *args: (
            {"layout_valid": torch.tensor([True, episodes[1].get("resampled", False)])},
            {},
        ),
    )
    monkeypatch.setattr(
        initial,
        "resample_basket_payloads",
        lambda episodes, rows, basket, attempt, **kwargs: {
            row: {**episodes[row], "resampled": True} for row in rows
        },
    )
    _, _, _, checks, _ = initial.admit_pick_batch(runner, episodes, [object(), object()], tmp_path)
    assert all(check["accepted"] for check in checks)
    assert resets == [[0, 1]]
    assert len(payload_resets) == 1 and payload_resets[0].tolist() == [1]
    assert len(calls) == 64
    timing = json.loads((tmp_path / "admission_timing.json").read_text())
    resampled = [phase for phase in timing if phase["payload_only_resets"]]
    assert len(resampled) == 1 and resampled[0]["physics_steps"] == 32
    assert resampled[0]["reset_robots"] == 0 and resampled[0]["payload_only_resets"] == 1


def test_partner_retry_does_not_change_accepted_rows_warmup_commands(monkeypatch, tmp_path):
    runner, episodes, state, calls, resets = batch_fixture(monkeypatch)
    original_step = runner.step_once
    commands = []

    def recorded_step():
        env = runner.env
        commands.append(
            tuple(
                value[0].clone()
                for value in (
                    env.teleop_raw_commands,
                    env.teleop_arm_joint_pos_targets,
                    env.teleop_arm_joint_control_mask,
                    env.gripper_pos_targets,
                )
            )
        )
        return original_step()

    runner.step_once = recorded_step
    monkeypatch.setattr(
        initial,
        "basket_object_checks_batch",
        lambda basket, observed, *args: (
            {
                "layout_valid": torch.tensor(
                    [item["seed"] == 1700 or item.get("resampled", False) for item in episodes]
                )
            },
            {},
        ),
    )
    monkeypatch.setattr(
        initial,
        "resample_basket_payloads",
        lambda episodes, rows, basket, attempt, **kwargs: {
            row: {**episodes[row], "resampled": True} for row in rows
        },
    )
    observed, _, _, checks, _ = initial.admit_pick_batch(
        runner, episodes, [object(), object()], tmp_path
    )
    assert observed is state
    assert len(commands) == 64 and resets == [[0, 1], [1]]
    assert all(check["accepted"] for check in checks)
    for first, repeated in zip(commands[:32], commands[32:], strict=True):
        for first_value, repeated_value in zip(first, repeated, strict=True):
            torch.testing.assert_close(first_value, repeated_value, rtol=0, atol=0)
    history = json.loads((tmp_path / "seed_1700/initial_attempts.json").read_text())
    assert len(history) == 1 and history[0]["attempt"] == 0


def test_final_recheck_retries_an_unstable_partner_within_its_own_budget(monkeypatch, tmp_path):
    runner, episodes, state, calls, resets = batch_fixture(monkeypatch)
    stability = initial.update_robot_stability

    def changed(*args, **kwargs):
        stable, steps, diagnostics = stability(*args, **kwargs)
        if len(calls) > 32:
            stable[0] = False
        return stable, steps, diagnostics

    monkeypatch.setattr(initial, "update_robot_stability", changed)
    monkeypatch.setattr(
        initial,
        "basket_object_checks_batch",
        lambda basket, observed, *args: (
            {"layout_valid": torch.tensor([True, episodes[1].get("resampled", False)])},
            {},
        ),
    )
    monkeypatch.setattr(
        initial,
        "resample_basket_payloads",
        lambda episodes, rows, basket, attempt, **kwargs: {
            row: {**episodes[row], "resampled": True} for row in rows
        },
    )
    clears = []
    clear = initial.TensorStabilityWindow.clear

    def cleared(window, rows):
        clears.append(rows.clone())
        clear(window, rows)

    monkeypatch.setattr(initial.TensorStabilityWindow, "clear", cleared)
    _, _, _, checks, _ = initial.admit_pick_batch(
        runner, episodes, [object(), object()], tmp_path, attempts=2
    )
    assert len(calls) == 364 and resets == [[0, 1], [1], [0]]
    assert [rows.tolist() for rows in clears] == [[0, 1], [1], [0]]
    assert not checks[0]["accepted"]
    assert not checks[0]["checks"]["robot_stable_25_frames"]
    assert checks[1]["accepted"]
    assert checks[0]["attempt"] == 1 and checks[0]["settle_steps"] == 300


def test_revalidation_reference_tracks_last_plan_not_each_waiting_round(monkeypatch, tmp_path):
    runner, episodes, state, calls, resets = batch_fixture(monkeypatch)
    original = runner.step_once

    def drifting():
        result = original()
        state["root_states"][0, 0] = 0.0006 * ((len(calls) - 1) // 32)
        return result

    runner.step_once = drifting
    monkeypatch.setattr(
        initial,
        "basket_object_checks_batch",
        lambda basket, observed, *args: (
            {"layout_valid": torch.tensor([True, episodes[1].get("round", 0) >= 2])},
            {},
        ),
    )
    monkeypatch.setattr(
        initial,
        "resample_basket_payloads",
        lambda episodes, rows, basket, attempt, **kwargs: {
            row: {**episodes[row], "round": int(attempt[index])} for index, row in enumerate(rows)
        },
    )
    _, _, _, checks, plans = initial.admit_pick_batch(
        runner, episodes, [object(), object()], tmp_path, attempts=3
    )
    assert all(check["accepted"] for check in checks)
    assert resets == [[0, 1], [1], [1]]
    history = json.loads((tmp_path / "seed_1700/initial_attempts.json").read_text())
    assert [entry["attempt"] for entry in history] == [0, 0]
    assert [entry["shared_physics_step"] for entry in history] == [32, 96]
    assert plans[0].initial_grasp_plan["state"]["root_states"][0, 0] == pytest.approx(0.0012)
