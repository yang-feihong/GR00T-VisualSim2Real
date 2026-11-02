"""Measured candidates preserve scalar gates, floor reference, ordering and scene ownership."""

import json

import pytest
import torch

from gr00t.rl.pick import grasp_quality, measured_station_batch
from gr00t.rl.pick.measured_commands import load_measured_commands, read_measured_report
from gr00t.rl.pick.reach_calibration import measured_grasp_matches
from gr00t.rl.tests.test_pick_measured_reach import fixture


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU measured station screening")
    return request.param


class Geometry:
    def __init__(self, rows):
        self.rows = rows

    def select(self, rows):
        return Geometry(self.rows[rows])

    def visible(self, origins, ends):
        assert len(origins) == len(self.rows)
        shape = origins.shape[:-1]
        return (self.rows != 2).reshape(len(self.rows), *([1] * (len(shape) - 1))).expand(shape)


class Collision:
    def gripper_clearance(self, geometry, poses, quats, *, maximum_distance_m=None):
        assert len(geometry.rows) == len(poses)
        return poses.new_full(poses.shape[:-1], min(0.01, maximum_distance_m or 0.01))

    def gripper_closing_clearance(
        self, geometry, poses, quats, aperture, *, maximum_distance_m=None
    ):
        # The second scene rejects yaw zero; others cannot borrow its clearance.
        blocked = (geometry.rows[:, None] == 1) & (quats[..., 3].abs() < 0.05)
        return torch.where(blocked, 0.0, min(0.01, maximum_distance_m or 0.01))


def setup(monkeypatch, tmp_path, device):
    _, probe, _ = fixture(tmp_path)
    root = torch.zeros(3, 13, device=device)
    root[:, 0] = root.new_tensor([0, 5, 10])
    floor = root.new_tensor([0, 0.2, 0])
    root[:, 2], root[:, 3] = 0.53 + floor, 1
    obj = root.clone()
    obj[:, 0] += 0.8
    obj[:, 2] = floor + 0.05
    state = dict(root_states=root, object_state=obj, ee_orn=root[:, 3:7])
    supports = root.new_zeros(3, 3, 3)
    mask = torch.ones(3, 3, device=device, dtype=torch.bool)

    def quality(orn, obj_q, support, **kwargs):
        fractions = kwargs.get("depth_fractions")
        shape = (*orn.shape[:2], 17 if fractions is None else 1)
        fit = torch.zeros(shape, device=device, dtype=torch.bool)
        if fractions is None:
            fit[..., 5] = True
        else:
            fit[:] = fractions == 5 / 16
        center = orn.new_zeros(*shape, 3)
        center[..., 2] = -0.0495
        return dict(
            pocket_fit=fit,
            antipodal_margin=orn.new_ones(shape),
            center_local=center,
            aperture_angle_rad=orn.new_ones(shape),
        )

    monkeypatch.setattr(grasp_quality, "convex_grasp_quality", quality)
    monkeypatch.setattr(measured_station_batch, "convex_grasp_quality", quality)
    monkeypatch.setattr(grasp_quality, "load_grasp_support", lambda *args: supports[0])
    commands = load_measured_commands([probe], root)
    return probe, state, floor, supports, mask, Geometry(torch.arange(3, device=device)), commands


def test_large_catalog_is_batch_independent(monkeypatch, tmp_path, device):
    _, state, floor, supports, mask, geometry, commands = setup(monkeypatch, tmp_path, device)
    goals, quats, _, checkpoint = commands
    commands = (
        goals[:1].expand(147, -1),
        quats[:1].expand(147, -1),
        torch.arange(147, device=device),
        checkpoint,
    )
    state["episode_seed"] = torch.tensor([1700, 1701, 1702], device=device)
    destination = state["root_states"][:, [0, 1, 0]].clone()
    destination[:, 2] = 0.3
    baseline = measured_station_batch.select_measured_station_batch(
        state, destination, supports, mask, floor, commands, Collision(), geometry
    )
    assert baseline[4]["measured_command_count"] == 147
    rows = torch.tensor([2, 0, 1], device=device)
    reordered = measured_station_batch.select_measured_station_batch(
        {key: value[rows] for key, value in state.items()},
        destination[rows],
        supports[rows],
        mask[rows],
        floor[rows],
        commands,
        Collision(),
        geometry.select(rows),
    )
    for actual, expected in zip(reordered[:4], baseline[:4], strict=True):
        torch.testing.assert_close(actual, expected[rows], atol=1e-6, rtol=0, equal_nan=True)


def test_measured_selection_matches_original_candidates_and_memory_blocks(
    monkeypatch, tmp_path, device
):
    probe, state, floor, supports, mask, geometry, commands = setup(monkeypatch, tmp_path, device)
    destination = state["root_states"][:, [0, 1, 0]].clone()
    destination[:, 2] = 0.3
    result = measured_station_batch.select_measured_station_batch(
        state,
        destination,
        supports,
        mask,
        floor,
        commands,
        Collision(),
        geometry,
    )
    assert result[3].tolist() == [True, True, False]
    for row in range(3):
        bank = measured_grasp_matches(
            [probe],
            None,
            None,
            device=device,
            episode=dict(
                support_type="floor", support_z=float(floor[row]), asset={"usd_path": "unused"}
            ),
            state={key: value[row : row + 1] for key, value in state.items()},
            collision=Collision(),
            geometry=geometry.select(torch.tensor([row], device=device)),
        )
        assert int(result[-1]["matched_geometry_count"][row]) == len(bank["stations"])
        assert result[-1]["scene_safe_candidate_count"][row] == bank["feasible"].sum()
        if bank["feasible"].any():
            distance = (bank["stations"][:, :2] - state["root_states"][row, :2]).norm(dim=-1)
            selected = distance.masked_fill(~bank["feasible"], torch.inf).argmin()
            torch.testing.assert_close(result[0][row], bank["stations"][selected])
            torch.testing.assert_close(result[1][row], bank["orientations"][selected])
            torch.testing.assert_close(result[2][row], bank["depth_preferences"][selected])
        else:
            torch.testing.assert_close(result[0][row], destination[row])
            torch.testing.assert_close(result[1][row], state["ee_orn"][row])
            assert result[-1]["selected_sample_id"][row] == -1
    split = measured_station_batch.select_measured_station_batch(
        state,
        destination,
        supports,
        mask,
        floor,
        commands,
        Collision(),
        geometry,
        max_evaluations=100,
    )
    for expected, actual in zip(result[:4], split[:4]):
        torch.testing.assert_close(actual, expected)
    for key, value in result[-1].items():
        if torch.is_tensor(value):
            torch.testing.assert_close(split[-1][key], value, equal_nan=True)
        else:
            assert split[-1][key] == value


def test_measured_batch_permutation_and_empty_height_matches_need_no_host_reads(
    monkeypatch, tmp_path, device
):
    _, state, floor, supports, mask, geometry, commands = setup(monkeypatch, tmp_path, device)
    destination = state["root_states"][:, [0, 1, 0]].clone()
    destination[:, 2] = 0
    order = torch.tensor([2, 0, 1], device=device)
    reference = measured_station_batch.select_measured_station_batch(
        state,
        destination,
        supports,
        mask,
        floor,
        commands,
        Collision(),
        geometry,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Measured station decisions must remain on device")

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        result = measured_station_batch.select_measured_station_batch(
            {key: value[order] for key, value in state.items()},
            destination[order],
            supports[order],
            mask[order],
            floor[order],
            commands,
            Collision(),
            geometry.select(order),
            max_evaluations=100,
        )
        empty = measured_station_batch.select_measured_station_batch(
            state,
            destination,
            supports,
            mask,
            floor + 1,
            commands,
            Collision(),
            geometry,
        )
    for expected, actual in zip(reference[:4], result[:4]):
        torch.testing.assert_close(actual, expected[order])
    assert not empty[3].any()
    torch.testing.assert_close(empty[0], destination)
    assert (empty[-1]["matched_geometry_count"] == 0).all()


def test_catalog_cache_reuses_immutable_data_and_refreshes_changed_file(tmp_path):
    _, probe, report = fixture(tmp_path)
    path = probe / "summary.json"
    reference = torch.zeros(1)
    first = load_measured_commands([probe], reference)
    again = load_measured_commands([path], reference)
    assert first[0].data_ptr() == again[0].data_ptr()
    _, digest = read_measured_report(probe)
    report["goals_world"][0][0] += 0.1
    report["revision"] = 1
    path.write_text(json.dumps(report))
    updated = load_measured_commands([probe], reference)
    _, new_digest = read_measured_report(path)
    assert first[0].data_ptr() != updated[0].data_ptr() and digest != new_digest
    torch.testing.assert_close(updated[0] - first[0], torch.tensor([[0.1, 0, 0]]))


def test_measured_station_uses_native_convex_pocket_geometry_consistently(monkeypatch, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("Deployed gripper convex geometry")
    native_quality = grasp_quality.convex_grasp_quality
    probe, state, floor, _, _, geometry, _ = setup(monkeypatch, tmp_path, "cuda")
    monkeypatch.setattr(grasp_quality, "convex_grasp_quality", native_quality)
    monkeypatch.setattr(measured_station_batch, "convex_grasp_quality", native_quality)
    cube = torch.cartesian_prod(*[torch.tensor([-0.03, 0.03], device="cuda")] * 3)
    state["object_state"][:, 2] = floor + 0.03
    base_quality = native_quality(
        state["ee_orn"][:1, None],
        state["object_state"][:1, 3:7],
        cube,
        return_depth_candidates=True,
    )
    feasible = base_quality["pocket_fit"][0, 0] & (base_quality["antipodal_margin"][0, 0] >= 0)
    assert feasible.any()
    depth = torch.where(feasible)[0][0]
    goal_z = 0.03 - float(base_quality["center_local"][0, 0, depth, 2]) + 0.0005
    path = probe / "summary.json"
    report = json.loads(path.read_text())
    report["goals_world"][0][2] = goal_z
    path.write_text(json.dumps(report))
    commands = load_measured_commands([probe], state["root_states"])
    monkeypatch.setattr(grasp_quality, "load_grasp_support", lambda *args: cube)
    destination = state["root_states"][:, [0, 1, 0]].clone()
    destination[:, 2] = 0
    result = measured_station_batch.select_measured_station_batch(
        state,
        destination,
        cube[None].expand(3, -1, -1),
        torch.ones(3, len(cube), device="cuda", dtype=torch.bool),
        floor,
        commands,
        Collision(),
        geometry,
    )
    assert result[3].tolist() == [True, True, False]
    for row in (0, 1):
        bank = measured_grasp_matches(
            [probe],
            None,
            None,
            device="cuda",
            episode=dict(
                support_type="floor", support_z=float(floor[row]), asset={"usd_path": "unused"}
            ),
            state={key: value[row : row + 1] for key, value in state.items()},
            collision=Collision(),
            geometry=geometry.select(torch.tensor([row], device="cuda")),
        )
        distance = (bank["stations"][:, :2] - state["root_states"][row, :2]).norm(dim=-1)
        chosen = distance.masked_fill(~bank["feasible"], torch.inf).argmin()
        torch.testing.assert_close(result[0][row], bank["stations"][chosen], atol=1e-6, rtol=0)
        torch.testing.assert_close(result[1][row], bank["orientations"][chosen], atol=1e-6, rtol=0)
        torch.testing.assert_close(result[2][row], bank["depth_preferences"][chosen])
