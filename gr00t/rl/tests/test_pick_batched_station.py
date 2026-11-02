"""Initial station ranking and endpoint admission preserve independent scene decisions."""

from dataclasses import dataclass

import pytest
import torch

from gr00t.rl.auto_open import pose_planning
from gr00t.rl.pick import station, station_batch


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU initial station screening")
    return request.param


@pytest.mark.parametrize(
    "volume,standing,all_postures,heights",
    [
        (False, False, False, False),
        (True, False, False, False),
        (False, True, False, False),
        (True, True, False, True),
        (False, True, True, True),
    ],
)
def test_station_ranking_matches_scene_reference_and_permutation(
    device, volume, standing, all_postures, heights
):
    generator = torch.Generator(device=device).manual_seed(417)
    cost = torch.rand(4, 3, 3, 4, 5, device=device, generator=generator)
    cost.masked_fill_(cost < 0.4, torch.inf)
    cost[1, :, 2] = torch.inf  # No standing solution: this row needs the fallback policy.
    cost[2] = torch.inf
    standing_mask = torch.zeros(4, 3, device=device, dtype=torch.bool)
    standing_mask[:, 2] = True
    height = cost.new_tensor([[0.3, 0.4, 0.53]]).expand(4, -1)
    options = dict(
        body_volume=volume,
        require_all_postures=all_postures,
        standing_postures=standing_mask if standing else None,
        posture_heights=height if heights else None,
    )
    eligible, worst, diagnostics = station_batch.rank_station_candidates_batch(cost, 1, **options)
    for row in range(4):
        expected, expected_cost, report = station.rank_station_candidates(
            cost[row],
            1,
            body_volume=volume,
            require_all_postures=all_postures,
            standing_postures=standing_mask[row] if standing else None,
            posture_heights=height[row] if heights else None,
        )
        torch.testing.assert_close(eligible[row], expected)
        torch.testing.assert_close(worst[row], expected_cost)
        for key in ("arrival_coverage_counts", "body_arrival_support_counts"):
            torch.testing.assert_close(diagnostics[key][row], report[key])
        assert (
            station_batch.STATION_POLICIES[int(diagnostics["selection_policy_code"][row])]
            == report["selection_policy"]
        )
    order = torch.tensor([2, 0, 3, 1], device=device)
    options["standing_postures"] = standing_mask[order] if standing else None
    options["posture_heights"] = height[order] if heights else None
    shuffled, shuffled_cost, _ = station_batch.rank_station_candidates_batch(
        cost[order], 1, **options
    )
    torch.testing.assert_close(shuffled, eligible[order])
    torch.testing.assert_close(shuffled_cost, worst[order])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU station reduction timing")
def test_real_size_station_reductions_complete_and_preserve_rows():
    generator = torch.Generator(device="cuda").manual_seed(417)
    cost = torch.rand(2, 9, 8, 15, 1794, device="cuda", generator=generator)
    cost.masked_fill_(cost < 0.9, torch.inf)
    cost[0, :, -2:] = torch.inf
    standing = torch.zeros(2, 8, device="cuda", dtype=torch.bool)
    standing[:, -2:] = True
    heights = cost.new_tensor([0.26, 0.26, 0.36, 0.36, 0.46, 0.46, 0.536, 0.536])
    options = {
        "standing_postures": standing,
        "posture_heights": heights[None].expand(2, -1),
    }
    station_batch.rank_station_candidates_batch(cost, 1, **options)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    eligible, worst, diagnostics = station_batch.rank_station_candidates_batch(cost, 1, **options)
    end.record()
    end.synchronize()
    print({"station_cost_shape": list(cost.shape), "ranking_cuda_ms": start.elapsed_time(end)})
    assert diagnostics["selection_policy_code"].tolist() == [5, 3]
    for row in range(2):
        expected, expected_cost, _ = station.rank_station_candidates(
            cost[row], 1, standing_postures=standing[row], posture_heights=heights
        )
        torch.testing.assert_close(eligible[row], expected)
        torch.testing.assert_close(worst[row], expected_cost)


@dataclass
class Result:
    arm_joint_positions: torch.Tensor
    position_error: torch.Tensor
    exact: torch.Tensor


class Geometry:
    def __init__(self, scenes):
        self.scenes, self.device = scenes, scenes.device

    def select(self, rows):
        return Geometry(self.scenes[rows])

    def visible(self, origins, ends):
        assert origins.shape[0] == len(self.scenes)
        return torch.ones(origins.shape[:-1], dtype=torch.bool, device=origins.device)


class Collision:
    def __init__(self, device):
        self.arm_joint_ids = torch.arange(6, device=device)

    def clearance(self, geometry, roots, joints, **kwargs):
        assert kwargs["include_gripper"] is False
        # Both the packed and scalar calls must retain their scene's BVH.
        scene = geometry.scenes.expand(len(roots))
        assert ((roots[:, 0] - scene).abs() < 1).all()
        return roots.new_full(roots.shape[:-1], 0.03)

    def gripper_clearance(self, geometry, points, orientations, **kwargs):
        if "maximum_distance_m" in kwargs:
            assert kwargs["maximum_distance_m"] == 0.0021
        assert points.shape[0] == len(geometry.scenes)
        assert ((points[..., 0] - geometry.scenes[:, None]).abs() < 2).all()
        return points.new_full(points.shape[:-1], 0.03)


def test_station_gripper_queries_are_deduplicated_without_scene_or_mask_leakage(device):
    class Surface:
        def __init__(self):
            self.requests = 0

        def gripper_clearance(self, geometry, points, orientations, **kwargs):
            self.requests += points.shape[0] * points.shape[1]
            return points[..., 2] + geometry.scenes[:, None] + orientations[..., 3]

    geometry = Geometry(torch.tensor([0.0, 10.0], device=device))
    owners = torch.arange(2, device=device).repeat_interleave(12)
    candidates = torch.arange(4, device=device).repeat_interleave(3).repeat(2)
    bodies = torch.arange(3, device=device).repeat(8)
    points = torch.zeros(24, 3, 3, device=device)
    points[..., 0] = geometry.scenes[owners, None] + 0.1 * candidates[:, None]
    points[..., 2] = points.new_tensor([0.001, 0.002, 0.003])
    points[:, 2, 2] += bodies * 0.001
    q = points.new_zeros(24, 4)
    q[:, 0] = 1
    active = torch.ones(24, 3, device=device, dtype=torch.bool)
    active[::3, 0] = False
    surface = Surface()
    expected = points[..., 2] + geometry.scenes[owners, None]
    expected = torch.where(active, expected, expected.new_tensor(0.0021))
    actual = station_batch._station_gripper_clearance(geometry, surface, owners, points, q, active)
    torch.testing.assert_close(actual, expected)
    assert surface.requests == 40  # Two common endpoints plus three withdrawal endpoints per grasp.
    order = torch.arange(23, -1, -1, device=device)
    shuffled = station_batch._station_gripper_clearance(
        geometry, surface, owners[order], points[order], q[order], active[order]
    )
    torch.testing.assert_close(shuffled, expected[order])


def tools(monkeypatch, device):
    def bank(target, headings, size, origins, **kwargs):
        q = target.new_zeros(len(target), 3, 4)
        q[..., 0] = 1
        return target[:, None].expand(-1, 3, -1), q, q[..., :3]

    def pockets(q, target, obj_q, support, collision, geometry, **kwargs):
        shape = q.shape[:-1]
        points = target[:, None].expand(-1, 3, -1).clone()
        points[:, :, 1] += points.new_tensor([0, -0.01, 0.01])
        return points, {
            "pocket_fit": (target[:, :1] < 15).expand(shape),
            "antipodal_margin": target.new_full(shape, 0.2),
            "environment_clearance_m": target.new_full(shape, 0.03),
        }

    requests = []

    def inverse(p, q, **kwargs):
        requests.append(len(p))
        roots, seed = kwargs["base_position_world"], kwargs["initial_arm_joint_positions"]
        # Scene 10 needs different body heights for grasp and carrying.
        fallback = roots[:, 0] > 5
        grasp = p[:, 2] < 0.15
        exact = ~fallback | torch.where(grasp, roots[:, 2] < 0.4, roots[:, 2] > 0.5)
        joints = seed.clone()
        joints[:, 5] = roots[:, 2] * 0.1
        error = torch.zeros_like(p)
        error[:, 0] = (roots[:, 0] - p[:, 0]).abs() * 0.001
        return Result(joints, error, exact)

    def scalar_inverse(p, q, roots, seeds, limits):
        return inverse(
            p.flatten(0, 1),
            q.flatten(0, 1),
            base_position_world=roots[:, None, :3].expand(-1, 3, -1).flatten(0, 1),
            initial_arm_joint_positions=seeds[:, None].expand(-1, 3, -1).flatten(0, 1),
        )

    for module in (station, station_batch):
        monkeypatch.setattr(module, "grasp_pose_bank", bank)
        monkeypatch.setattr(module, "pocket_pose_targets", pockets)
        monkeypatch.setattr(module, "object_frame_grasp_bank", lambda q, size: q[:, None][:, :0])
    monkeypatch.setattr(station, "inverse_bank", scalar_inverse)
    monkeypatch.setattr(pose_planning, "solve_pose_targets", inverse)
    root = torch.zeros(3, 13, device=device)
    root[:, 0] = root.new_tensor([0, 10, 20])
    root[:, 2], root[:, 3] = 0.536, 1
    obj = torch.cat((root[:, :3] + root.new_tensor([0.8, 0, -0.50]), root[:, 3:7]), -1)
    state = dict(
        root_states=root, object_state=obj, dof_pos=root.new_zeros(3, 6), ee_orn=root[:, 3:7]
    )
    destination = torch.cat((root[:, :2] + root.new_tensor([0.1, 0]), root.new_zeros(3, 1)), -1)
    return state, destination, Geometry(root[:, 0]), Collision(device), requests


@pytest.mark.parametrize("all_postures", [False, True])
def test_batched_station_matches_original_selection_and_own_row_fallback(
    monkeypatch, device, all_postures
):
    state, destination, geometry, collision, requests = tools(monkeypatch, device)
    limits = destination.new_tensor([[-3, 3]] * 6)
    support = destination.new_zeros(3, 8, 3)
    options = dict(
        body_heights=(0.30, 0.536),
        body_pitches_deg=(0,),
        yaw_offsets_deg=None,
        forward_offsets_m=(-0.1, 0, 0.1),
        lateral_offsets_m=(0,),
        require_all_postures=all_postures,
    )
    result = station_batch.select_initial_pick_station_batch(
        state,
        destination,
        destination.new_full((3, 3), 0.06),
        support,
        limits,
        collision,
        geometry,
        max_requests=137,
        **options,
    )
    assert max(requests) <= 137
    pose, q, joints, valid, report = result
    assert not valid[2]
    torch.testing.assert_close(pose[2], destination[2])
    torch.testing.assert_close(q[2], state["ee_orn"][2])
    torch.testing.assert_close(joints[2], state["dof_pos"][2])
    for row in range(3):
        local = {key: value[row : row + 1] for key, value in state.items()}
        expected_p, expected_q, expected_j, diagnostics = station.select_initial_pick_station(
            local,
            destination[row : row + 1],
            [0.06] * 3,
            support[row],
            limits,
            collision,
            geometry.select(torch.tensor([row], device=device)),
            **options,
        )
        assert bool(valid[row]) == (expected_p is not None)
        if expected_p is not None:
            torch.testing.assert_close(pose[row : row + 1], expected_p)
            torch.testing.assert_close(q[row : row + 1], expected_q)
            torch.testing.assert_close(joints[row : row + 1], expected_j)
            for key in (
                "selected",
                "selected_candidate",
                "selected_arrival_coverage",
            ):
                torch.testing.assert_close(report[key][row], diagnostics[key])
    order = torch.tensor([2, 0, 1], device=device)
    permuted = station_batch.select_initial_pick_station_batch(
        {key: value[order] for key, value in state.items()},
        destination[order],
        destination.new_full((3, 3), 0.06),
        support[order],
        limits,
        collision,
        geometry.select(order),
        max_requests=101,
        **options,
    )
    for actual, original in zip(permuted[:4], result[:4]):
        torch.testing.assert_close(actual, original[order])


def test_station_memory_groups_preserve_all_diagnostics_including_empty_groups(monkeypatch, device):
    state, destination, geometry, collision, _ = tools(monkeypatch, device)
    order = torch.tensor([0, 1, 2, 2, 0, 1], device=device)
    state = {key: value[order] for key, value in state.items()}
    destination = destination[order]
    geometry = geometry.select(order)
    limits = destination.new_tensor([[-3, 3]] * 6)
    support = destination.new_zeros(6, 8, 3)
    options = dict(
        body_heights=(0.30, 0.536),
        body_pitches_deg=(0,),
        yaw_offsets_deg=None,
        forward_offsets_m=(0,),
        lateral_offsets_m=(0,),
        max_requests=211,
    )
    full = station_batch.select_initial_pick_station_batch(
        state,
        destination,
        destination.new_full((6, 3), 0.06),
        support,
        limits,
        collision,
        geometry,
        max_scenes=6,
        **options,
    )
    split = station_batch.select_initial_pick_station_batch(
        state,
        destination,
        destination.new_full((6, 3), 0.06),
        support,
        limits,
        collision,
        geometry,
        max_scenes=2,
        **options,
    )
    for actual, expected in zip(split[:4], full[:4]):
        torch.testing.assert_close(actual, expected)
    assert split[-1].keys() == full[-1].keys()
    for key, value in full[-1].items():
        if torch.is_tensor(value):
            torch.testing.assert_close(split[-1][key], value, equal_nan=True)
        else:
            assert split[-1][key] == value


def test_station_gpu_planning_does_not_read_host_tensors(monkeypatch, device):
    state, destination, geometry, collision, _ = tools(monkeypatch, device)

    def forbidden(*args, **kwargs):
        raise AssertionError("Station planning must retain device-side decisions")

    for name in ("item", "cpu", "tolist"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    result = station_batch.select_initial_pick_station_batch(
        state,
        destination,
        [0.06] * 3,
        destination.new_zeros(3, 8, 3),
        destination.new_tensor([[-3, 3]] * 6),
        collision,
        geometry,
        body_heights=(0.30, 0.536),
        body_pitches_deg=(0,),
        yaw_offsets_deg=None,
        forward_offsets_m=(0,),
        lateral_offsets_m=(0,),
        max_scenes=2,
        max_requests=127,
    )
    assert result[3].device == destination.device


def test_station_selection_with_native_inverse_matches_original_scene_solver(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("Native GPU station inverse")
    from legged_gym.utils.ee_steady_inverse_gpu import _matrix_to_quaternion_wxyz

    from gr00t.rl.envs.door.b2z1_reset_geometry import b2z1_gripper_pose_and_jacobian

    inverse, bank_inverse = pose_planning.solve_pose_targets, station.inverse_bank
    state, destination, geometry, collision, _ = tools(monkeypatch, "cuda")
    monkeypatch.setattr(pose_planning, "solve_pose_targets", inverse)
    monkeypatch.setattr(station, "inverse_bank", bank_inverse)
    state = {key: value[:2].clone() for key, value in state.items()}
    state["dof_pos"][:] = destination.new_tensor([-1.57, 2.6, -1.5, 0.4, 0, 0])
    p, rotation, _ = b2z1_gripper_pose_and_jacobian(
        state["root_states"][:, :3], state["root_states"][:, [4, 5, 6, 3]], state["dof_pos"]
    )
    state["ee_orn"] = _matrix_to_quaternion_wxyz(rotation)
    state["object_state"][:, :3] = p
    state["object_state"][:, 3:7] = state["ee_orn"]

    def bank(target, heading, size, origin, **kwargs):
        q = kwargs["reference_orientation"][:, None]
        return target[:, None], q, target[:, None]

    def pockets(q, target, *args, **kwargs):
        return target[:, None], {
            "pocket_fit": torch.ones(q.shape[:-1], device=q.device, dtype=torch.bool),
            "antipodal_margin": q.new_full(q.shape[:-1], 0.2),
            "environment_clearance_m": q.new_full(q.shape[:-1], 0.03),
        }

    for module in (station, station_batch):
        monkeypatch.setattr(module, "grasp_pose_bank", bank)
        monkeypatch.setattr(module, "pocket_pose_targets", pockets)
    limits = destination.new_tensor([[-3, 3]] * 6)
    options = dict(
        body_heights=(0.30, 0.536),
        body_pitches_deg=(0,),
        yaw_offsets_deg=None,
        forward_offsets_m=(0,),
        lateral_offsets_m=(0,),
    )
    # Keep the station yaw at the actual robot yaw so the task orientation is unchanged.
    destination = state["root_states"][:, [0, 1, 0]].clone()
    destination[:, 2] = 0
    geometry = Geometry(state["root_states"][:, 0])
    result = station_batch.select_initial_pick_station_batch(
        state,
        destination,
        [0.06] * 3,
        destination.new_zeros(2, 8, 3),
        limits,
        collision,
        geometry,
        max_requests=113,
        **options,
    )
    assert result[3].all()
    for row in range(2):
        local = {key: value[row : row + 1] for key, value in state.items()}
        expected = station.select_initial_pick_station(
            local,
            destination[row : row + 1],
            [0.06] * 3,
            destination.new_zeros(8, 3),
            limits,
            collision,
            geometry.select(torch.tensor([row], device="cuda")),
            **options,
        )
        assert expected[0] is not None
        for actual, reference in zip(result[:3], expected[:3]):
            torch.testing.assert_close(actual[row : row + 1], reference, atol=1e-5, rtol=0)
