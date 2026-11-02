"""Cached direct paths still require the original realized-height endpoint gate."""

import math

import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cached_direct_gate_keeps_only_full_path_and_endpoint_success(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU admission")
    from gr00t.rl.pick import basket_initialization as module
    from gr00t.rl.pick import basket_inverse

    roots = torch.zeros(3, 13, device=device)
    roots[:, 3] = 1
    state = {
        "root_states": roots,
        "object_state": roots.clone(),
        "ee_pos": roots[:, :3],
        "dof_pos": roots.new_zeros(3, 6),
        "optional_camera": None,
    }
    goals = torch.arange(27, device=device).float().reshape(3, 3, 3)
    orientations = roots.new_tensor([1, 0, 0, 0]).expand(3, 3, 4)
    quality = {
        "pocket_fit": torch.ones(3, 3, device=device, dtype=torch.bool),
        "antipodal_margin": roots.new_ones(3, 3),
        "support_pinch_compatible": torch.ones(3, 3, device=device, dtype=torch.bool),
        "environment_clearance_m": roots.new_full((3, 3), 0.007),
    }
    bank = orientations, goals, quality, 3, 0
    selected_pre = roots.new_tensor([[0, 1, 2], [4, 5, 6], [8, 9, 10]])
    paths = (goals, orientations, torch.full((3,), 3, device=device, dtype=torch.long))
    monkeypatch.setattr(
        module, "basket_required_height", lambda *a, **kw: roots.new_full((3,), 0.9)
    )

    def plan(*args, **kwargs):
        assert kwargs["candidate_bank"] is bank
        assert kwargs["minimum_realized_grasp_margin_rad"] == math.radians(2)
        assert kwargs["realized_body"].all() and not kwargs["prefer_realized_body"].any()
        return (
            paths,
            torch.tensor([True, False, True], device=device),
            {
                "selected": torch.tensor([0, -1, 1], device=device),
                "candidate_source_indices": torch.tensor([[2, 0], [0, 1], [1, 2]], device=device),
                "selected_pregrasp_position_world_m": selected_pre,
            },
        )

    checked = []

    def endpoints(local, p, q, limits, collision, geometry, **kwargs):
        assert kwargs["body_height_offsets"] == (-0.01, 0.0, 0.01)
        assert kwargs["final_joint_margin_rad"] == math.radians(2)
        assert kwargs["minimum_arm_clearance_m"] == 0.006
        torch.testing.assert_close(p[:, 0, 0], selected_pre[[0, 2]])
        torch.testing.assert_close(p[:, 0, 1], goals[[0, 2], 2])
        checked.append(len(p))
        return torch.tensor([[True], [False]], device=device), None, {}

    monkeypatch.setattr(module, "plan_basket_grasp_batch", plan)
    monkeypatch.setattr(basket_inverse, "safe_inverse_paths_batch", endpoints)
    from types import SimpleNamespace

    accepted, report, actual_path = module.basket_grasp_access_batch(
        state,
        roots.new_full((3, 3), 0.04),
        None,
        None,
        None,
        None,
        SimpleNamespace(select=lambda rows: rows),
        None,
        None,
        roots.new_zeros(3, 8, 3),
        torch.ones(3, 8, device=device, dtype=torch.bool),
        roots.new_zeros(3, 3),
        roots.new_tensor([[-3, 3]] * 6),
        0.02,
        lift_margin=roots.new_full((3,), 0.1),
        candidate_bank=bank,
        direct_only=True,
    )
    torch.testing.assert_close(accepted, torch.tensor([True, False, False], device=device))
    assert checked == [2] and actual_path is paths
    assert report["direct_access.reachable_grasp_count_is_lower_bound"].all()
    assert not report["initial_reposition_required"].any()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_actual_initialization_cache_branch_ignores_optional_sdk_state(device):
    import ast
    import copy
    import inspect
    from types import SimpleNamespace

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU initialization")
    from gr00t.rl.pick import task_initialization as module

    tree = ast.parse(inspect.getsource(module.admit_pick_batch))
    branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "cached_intrinsic_proposals"
            for child in ast.walk(node)
        )
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "probe"
    ]
    assert len(branches) == 1
    template = ast.parse(
        "def probe_cache(state, selected, basket_inputs, episode_seeds,\n"
        "                collision, scene_geometry):\n"
        "    probe = 'contacts'\n"
        "    options = {}\n"
        "    return options\n"
    )
    template.body[0].body.insert(-1, copy.deepcopy(branches[0]))
    state = {
        "root_states": torch.zeros(3, 13, device=device),
        "object_state": torch.ones(3, 13, device=device),
        "optional_camera": None,
    }
    rows = torch.tensor([2, 0], device=device)
    seen = []

    class Cache:
        def select(self, selected):
            torch.testing.assert_close(selected, rows)
            return self

        def candidate_bank(self, local, collision, geometry):
            assert set(local) == {"root_states", "object_state"}
            for key in local:
                torch.testing.assert_close(local[key], state[key][rows])
            seen.append(True)
            return "bank"

    namespace = {
        "cached_intrinsic_proposals": lambda *a: Cache(),
        "BasketArmCollisionSamples": lambda collision: collision,
    }
    exec(compile(ast.fix_missing_locations(template), "actual_cache_entry", "exec"), namespace)
    result = namespace["probe_cache"](
        state, rows, None, None, None, SimpleNamespace(select=lambda rows: None)
    )
    assert result == {"candidate_bank": "bank", "direct_only": True} and seen == [True]
