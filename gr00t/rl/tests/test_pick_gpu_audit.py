"""Inventory host boundaries and prevent scalar planner dispatch in production paths."""

import ast
import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from gr00t.rl.auto_open.tensor_metadata import to_host
from gr00t.rl.auto_open.tensor_random import episode_uniform
from gr00t.rl.pick.tensor_rows import pack_scalars, pad_device_rows

ROOT = Path(__file__).parents[1]


class Inventory(ast.NodeVisitor):
    def __init__(self):
        self.scope, self.transfers, self.loops, self.synchronizations = [], [], [], []

    def visit_FunctionDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "synchronize":
            self.synchronizations.append({"scope": ".".join(self.scope), "line": node.lineno})
        if isinstance(node.func, ast.Attribute) and node.func.attr in {
            "cpu",
            "item",
            "tolist",
            "numpy",
        }:
            self.transfers.append(
                {
                    "scope": ".".join(self.scope),
                    "line": node.lineno,
                    "operation": node.func.attr,
                }
            )
        self.generic_visit(node)

    def visit_For(self, node):
        self.loops.append(
            {
                "scope": ".".join(self.scope),
                "line": node.lineno,
                "target": ast.unparse(node.target),
                "iterator": ast.unparse(node.iter),
            }
        )
        self.generic_visit(node)

    def visit_comprehension(self, node):
        self.loops.append(
            {
                "scope": ".".join(self.scope),
                "line": node.target.lineno,
                "target": ast.unparse(node.target),
                "iterator": ast.unparse(node.iter),
                "kind": "comprehension",
            }
        )
        self.generic_visit(node)


def test_inventory_every_pick_host_boundary_and_loop(tmp_path):
    report = {}
    shared = [
        ROOT / "auto_open" / f"{name}.py"
        for name in (
            "acceptance",
            "execution",
            "control",
            "base_trajectory",
            "motion_targets",
            "pose_planning",
            "stages",
            "planning_geometry",
            "motion_math",
            "stage_interfaces",
            "planning_ik",
            "tensor_random",
            "tensor_metadata",
        )
    ]
    inverse = (
        ROOT / "third_party/visual_wholebody_lowlevel/legged_gym/utils/ee_steady_inverse_gpu.py"
    )
    for path in sorted([*(ROOT / "pick").glob("*.py"), *shared, inverse]):
        inventory = Inventory()
        inventory.visit(ast.parse(path.read_text()))
        report[str(path.relative_to(ROOT))] = {
            "host_boundaries": inventory.transfers,
            "loops": inventory.loops,
            "explicit_host_synchronizations": inventory.synchronizations,
        }
    destination = tmp_path / "gpu_host_boundary_inventory.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(destination)
    reviewed = json.loads(
        (Path(__file__).with_name("data") / "pick_host_boundary_allowlist.json").read_text()
    )
    expected = Counter(
        {
            (entry["file"], entry["scope"], entry["operation"]): entry["count"]
            for entry in reviewed["entries"]
        }
    )
    actual = Counter(
        (file, site["scope"], site["operation"])
        for file, item in report.items()
        for site in item["host_boundaries"]
    )
    assert actual == expected, (
        "Host boundary changed: review its necessity before updating the allowlist"
    )
    # All spatial decisions in these modules must stay on device. The single
    # explicit serializer is host_metadata, after decisions have been made.
    for name in (
        "randomization",
        "distractors",
        "basket_layout",
        "basket_preparation",
        "external_preparation",
        "task_initialization",
        "tensor_fsm",
        "tensor_rows",
    ):
        assert report[f"pick/{name}.py"]["host_boundaries"] == [], name
    for site in report["pick/tensor_planning.py"]["host_boundaries"]:
        pytest.fail(f"Planning host boundary must use the audited bulk serializer: {site}")
    for path in [*shared[1:-1], inverse]:
        assert not report[str(path.relative_to(ROOT))]["host_boundaries"], path
        expected = ["solve_candidate_ik"] if path.name == "planning_ik.py" else []
        assert [
            site["scope"]
            for site in report[str(path.relative_to(ROOT))]["explicit_host_synchronizations"]
        ] == expected


def test_preparation_and_resampling_cannot_call_scalar_planners():
    forbidden = {"prepare_episode", "prepare_basket_episode", "resample_basket_payload"}
    for name in (
        "prepare_batch",
        "task_initialization",
        "external_preparation",
        "basket_preparation",
    ):
        tree = ast.parse((ROOT / f"pick/{name}.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden, (name, node.lineno, node.func.id)
    for name in ("randomization", "distractors", "basket_layout", "basket_preparation"):
        source = (ROOT / f"pick/{name}.py").read_text()
        assert "torch.Generator" not in source
        assert "torch.rand(" not in source
        assert "torch.randint(" not in source
        assert "torch.randperm(" not in source
    spatial = {
        "rays",
        "visible",
        "distances",
        "clearance",
        "inverse",
        "inverse_gpu",
        "gripper_clearance",
        "gripper_closing_clearance",
        "solve_pose_grid",
        "solve_pose_targets",
        "convex_grasp_quality",
        "sample_candidates_batch",
    }
    for name in (
        "randomization",
        "distractors",
        "basket_layout",
        "basket_preparation",
        "external_preparation",
        "task_initialization",
        "tensor_planning",
        "tensor_fsm",
    ):
        tree = ast.parse((ROOT / f"pick/{name}.py").read_text())
        for loop in ast.walk(tree):
            if not isinstance(loop, ast.For):
                continue
            targets = {node.id for node in ast.walk(loop.target) if isinstance(node, ast.Name)}
            if not targets & {"row", "env_id", "controller", "seed", "episode"}:
                continue
            for node in ast.walk(loop):
                if isinstance(node, ast.Call):
                    called = (
                        node.func.attr
                        if isinstance(node.func, ast.Attribute)
                        else (node.func.id if isinstance(node.func, ast.Name) else None)
                    )
                    assert called not in spatial, (name, loop.lineno, called)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shared_counter_rng_is_batch_shape_independent_and_in_range(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA random streams")
    seeds = torch.tensor([1700, 1701, 2**40 + 19, 2**62 + 23], device=device)
    stream = torch.tensor([0, 7, 2**33 + 1, 17], device=device)
    reference = episode_uniform(seeds, stream, 128, 8)
    assert (reference >= 0).all() and (reference < 1).all()
    assert reference.std() > 0.25
    order = torch.tensor([3, 1, 0, 2], device=device)
    torch.testing.assert_close(
        episode_uniform(seeds[order], stream[order], 128, 8), reference[order], rtol=0, atol=0
    )
    for row in range(len(seeds)):
        torch.testing.assert_close(
            episode_uniform(seeds[row : row + 1], stream[row : row + 1], 32, 4),
            reference[row : row + 1, :32, :4],
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        reference.cpu(), episode_uniform(seeds.cpu(), stream.cpu(), 128, 8), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA metadata packing")
def test_device_packing_needs_no_tensor_scalar_reads(monkeypatch):
    values = [
        torch.rand(7, 3, device="cuda"),
        torch.rand(0, 3, device="cuda"),
        torch.rand(13, 3, device="cuda"),
    ]
    scalar = values[0][0, 0]

    def forbidden(*args, **kwargs):
        raise AssertionError("Packing must not read an individual device value")

    with monkeypatch.context() as scope:
        for name in ("cpu", "item", "tolist", "__bool__"):
            scope.setattr(torch.Tensor, name, forbidden)
        packed, mask = pad_device_rows(values, fill=torch.nan)
        scalars = pack_scalars([0.25, scalar, 0.75], values[0])
    assert packed.shape == (3, 13, 3)
    assert mask.sum(-1).tolist() == [7, 0, 13]
    torch.testing.assert_close(packed[0, :7], values[0])
    torch.testing.assert_close(scalars[1], scalar)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA metadata packing")
def test_metadata_tree_transfers_once_per_dtype_not_per_row(monkeypatch):
    tree = {
        "poses": [torch.rand(1, 3, device="cuda") for _ in range(100)],
        "indices": torch.arange(100, device="cuda"),
        "flags": torch.ones(100, device="cuda", dtype=torch.bool),
        "label": "unchanged",
        "scalar": torch.tensor(0.5, device="cuda"),
    }
    original = torch.Tensor.cpu
    calls = []

    def tracked(value, *args, **kwargs):
        calls.append((value.device, value.dtype))
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", tracked)
    result = to_host(tree)
    assert len(calls) == 3
    assert result["label"] == "unchanged" and result["scalar"] == 0.5
    assert result["indices"] == list(range(100))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA acceptance export")
def test_batched_acceptance_export_preserves_all_fields_without_scalar_device_reads(monkeypatch):
    from enum import IntEnum

    from gr00t.rl.auto_open.acceptance import StageAcceptanceMixin

    stages = IntEnum("Stages", {"FIRST": 1, "UNVISITED": 2})
    fsm = StageAcceptanceMixin()
    fsm.device, fsm.num_envs = torch.device("cuda"), 8
    fsm.stage_steps = torch.ones(8, dtype=torch.int32, device="cuda")
    fsm.stage_visit = torch.ones(8, 3, dtype=torch.int32, device="cuda")
    fsm._acceptance = {}
    rows = torch.arange(6, device="cuda")
    for step in range(3):
        fsm.stage_steps += 1
        value = torch.arange(6, device="cuda") * 0.01 + step * 0.001
        fsm.record_acceptance(
            1, rows, {"position": (value < 0.025, value, "<", 0.025)}, required_streak=2
        )
    fsm.stage_visit[0, 1] += 1
    expected = [
        {stage.name: fsm.acceptance_diagnostics(row, stage) for stage in stages} for row in range(8)
    ]
    original, calls = torch.Tensor.cpu, []

    def tracked(value, *args, **kwargs):
        calls.append(value.numel())
        return original(value, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Batched final diagnostics must not read device scalars")

    monkeypatch.setattr(torch.Tensor, "cpu", tracked)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    actual = fsm.acceptance_diagnostics_batch(stages)
    assert actual == expected
    assert len(calls) == 3
