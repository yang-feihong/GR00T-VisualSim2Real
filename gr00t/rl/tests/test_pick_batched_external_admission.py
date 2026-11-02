"""External object support admission matches each original scene under padded batching."""

import pytest
import torch

from gr00t.rl.pick.batched_geometry import BatchedSceneGeometry
from gr00t.rl.pick.distractors import object_category
from gr00t.rl.pick.external_initialization import ExternalAdmissionInputs, scene_object_checks_batch
from gr00t.rl.pick.geometry_views import TranslatedGeometry
from gr00t.rl.pick.scene_geometry import SURFACE_NAMES, SceneGeometry
from gr00t.rl.pick.tensor_planning import episode_in_world

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU scene admission")


def reference_object_checks(episode, state, geometry):
    """Original single-scene equations, retained only as an independent test oracle."""
    descriptors = [episode, *episode.get("distractors", [])]
    poses = torch.cat((state["object_state"][:, None], state["distractor_states"]), 1)
    positions = poses[..., :3]
    ends = positions.clone()
    ends[..., 2] = -0.20
    hits, _, _ = geometry.rays(positions, ends)
    labels = torch.where(
        (geometry.last_hit_types == 5) & (hits[..., 2] < 0.12), 0, geometry.last_hit_types
    )
    expected = labels.new_tensor(
        [SURFACE_NAMES.index(item["support_type"]) for item in descriptors]
    )
    categories = [object_category(item["asset"]) for item in descriptors]
    category_ids = labels.new_tensor([sorted(set(categories)).index(name) for name in categories])
    pairs = category_ids[None] * len(SURFACE_NAMES) + labels
    duplicate = pairs[..., :, None] == pairs[..., None, :]
    duplicate &= ~torch.eye(len(descriptors), device=poses.device, dtype=torch.bool)[None]
    initial = positions.new_tensor([item["object_position"] for item in descriptors])
    error = (positions[..., 2] - initial[:, 2]).abs()
    stable = (poses[..., 7:10].norm(dim=-1) < 0.03) & (poses[..., 10:13].norm(dim=-1) < 0.15)
    return {
        "object_support_labels_match": (labels == expected).all(-1),
        "object_category_support_pairs_unique": ~duplicate.any(dim=(-1, -2)),
        "all_object_support_errors_below_2cm": (error < 0.02).all(-1),
        "all_objects_stable": stable.all(-1),
    }, {
        "actual_support_type_ids": labels.cpu().tolist(),
        "object_support_error_m": error.cpu().tolist(),
        "all_object_speeds_m_s": poses[..., 7:10].norm(dim=-1).cpu().tolist(),
    }


def inputs():
    points = torch.tensor(
        [[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]], device="cuda", dtype=torch.float32
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], device="cuda")
    rooms = [
        SceneGeometry(points, faces, labels=torch.zeros(2, device="cuda", dtype=torch.long)),
        SceneGeometry(
            points + points.new_tensor([0, 0, 0.5]),
            faces,
            labels=torch.ones(2, device="cuda", dtype=torch.long),
        ),
        SceneGeometry(points, faces),
    ]
    origins = points.new_tensor([[0, 0, 0], [5, 4, 0], [-5, 0, 0]])

    def descriptor(category, support, position):
        return dict(
            asset={"name": category, "category": category},
            support_type=support,
            object_position=position,
            support_z=0.5 if support == "table" else 0.0,
        )

    episodes = [
        descriptor("orange", "floor", [0.3, -0.2, 0.03]),
        descriptor("orange", "table", [0.3, -0.2, 0.53]),
        descriptor("bottle", "floor", [0.3, -0.2, 0.03]),
    ]
    episodes[0]["distractors"] = [descriptor("apple", "floor", [-0.3, 0.2, 0.03])]
    episodes[1]["distractors"] = [descriptor("orange", "table", [-0.3, 0.2, 0.53])]
    for episode in episodes:
        episode.update(base_xy=[0, 0], station_xy=[0, 0])
    metadata = ExternalAdmissionInputs(episodes, origins, 2)
    root = points.new_zeros(3, 13)
    root[:, :3], root[:, 3] = origins, 1
    root[:, 2] = 0.53
    poses = points.new_zeros(3, 2, 13)
    poses[..., :3], poses[..., 3] = metadata.initial_positions, 1
    poses[2, 1] = torch.nan
    poses[0, 1, 7] = 0.04
    poses[1, 0, 2] += 0.03
    state = dict(root_states=root, object_state=poses[:, 0], distractor_states=poses[:, 1:])
    geometry = BatchedSceneGeometry(rooms, torch.arange(3, device="cuda"), origins)
    return episodes, origins, metadata, state, rooms, geometry


def test_external_admission_matches_scalar_gates_metrics_and_unobserved_slots(monkeypatch):
    episodes, origins, metadata, state, rooms, geometry = inputs()

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "External admission must not serialize decisions inside GPU predicates"
        )

    with monkeypatch.context() as scope:
        for name in ("item", "cpu", "tolist"):
            scope.setattr(torch.Tensor, name, forbidden)
        checks, metrics = scene_object_checks_batch(state, geometry, metadata)
    assert checks["all_objects_stable"].tolist() == [False, True, True]
    assert checks["object_category_support_pairs_unique"].tolist() == [True, False, True]
    assert checks["all_object_support_errors_below_2cm"].tolist() == [True, False, True]
    assert checks["object_support_labels_match"].all()
    for row in range(3):
        local = {key: value[row : row + 1] for key, value in state.items()}
        count = metadata.counts[row]
        local["distractor_states"] = local["distractor_states"][:, : count - 1]
        expected, expected_metrics = reference_object_checks(
            episode_in_world(episodes[row], origins[row]),
            local,
            TranslatedGeometry(rooms[row], origins[row]),
        )
        for key, value in expected.items():
            torch.testing.assert_close(checks[key][row : row + 1], value)
        for key in ("actual_support_type_ids", "object_support_error_m", "all_object_speeds_m_s"):
            torch.testing.assert_close(
                metrics[key][row : row + 1, :count],
                torch.tensor(expected_metrics[key], device="cuda"),
            )
    assert metrics["actual_support_type_ids"][2, 1] == -1
    assert torch.isnan(metrics["object_support_error_m"][2, 1])


def test_external_admission_is_independent_of_scene_order_and_padded_objects():
    episodes, origins, metadata, state, _, geometry = inputs()
    checks, metrics = scene_object_checks_batch(state, geometry, metadata)
    order = torch.tensor([2, 0, 1], device="cuda")
    shuffled = ExternalAdmissionInputs([episodes[i] for i in (2, 0, 1)], origins[order], 2)
    result, values = scene_object_checks_batch(
        {key: value[order] for key, value in state.items()}, geometry.select(order), shuffled
    )
    for key, value in checks.items():
        torch.testing.assert_close(result[key], value[order])
    for key, value in metrics.items():
        torch.testing.assert_close(values[key], value[order], equal_nan=True)
