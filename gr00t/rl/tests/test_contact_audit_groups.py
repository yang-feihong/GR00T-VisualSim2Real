"""One audit implementation supports heterogeneous scene collider counts."""

from types import SimpleNamespace

from gr00t.rl.data_collection.contact_audit import _create_contact_views


def test_heterogeneous_views_preserve_every_filter_and_actual_sensor_order():
    calls = []

    def create(patterns, *, filter_patterns, max_contact_data_count):
        calls.append((patterns, filter_patterns, max_contact_data_count))
        return SimpleNamespace(sensor_paths=list(reversed(patterns)))

    patterns = [
        "/World/envs/env_0/Robot/a",
        "/World/envs/env_1/Robot/a",
        "/World/envs/env_2/Robot/a",
    ]
    filters = [["wall0", "object0"], ["wall1", "object1", "table1"], ["wall2", "object2", "table2"]]
    groups = _create_contact_views(
        SimpleNamespace(create_rigid_contact_view=create), patterns, filters, 1024
    )
    assert len(groups) == 2
    restored = {
        path: paths for _, sensors, values in groups for path, paths in zip(sensors, values)
    }
    assert restored == dict(zip(patterns, map(tuple, filters)))
    assert [call[2] for call in calls] == [1024, 2048]


def test_homogeneous_scenes_keep_one_batched_physx_view():
    calls = []

    def create(patterns, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(sensor_paths=patterns)

    patterns = [f"/World/envs/env_{row}/Robot/a" for row in range(250)]
    groups = _create_contact_views(
        SimpleNamespace(create_rigid_contact_view=create),
        patterns,
        [["wall", "object"]] * 250,
        1024,
    )
    assert len(groups) == 1
    assert len(groups[0][1]) == 250
    assert calls[0]["max_contact_data_count"] == 256000
