"""Asset diversity must retain identity and agree across rendering, physics and planning."""

import copy
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from gr00t.rl.pick.asset_library import (
    acquire,
    episode_asset_pool,
    installed_assets,
    materialize_variant,
    safe_extract,
    sample_instances,
    source_entry,
    target_shape_eligible,
    variant_parameters,
)
from gr00t.rl.pick.assets import prepare_mesh_asset
from gr00t.rl.pick.distractors import target_semantics


@pytest.mark.parametrize(
    "size,expected",
    [
        ([0.055, 0.060, 0.065], True),
        ([0.02, 0.02, 0.15], True),
        ([0.016, 0.08, 0.12], False),
        ([0.005, 0.05, 0.05], False),
        ([0.11, 0.12, 0.13], False),
        ([0.03, 0.04, float("nan")], False),
    ],
)
def test_target_shapes_require_two_graspable_dimensions_without_plate_geometry(size, expected):
    assert target_shape_eligible(size) is expected


@pytest.fixture
def scan(tmp_path):
    import trimesh

    directory = tmp_path / "test_apple"
    directory.mkdir()
    mesh_path = directory / "source.obj"
    trimesh.creation.box(extents=[0.055, 0.060, 0.065]).export(mesh_path)
    return prepare_mesh_asset(
        "test_apple",
        directory,
        mesh_path,
        provenance={"category": "apple", "instance_id": "test_apple", "archive_sha256": "test"},
    )


def test_category_balanced_instance_selection_is_seed_local():
    assets = [{"name": f"apple{i}", "category": "apple"} for i in range(20)]
    assets += [{"name": "lemon", "category": "lemon"}]
    first = sample_instances(assets, 1700)
    assert [a["category"] for a in first] == ["apple", "lemon"]
    np.random.seed(99)
    np.random.rand(100)
    assert sample_instances(assets, 1700) == first
    assert len({sample_instances(assets, seed)[0]["name"] for seed in range(100)}) > 10


def test_randomization_is_repeatable_varies_across_seeds_and_preserves_identity(scan):
    original = copy.deepcopy(scan)
    a = variant_parameters(scan, 1700)
    assert a == variant_parameters(scan, 1700)
    assert a != variant_parameters(scan, 1701)
    assert a != variant_parameters(scan, 1700, 1)
    assert np.max(np.abs(np.asarray(a["scale_xyz"]) - 1)) < 0.15
    assert np.min(a["texture_rgb_multiplier"]) >= 0.88
    assert scan == original


def test_episode_shape_is_selected_only_from_gpu_admitted_bank(scan):
    bank = [[0.91, 0.93, 0.95], [0.96, 0.98, 0.97]]
    asset = {**scan, "admitted_scale_xyz": bank}
    scales = [variant_parameters(asset, seed)["scale_xyz"] for seed in range(100)]
    assert all(scale in bank for scale in scales)
    assert len({tuple(scale) for scale in scales}) == 2


def test_size_and_stable_face_are_sampled_as_a_joint_admitted_pair(scan):
    rotation = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    bank = [
        {"scale_xyz": [0.91, 0.93, 0.95], "poses": [{"rotation": np.eye(3).tolist()}]},
        {"scale_xyz": [1.03, 1.02, 1.01], "poses": [{"rotation": rotation}]},
    ]
    asset = {**scan, "supported_placement_shapes": bank}
    seen = set()
    for seed in range(100):
        params = variant_parameters(asset, seed)
        index = next(i for i, entry in enumerate(bank) if entry["scale_xyz"] == params["scale_xyz"])
        assert params["canonical_rotation"] == bank[index]["poses"][0]["rotation"]
        assert params["support_plane_admitted"]
        seen.add(index)
    assert seen == {0, 1}


def test_basket_admission_uses_its_own_joint_shape_bank_without_changing_external(scan, tmp_path):
    generic = [{"scale_xyz": [0.91, 0.93, 0.95], "poses": [{"rotation": np.eye(3).tolist()}]}]
    basket = [{"scale_xyz": [1.03, 1.02, 1.01], "poses": [{"rotation": np.eye(3).tolist()}]}]
    source = {
        **scan,
        "supported_placement_shapes": generic,
        "placement_profiles": {"basket": {"supported_shapes": basket}},
    }
    Path(scan["usd_path"]).with_name("asset.json").write_text(json.dumps(source))
    external = variant_parameters(source, 1700)
    params = variant_parameters(source, 1700, placement_profile="basket")
    assert external["scale_xyz"] == generic[0]["scale_xyz"]
    assert params["scale_xyz"] == basket[0]["scale_xyz"]
    assert "placement_profile" not in external
    preview = episode_asset_pool(1700, root=tmp_path, placement_profile="basket")[0]
    actual = materialize_variant(preview, 0, root=tmp_path)
    assert actual["randomization"] == params
    assert actual["size_m"] == pytest.approx(preview["size_m"])
    assert actual["mass_kg"] == pytest.approx(preview["mass_kg"])
    assert variant_parameters(source, 1700) == external


def test_missing_basket_profile_is_not_treated_as_external_admission(scan, tmp_path):
    with pytest.raises(ValueError, match="No admitted basket"):
        variant_parameters(scan, 1700, placement_profile="basket")
    assert episode_asset_pool(1700, root=tmp_path, placement_profile="basket") == []


def test_offline_placement_support_is_identical_to_the_materialized_asset(tmp_path):
    import trimesh

    from gr00t.rl.pick.grasp_quality import load_grasp_support
    from gr00t.rl.pick.supported_asset_audit import baked_pose_support

    directory = tmp_path / "dense_apple"
    directory.mkdir()
    mesh = trimesh.creation.icosphere(subdivisions=3)
    mesh.apply_scale([0.0275, 0.03, 0.0325])
    mesh_path = directory / "source.obj"
    mesh.export(mesh_path)
    scan = prepare_mesh_asset("dense_apple", directory, mesh_path, provenance={"category": "apple"})
    rotation = trimesh.transformations.euler_matrix(0.4, 0.7, -0.2)[:3, :3]
    source = {**scan, "stable_placement_rotations": [rotation.tolist()]}
    params = variant_parameters(source, 1700)
    actual = materialize_variant(source, 1700, root=tmp_path)
    directory = Path(source["usd_path"]).parent
    points = np.load(directory / "collision_support.npy") * params["scale_xyz"]
    faces = np.load(directory / "collision_faces.npy")
    audited = baked_pose_support(points, faces, rotation, "cpu")
    loaded = load_grasp_support(actual["usd_path"], "cpu")
    assert len(loaded) > 384
    torch.testing.assert_close(audited, loaded, rtol=0, atol=0)
    assert len(loaded) == len(points) + 4 * len(faces)
    original_vertices = np.load(Path(actual["usd_path"]).with_name("collision_support.npy"))
    torch.testing.assert_close(
        loaded[: len(points)], torch.as_tensor(original_vertices), rtol=0, atol=0
    )


def test_no_infinite_plane_solution_does_not_disable_furniture_edge_assets(scan, tmp_path):
    source = {**scan, "enabled": True, "supported_placement_shapes": []}
    Path(scan["usd_path"]).with_name("asset.json").write_text(json.dumps(source))
    preview = episode_asset_pool(1700, root=tmp_path)[0]
    assert preview["support_plane_admitted"] is False
    asset = materialize_variant(preview, 0, root=tmp_path)
    assert asset["enabled"]
    assert asset["support_plane_admitted"] is False
    assert installed_assets(tmp_path) == [source]


def test_force_bound_uses_actual_jaw_opening_not_a_fictitious_flat_object_diameter():
    from gr00t.rl.pick.planning import minimum_grip_normal_force

    force = minimum_grip_normal_force([0.12, 0.08, 0.016], 0.1, aperture_angle_rad=math.pi / 3)
    assert force == pytest.approx(0.1 * 9.81 / (0.75 * math.cos(math.pi / 6) - 0.5))


def test_thin_asset_is_excluded_even_when_legacy_metadata_marks_it_enabled(scan, tmp_path):
    path = Path(scan["usd_path"]).with_name("asset.json")
    metadata = {**scan, "enabled": True, "size_m": [0.12, 0.08, 0.016]}
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="No installed pick assets"):
        installed_assets(tmp_path)
    assert len(installed_assets(tmp_path, include_disabled=True)) == 1


def test_visual_collider_mass_and_planner_hull_use_identical_variant(scan, tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics

    asset = materialize_variant(scan, 1700, root=tmp_path)
    assert materialize_variant(scan, 1700, root=tmp_path) == asset
    scale = np.asarray(asset["randomization"]["scale_xyz"])
    directory = Path(asset["usd_path"]).parent
    hull = np.load(directory / "collision_support.npy")
    source_hull = np.load(Path(scan["usd_path"]).with_name("collision_support.npy"))
    np.testing.assert_allclose(hull, source_hull * scale, atol=1e-8)
    stage = Usd.Stage.Open(asset["usd_path"])
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Object/Scan"))
    points = np.asarray(mesh.GetPointsAttr().Get())
    np.testing.assert_allclose(points.ptp(0), asset["size_m"], atol=1e-8)
    assert mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
    mass = UsdPhysics.MassAPI(stage.GetPrimAtPath("/Object")).GetMassAttr().Get()
    assert mass == pytest.approx(scan["mass_kg"] * scale.prod())
    assert asset["mass_kg"] == pytest.approx(mass)
    assert asset["source_instance_id"] == scan["instance_id"]
    assert not (directory / "albedo.png").exists()
    assert installed_assets(tmp_path) == [scan]


def test_pool_defers_geometry_work_until_only_chosen_assets_are_materialized(scan, tmp_path):
    pool = episode_asset_pool(1700, root=tmp_path)
    assert len(pool) == 1
    assert not (tmp_path / ".variants").exists()
    asset = materialize_variant(pool[0], 999, root=tmp_path)
    np.testing.assert_allclose(asset["size_m"], pool[0]["size_m"], atol=1e-8)
    assert "_variant_source" not in asset
    assert asset["randomization"] == variant_parameters(scan, 1700, 0)


def test_arbitrary_resting_rotation_preserves_preview_and_recenters_both_meshes(scan, tmp_path):
    from pxr import Usd, UsdGeom

    angle = 0.37
    rotation = [
        [1, 0, 0],
        [0, math.cos(angle), -math.sin(angle)],
        [0, math.sin(angle), math.cos(angle)],
    ]
    source = {**scan, "stable_placement_rotations": [rotation]}
    Path(scan["usd_path"]).with_name("asset.json").write_text(json.dumps(source))
    preview = episode_asset_pool(1700, root=tmp_path)[0]
    asset = materialize_variant(preview, 0, root=tmp_path)
    assert asset["randomization"]["canonical_rotation"] == rotation
    stage = Usd.Stage.Open(asset["usd_path"])
    visual = np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath("/Object/Scan")).GetPointsAttr().Get())
    hull = np.load(Path(asset["usd_path"]).with_name("collision_support.npy"))
    np.testing.assert_allclose(preview["size_m"], asset["size_m"], atol=2e-8)
    np.testing.assert_allclose(
        asset["intrinsic_size_m"], np.asarray(scan["size_m"]) * asset["randomization"]["scale_xyz"]
    )
    assert not np.allclose(asset["intrinsic_size_m"], asset["size_m"])
    np.testing.assert_allclose(hull.min(0), visual.min(0), atol=2e-8)
    np.testing.assert_allclose(hull.max(0), visual.max(0), atol=2e-8)
    np.testing.assert_allclose(hull.min(0) + hull.max(0), 0, atol=2e-8)


def test_variant_parameters_are_recorded_in_target_semantics(scan, tmp_path):
    asset = materialize_variant(scan, 1700, root=tmp_path)
    target = target_semantics({"asset": asset, "support_type": "floor"})
    assert target["object_category"] == "apple"
    assert target["source_instance_id"] == "test_apple"
    assert target["variant_id"] == asset["variant_id"]
    assert target["randomization"] == asset["randomization"]
    assert target["size_m"] == asset["size_m"]


def test_archive_cannot_escape_destination(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("../escaped", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract(archive, tmp_path / "extract")
    assert not (tmp_path / "escaped").exists()


def test_original_gso_license_is_required_before_conversion(tmp_path):
    entry = source_entry({"source": "gso", "model": "bad", "category": "toy"})
    directory = tmp_path / ".downloads" / entry["name"]
    directory.mkdir(parents=True)
    with zipfile.ZipFile(directory / "source.zip", "w") as out:
        out.writestr("metadata.pbtxt", 'license: "unknown"')
    with pytest.raises(ValueError, match="CC BY 4.0"):
        acquire(entry, tmp_path)
    assert not (tmp_path / entry["name"] / "asset.json").exists()


def test_excluded_scan_cannot_remain_enabled_after_annotation_sync(tmp_path):
    entry = source_entry(
        {
            "source": "gso",
            "model": "two_objects",
            "category": "toy car",
            "exclude_reason": "two separate cars",
        }
    )
    directory = tmp_path / entry["name"]
    directory.mkdir()
    (directory / "asset.json").write_text(json.dumps({"name": entry["name"], "enabled": True}))
    assert not acquire(entry, tmp_path)["enabled"]


def test_manifest_contains_real_instance_diversity_and_pinned_sources():
    from gr00t.rl.pick import asset_library

    manifest = json.loads(Path(asset_library.__file__).with_name("asset_sources.json").read_text())
    entries = [source_entry(e) for e in manifest["assets"]]
    assert len({e["category"] for e in entries}) >= 30
    assert len({e["name"] for e in entries}) == len(entries)
    assert sum(e["category"] == "computer mouse" for e in entries) >= 3
    assert sum(e["category"] == "robot figurine" for e in entries) >= 3
    assert all(asset_library.GSO_REVISION in e["url"] for e in entries if e["source"] == "gso")


def test_impossible_source_cannot_be_rescued_by_rescaling(tmp_path):
    import trimesh

    mesh = tmp_path / "huge.obj"
    trimesh.creation.box(extents=[1, 1, 1]).export(mesh)
    with pytest.raises(ValueError, match="not a small graspable"):
        prepare_mesh_asset("huge", tmp_path, mesh, provenance={})
