"""All current supports are fully occupied, not hanging/overhanging placements."""

import pytest

from gr00t.rl.pick import external_preparation as implementation
from gr00t.rl.pick import prepare


@pytest.mark.parametrize("support", ["any", "floor", "table", "chair", "furniture"])
def test_unsupported_plane_targets_are_not_randomly_selected(monkeypatch, tmp_path, support):
    bad = {"name": "unsupported", "support_plane_admitted": False}
    good = {"name": "supported", "support_plane_admitted": True}
    monkeypatch.setattr(implementation, "discover_vlnverse_scenes", lambda *a, **kw: [])
    monkeypatch.setattr(implementation, "episode_asset_pool", lambda *a, **kw: [bad, good])
    chosen = []

    def materialize(asset, seed):
        chosen.append(asset["name"])
        return asset

    monkeypatch.setattr(implementation, "materialize_variant", materialize)
    with pytest.raises(prepare.NoValidInitialScene):
        prepare.prepare_episode(12, tmp_path, tmp_path / "episode.json", support=support)
    assert chosen == ["supported"]


def test_explicit_unsupported_furniture_target_is_rejected(monkeypatch, tmp_path):
    bad = {"name": "unsupported", "support_plane_admitted": False}
    good = {"name": "supported", "support_plane_admitted": True}
    monkeypatch.setattr(implementation, "discover_vlnverse_scenes", lambda *a, **kw: [])
    monkeypatch.setattr(implementation, "episode_asset_pool", lambda *a, **kw: [bad, good])
    monkeypatch.setattr(implementation, "installed_assets", lambda: [bad, good])
    monkeypatch.setattr(implementation, "materialize_variant", lambda asset, seed: asset)
    with pytest.raises(prepare.NoValidInitialScene, match="no admitted support-plane grasp"):
        prepare.prepare_episode(
            12, tmp_path, tmp_path / "episode.json", support="table", asset_name="unsupported"
        )


def test_installed_asset_resampling_does_not_regenerate_unused_variants(monkeypatch, tmp_path):
    monkeypatch.setattr(implementation, "discover_vlnverse_scenes", lambda *a, **kw: [])

    def forbidden(*args, **kwargs):
        raise AssertionError("Installed task assets must not rebuild the random variant pool")

    monkeypatch.setattr(implementation, "episode_asset_pool", forbidden)
    monkeypatch.setattr(implementation, "materialize_variant", forbidden)
    with pytest.raises(prepare.NoValidInitialScene, match="No valid initial scene"):
        prepare.prepare_episode(
            12,
            tmp_path,
            tmp_path / "episode.json",
            fixed_asset={"name": "supported", "support_plane_admitted": True},
            distractor_assets=[],
        )
