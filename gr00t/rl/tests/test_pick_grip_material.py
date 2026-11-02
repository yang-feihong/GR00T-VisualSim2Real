"""Offline USD resolution of the grip material on actual target collision meshes."""

from pathlib import Path

import pytest
from pxr import Usd, UsdPhysics, UsdShade

from gr00t.rl.pick.pocket import GRIP_DYNAMIC_FRICTION, GRIP_STATIC_FRICTION


def test_grip_material_override_resolves_on_actual_target_collision_meshes():
    asset = Path(__file__).parents[1] / "data/pick_assets/gso_SNAIL_MEASURING_TAPE/object.usda"
    if not asset.is_file():
        pytest.skip("downloaded target asset unavailable")
    stage = Usd.Stage.CreateInMemory()
    body = stage.DefinePrim("/World/PickObject", "Xform")
    body.GetReferences().AddReference(str(asset))
    material = UsdShade.Material.Define(stage, "/World/PickGripMaterial")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(GRIP_STATIC_FRICTION)
    physics.CreateDynamicFrictionAttr(GRIP_DYNAMIC_FRICTION)
    UsdShade.MaterialBindingAPI.Apply(body).Bind(
        material, UsdShade.Tokens.strongerThanDescendants, "physics"
    )
    colliders = [prim for prim in Usd.PrimRange(body) if prim.HasAPI(UsdPhysics.CollisionAPI)]
    assert colliders
    for collider in colliders:
        bound, relationship = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial("physics")
        resolved = UsdPhysics.MaterialAPI(bound.GetPrim())
        print(
            {
                "collider": str(collider.GetPath()),
                "binding": str(relationship.GetPath()),
                "dynamic_friction": resolved.GetDynamicFrictionAttr().Get(),
                "static_friction": resolved.GetStaticFrictionAttr().Get(),
            }
        )
        assert resolved.GetDynamicFrictionAttr().Get() == pytest.approx(GRIP_DYNAMIC_FRICTION)
        assert resolved.GetStaticFrictionAttr().Get() == pytest.approx(GRIP_STATIC_FRICTION)
