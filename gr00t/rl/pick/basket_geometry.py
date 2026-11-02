"""Reuse room and object BVHs; bind current basket/distractor frames on the GPU."""

import torch

from gr00t.rl.pick.batched_geometry import BatchedCompositeGeometry, BatchedSceneGeometry
from gr00t.rl.pick.scene_object_geometry import SceneObjectGeometryBatch, load_collider_mesh


class BatchedBasketPlanningGeometry(BatchedCompositeGeometry):
    def __init__(self, nonrobot, basket_world, basket_model):
        super().__init__((nonrobot, basket_world))
        self.nonrobot_geometry = nonrobot
        self.basket_world = basket_world
        self.basket_model = basket_model

    def select(self, rows):
        return type(self)(
            self.nonrobot_geometry.select(rows), self.basket_world.select(rows), self.basket_model
        )


class BasketGeometryBatch(SceneObjectGeometryBatch):
    """Load immutable asset metadata once, then update only device pose tensors."""

    def __init__(self, controllers, *, mesh_loader=load_collider_mesh):
        super().__init__(
            [c.episode for c in controllers],
            BatchedSceneGeometry.from_views([c.static_scene for c in controllers]),
            mesh_loader=mesh_loader,
        )
        device = self.rooms.device
        origins = torch.zeros(self.count, 3, device=device)
        model = controllers[0].basket
        self.basket = BatchedSceneGeometry(
            [model.geometry], torch.zeros(self.count, dtype=torch.long, device=device), origins
        )
        self.geometry = BatchedBasketPlanningGeometry(self.nonrobot_geometry, self.basket, model)

    def update(self, state):
        super().update(state)
        roots = state["root_states"]
        self._bind(self.basket, roots[:, :3], roots[:, 3:7])
        return self.geometry

    def with_target(self, state):
        """Bind the target collider for release/entry checks without rebuilding a BVH."""
        nonrobot, target = super().with_target(state)
        geometry = BatchedBasketPlanningGeometry(
            nonrobot,
            self.basket,
            self.geometry.basket_model,
        )
        return geometry, target
