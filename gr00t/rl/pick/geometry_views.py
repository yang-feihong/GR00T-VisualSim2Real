"""Share static BVHs across episodes without copying full rooms on each replan."""

import torch

from gr00t.rl.pick.scene_geometry import SceneGeometry


class TranslatedGeometry:
    def __init__(self, source, origin):
        self.source, self.origin, self.device = source, origin, source.device
        self.labels = source.labels

    @property
    def points(self):
        return self.source.points + self.origin

    @property
    def faces(self):
        return self.source.faces

    def rays(self, origins, targets):
        hit, normal, distance = self.source.rays(origins - self.origin, targets - self.origin)
        self.last_hit_types = self.source.last_hit_types
        return hit + self.origin, normal, distance

    def visible(self, origins, targets, tolerance_m=0.01):
        return self.rays(origins, targets)[2] >= (targets - origins).norm(dim=-1) - tolerance_m

    def distances(self, points):
        return self.source.distances(points - self.origin)

    def rigid_surface_clearance(self, positions, orientations, local_points, **kwargs):
        return self.source.rigid_surface_clearance(
            positions - self.origin, orientations, local_points, **kwargs
        )

    def with_obstacles(self, obstacles):
        return CompositeGeometry(
            (self, SceneGeometry(obstacles["points"], obstacles["faces"], self.device))
        )


class CompositeGeometry:
    def __init__(self, components):
        self.components = tuple(components)
        self.device = self.components[0].device
        self.labels = None

    @property
    def points(self):
        return torch.cat([part.points for part in self.components])

    @property
    def faces(self):
        offset, parts = 0, []
        for part in self.components:
            parts.append(part.faces.reshape(-1, 3) + offset)
            offset += len(part.points)
        return torch.cat(parts).flatten()

    def with_obstacles(self, obstacles):
        return CompositeGeometry(
            (*self.components, SceneGeometry(obstacles["points"], obstacles["faces"], self.device))
        )

    def rays(self, origins, targets):
        hit, normal, distance = self.components[0].rays(origins, targets)
        labels = self.components[0].last_hit_types.clone()
        for part in self.components[1:]:
            other, other_normal, other_distance = part.rays(origins, targets)
            closer = other_distance < distance
            hit = torch.where(closer[..., None], other, hit)
            normal = torch.where(closer[..., None], other_normal, normal)
            distance = torch.minimum(distance, other_distance)
            labels = torch.where(closer, part.last_hit_types, labels)
        self.last_hit_types = labels
        return hit, normal, distance

    def visible(self, origins, targets, tolerance_m=0.01):
        return self.rays(origins, targets)[2] >= (targets - origins).norm(dim=-1) - tolerance_m

    def distances(self, points):
        return torch.stack([part.distances(points) for part in self.components]).amin(0)

    def rigid_surface_clearance(self, positions, orientations, local_points, **kwargs):
        return torch.stack(
            [
                part.rigid_surface_clearance(positions, orientations, local_points, **kwargs)
                for part in self.components
            ]
        ).amin(0)
