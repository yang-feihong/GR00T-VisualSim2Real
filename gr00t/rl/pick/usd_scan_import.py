"""Offline, conservative conversion of opaque USD scans into the shared asset format."""

from pathlib import Path

import numpy as np


def linear_to_srgb(color):
    color = np.asarray(color, dtype=float)
    return np.where(color <= 0.0031308, color * 12.92, 1.055 * color ** (1 / 2.4) - 0.055)


def _connected_shader(value):
    from pxr import UsdShade

    source = value.GetConnectedSource()
    if source:
        return UsdShade.Shader(source[0].GetPrim())
    return None


def _texture_coordinates(value, uv):
    shader = _connected_shader(value)
    if not shader:
        raise ValueError("Texture coordinates must have an explicit primvar source")
    kind = shader.GetIdAttr().Get()
    if kind == "UsdPrimvarReader_float2":
        varname = shader.GetInput("varname")
        name = varname.Get()
        connections = varname.GetRawConnectedSourcePaths()
        if name is None and len(connections) == 1:
            name = shader.GetPrim().GetStage().GetAttributeAtPath(connections[0]).Get()
        if name != "st":
            raise ValueError(f"Unsupported texture primvar: {name}")
        return uv
    if kind != "UsdTransform2d":
        raise ValueError(f"Unsupported texture coordinate node: {kind}")
    result = _texture_coordinates(shader.GetInput("in"), uv)
    scale = shader.GetInput("scale").Get() or (1, 1)
    angle = np.deg2rad(shader.GetInput("rotation").Get() or 0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    translation = shader.GetInput("translation").Get() or (0, 0)
    return (result * np.asarray(scale)) @ rotation.T + np.asarray(translation)


def usd_visual_source(path, directory):
    """Bake transforms and UV nodes; reject material loss instead of using placeholders."""
    import trimesh
    from PIL import Image
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.Open(str(path))
    if not stage or UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError("USD scan importer requires a valid Z-up stage")
    if any(p.IsA(UsdPhysics.Joint) for p in stage.Traverse()):
        raise ValueError("Articulated targets cannot be flattened into a rigid scan")
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        raise ValueError("USD scan importer requires exactly one mesh")
    prim = meshes[0]
    mesh = UsdGeom.Mesh(prim)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    materials = {str(material.GetPath()): material} if material else {}
    covered = np.zeros(len(counts), dtype=bool)
    for subset in UsdGeom.Subset.GetGeomSubsets(mesh):
        bound, _ = UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()
        if bound:
            materials[str(bound.GetPath())] = bound
            covered[np.asarray(subset.GetIndicesAttr().Get(), dtype=int)] = True
    if not material and not covered.all():
        raise ValueError("Some mesh faces have no source material")
    if any("glass" in key.lower() for key in materials):
        raise ValueError("Glass scan requires a transparency-preserving multi-material importer")
    if len(materials) != 1:
        raise ValueError("Distinct per-face materials require explicit material atlas baking")
    material = next(iter(materials.values()))
    source = material.ComputeSurfaceSource()
    shader = source[0] if source and source[0] else None
    if not shader:
        shaders = [
            UsdShade.Shader(p) for p in Usd.PrimRange(material.GetPrim()) if p.IsA(UsdShade.Shader)
        ]
        if len(shaders) != 1:
            raise ValueError("Source material has no unambiguous surface shader")
        shader = shaders[0]
    kind = shader.GetIdAttr().Get()
    if kind not in (None, "UsdPreviewSurface"):
        raise ValueError(f"Unsupported source surface: {kind}")
    if kind is None and "OmniPBR" not in str(material.GetPath()):
        raise ValueError("Only opaque OmniPBR MDL materials are supported")
    for value in shader.GetInputs():
        if value.HasConnectedSource() and value.GetBaseName() != "diffuseColor":
            raise ValueError(f"Unsupported connected surface input: {value.GetBaseName()}")
        if kind is None and "texture" in value.GetBaseName().lower() and value.Get():
            raise ValueError("MDL texture input requires explicit material baking")
    opacity = shader.GetInput("opacity")
    if opacity and (opacity.HasConnectedSource() or opacity.Get() != 1):
        raise ValueError("Transparent source material is not an opaque grasp target")
    if np.any(counts < 3):
        raise ValueError("Source contains non-surface mesh faces")
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    source_indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    offsets = np.r_[0, np.cumsum(counts)[:-1]]
    triangle_corners = offsets[counts == 3, None] + np.arange(3)
    if np.any(counts == 4):
        quads = offsets[counts == 4, None] + np.arange(4)
        triangle_corners = np.concatenate(
            (triangle_corners, trimesh.geometry.triangulate_quads(quads))
        )
    if np.any(counts > 4):
        try:
            import mapbox_earcut
        except ImportError as error:
            raise ValueError("Offline polygon conversion requires mapbox-earcut==1.0.3") from error
        extra = []
        for offset, count in zip(offsets[counts > 4], counts[counts > 4]):
            polygon = points[source_indices[offset : offset + count]]
            centered = polygon - polygon.mean(0)
            _, _, basis = np.linalg.svd(centered, full_matrices=False)
            # Projection chooses diagonals only; original 3D vertices are never flattened.
            projected = np.ascontiguousarray(centered @ basis[:2].T)
            corners = mapbox_earcut.triangulate_float64(
                projected, np.array([count], dtype=np.uint32)
            ).reshape(-1, 3)
            normal = np.cross(polygon, np.roll(polygon, -1, axis=0)).sum(0)
            triangles = polygon[corners]
            orientation = np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ).sum(0)
            if normal @ orientation < 0:
                corners = corners[:, ::-1]
            if len(corners) != count - 2:
                raise ValueError("Source polygon triangulation is incomplete")
            extra.append(corners + offset)
        triangle_corners = np.concatenate((triangle_corners, *extra))
    faces = source_indices[triangle_corners]
    transform = np.asarray(mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    points = (np.c_[points, np.ones(len(points))] @ transform)[:, :3]
    points *= UsdGeom.GetStageMetersPerUnit(stage)
    color = shader.GetInput("diffuseColor" if kind else "diffuse_color_constant")
    texture = _connected_shader(color) if color else None
    if texture:
        if texture.GetIdAttr().Get() != "UsdUVTexture":
            raise ValueError("Diffuse color connection must be a USD texture")
        file = texture.GetInput("file").Get()
        if not file or not file.resolvedPath or not Path(file.resolvedPath).is_file():
            raise ValueError("Source texture is not resolved")
        with Image.open(file.resolvedPath) as image:
            if image.mode == "RGBA" and image.getextrema()[3] != (255, 255):
                raise ValueError("Source texture has non-opaque alpha")
            image = image.convert("RGB")
        if texture.GetInput("scale") or texture.GetInput("bias"):
            raise ValueError("Texture scale/bias requires explicit color baking")
        primvar = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
        if not primvar:
            raise ValueError("Textured source has no st primvar")
        uv = np.asarray(primvar.ComputeFlattened(), dtype=float)
        if primvar.GetInterpolation() in ("vertex", "varying"):
            uv = uv[faces]
        elif primvar.GetInterpolation() == "faceVarying":
            uv = uv[triangle_corners]
        else:
            raise ValueError("Unsupported UV interpolation")
        uv = _texture_coordinates(texture.GetInput("st"), uv)
    else:
        if not color or color.Get() is None:
            raise ValueError("Missing source diffuse color")
        rgb = np.round(linear_to_srgb(color.Get()).clip(0, 1) * 255).astype(int)
        image = Image.new("RGB", (2, 2), tuple(rgb))
        uv = np.zeros((*faces.shape, 2))
    if (mesh.GetOrientationAttr().Get() == "leftHanded") != (np.linalg.det(transform[:3, :3]) < 0):
        faces = faces[:, ::-1]
        uv = uv[:, ::-1]
    # Weld only identical position-and-UV tuples; texture seams remain distinct.
    packed = np.c_[points[faces].reshape(-1, 3), uv.reshape(-1, 2)]
    unique, indices = np.unique(packed, axis=0, return_inverse=True)
    visual = trimesh.Trimesh(unique[:, :3], indices.reshape(-1, 3), process=False)
    visual.visual = trimesh.visual.texture.TextureVisuals(uv=unique[:, 3:], image=image)
    output = Path(directory) / "source.obj"
    visual.export(output)
    texture_path = Path(directory) / "original_albedo.png"
    image.save(texture_path)
    return output, texture_path


def simplify_scan(path, directory, *, target_faces=20000, tolerance_m=0.0005):
    """Offline texture-aware decimation, accepted only after bidirectional surface checks."""
    import trimesh

    original = trimesh.load(path, force="mesh", process=False)
    if len(original.faces) <= target_faces:
        return path, {"source_faces": len(original.faces), "render_faces": len(original.faces)}
    try:
        import pymeshlab
    except ImportError as error:
        raise ValueError(
            "Offline dense-scan conversion requires pymeshlab==2023.12.post3"
        ) from error
    # Complexity adapts to source geometry, never to experiment seeds or grasp outcomes.
    for budget in (target_faces, target_faces * 2, target_faces * 4):
        if len(original.faces) <= budget:
            return path, {"source_faces": len(original.faces), "render_faces": len(original.faces)}
        simplifier = pymeshlab.MeshSet()
        simplifier.load_new_mesh(str(path))
        simplifier.apply_filter("meshing_remove_duplicate_vertices")
        simplifier.apply_filter(
            "meshing_decimation_quadric_edge_collapse_with_texture",
            targetfacenum=budget,
            preservenormal=True,
            preserveboundary=True,
            optimalplacement=True,
        )
        output = Path(directory) / "simplified.obj"
        simplifier.save_current_mesh(str(output))
        reduced = trimesh.load(output, force="mesh", process=False)
        if len(reduced.faces) > budget * 1.1:
            continue
        distances = []
        for mesh, reference in ((original, reduced), (reduced, original)):
            index = np.linspace(0, len(mesh.vertices) - 1, min(8192, len(mesh.vertices)), dtype=int)
            _, distance, _ = trimesh.proximity.closest_point(reference, mesh.vertices[index])
            distances.append(float(distance.max()))
        size_error = float(np.max(np.abs(original.extents - reduced.extents)))
        if max(distances) <= tolerance_m and size_error <= tolerance_m:
            return output, {
                "source_faces": len(original.faces),
                "render_faces": len(reduced.faces),
                "sampled_bidirectional_max_error_m": max(distances),
                "extent_max_error_m": size_error,
                "surface_samples_each_direction": 8192,
            }
    raise ValueError(f"Render LOD could not satisfy {tolerance_m} m within the face budget")
