from datetime import datetime, timezone
import xml.etree.ElementTree as ElementTree


COLLADA_NAMESPACE = "http://www.collada.org/2005/11/COLLADASchema"


def read_dae(path):
    root = ElementTree.parse(path).getroot()
    namespaces = {"dae": COLLADA_NAMESPACE}
    unit = root.find("dae:asset/dae:unit", namespaces)
    up_axis = root.findtext("dae:asset/dae:up_axis", namespaces=namespaces)
    if unit is None or float(unit.get("meter")) != 1 or up_axis != "Z_UP":
        raise ValueError(f"Expected meter units and Z_UP in {path}")
    mesh = root.find("dae:library_geometries/dae:geometry/dae:mesh", namespaces)
    vertices = mesh.find("dae:vertices/dae:input[@semantic='POSITION']", namespaces)
    source = mesh.find(f"dae:source[@id='{vertices.get('source')[1:]}']/dae:float_array", namespaces)
    coordinates = [float(value) for value in source.text.split()]
    points = [tuple(coordinates[index:index + 3]) for index in range(0, len(coordinates), 3)]
    triangles = mesh.find("dae:triangles", namespaces)
    inputs = triangles.findall("dae:input", namespaces)
    if len(inputs) != 1 or inputs[0].get("semantic") != "VERTEX" or inputs[0].get("offset") != "0":
        raise ValueError(f"Unsupported triangle inputs in {path}")
    indices = [int(value) for value in triangles.findtext("dae:p", namespaces=namespaces).split()]
    if len(indices) != 3 * int(triangles.get("count")) or any(index < 0 or index >= len(points) for index in indices):
        raise ValueError(f"Invalid triangle indices in {path}")
    return points, [indices[index:index + 3] for index in range(0, len(indices), 3)]


def write_dae(path, points, faces, color):
    ElementTree.register_namespace("", COLLADA_NAMESPACE)
    def element(parent, name, attributes=None, text=None):
        child = ElementTree.SubElement(parent, f"{{{COLLADA_NAMESPACE}}}{name}", attributes or {})
        child.text = text
        return child

    root = ElementTree.Element(f"{{{COLLADA_NAMESPACE}}}COLLADA", {"version": "1.4.1"})
    asset = element(root, "asset")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    element(asset, "created", text=timestamp)
    element(asset, "modified", text=timestamp)
    element(asset, "unit", {"name": "meter", "meter": "1"})
    element(asset, "up_axis", text="Z_UP")
    effect = element(element(root, "library_effects"), "effect", {"id": "back-box-effect"})
    technique = element(element(effect, "profile_COMMON"), "technique", {"sid": "common"})
    diffuse = element(element(technique, "lambert"), "diffuse")
    element(diffuse, "color", text=" ".join(str(value) for value in (*color, 1)))
    material = element(element(root, "library_materials"), "material", {"id": "back-box-material"})
    element(material, "instance_effect", {"url": "#back-box-effect"})
    geometry = element(element(root, "library_geometries"), "geometry", {"id": "back-box-geometry"})
    mesh = element(geometry, "mesh")
    source = element(mesh, "source", {"id": "back-box-positions"})
    element(source, "float_array", {"id": "back-box-positions-array", "count": str(len(points) * 3)},
            " ".join(str(float(value)) for point in points for value in point))
    accessor = element(element(source, "technique_common"), "accessor", {
        "source": "#back-box-positions-array", "count": str(len(points)), "stride": "3",
    })
    for axis in ("X", "Y", "Z"):
        element(accessor, "param", {"name": axis, "type": "float"})
    vertices = element(mesh, "vertices", {"id": "back-box-vertices"})
    element(vertices, "input", {"semantic": "POSITION", "source": "#back-box-positions"})
    triangles = element(mesh, "triangles", {"count": str(len(faces)), "material": "back-box-silver"})
    element(triangles, "input", {"semantic": "VERTEX", "source": "#back-box-vertices", "offset": "0"})
    element(triangles, "p", text=" ".join(str(int(index)) for face in faces for index in face))
    scene = element(element(root, "library_visual_scenes"), "visual_scene", {"id": "Scene"})
    instance = element(element(scene, "node", {"id": "back-box"}), "instance_geometry", {"url": "#back-box-geometry"})
    binding = element(element(instance, "bind_material"), "technique_common")
    element(binding, "instance_material", {"symbol": "back-box-silver", "target": "#back-box-material"})
    element(element(root, "scene"), "instance_visual_scene", {"url": "#Scene"})
    ElementTree.indent(root, space="  ")
    ElementTree.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def main():
    from b2z1_cad import read_step
    from gr00t.rl.utils.b2z1_asset import BOX_COLOR, ROBOT_RESOURCES

    solids, _, _ = read_step(ROBOT_RESOURCES / "cad/back_box.step")
    points, faces = [], []
    for vertices, indices in solids:
        offset = len(points)
        points.extend(vertices)
        faces.extend(indices + offset)
    write_dae(ROBOT_RESOURCES / "meshes/back_box.dae", points, faces, BOX_COLOR)


if __name__ == "__main__":
    main()
