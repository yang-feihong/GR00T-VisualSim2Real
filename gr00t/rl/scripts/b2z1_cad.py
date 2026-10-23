import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS


def read_step(source):
    reader = STEPControl_Reader()
    if reader.ReadFile(str(source)) != IFSelect_RetDone:
        raise RuntimeError(f"Cannot read STEP file: {source}")
    if reader.TransferRoots() == 0:
        raise RuntimeError("STEP file has no transferable roots")
    shape = reader.OneShape()
    bounds = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, bounds)
    minimum, maximum = bounds.CornerMin(), bounds.CornerMax()
    bounds_mm = np.array([[minimum.X(), minimum.Y(), minimum.Z()],
                          [maximum.X(), maximum.Y(), maximum.Z()]])
    origin_mm = np.array([bounds_mm[:, 0].mean(), bounds_mm[:, 1].mean(), bounds_mm[0, 2]])
    BRepMesh_IncrementalMesh(shape, 0.18, False, 0.25, True).Perform()
    solids = []
    solid_explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while solid_explorer.More():
        points, faces = [], []
        face_explorer = TopExp_Explorer(solid_explorer.Current(), TopAbs_FACE)
        while face_explorer.More():
            face = TopoDS.Face(face_explorer.Current())
            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation_s(face, location)
            if triangulation is None:
                raise RuntimeError("CAD face could not be tessellated")
            offset = len(points)
            for index in range(1, triangulation.NbNodes() + 1):
                point = triangulation.Node(index).Transformed(location.Transformation())
                points.append([point.X(), point.Y(), point.Z()])
            for index in range(1, triangulation.NbTriangles() + 1):
                triangle = triangulation.Triangle(index)
                corners = [triangle.Value(corner) - 1 + offset for corner in (1, 2, 3)]
                if face.Orientation() == TopAbs_REVERSED:
                    corners.reverse()
                faces.append(corners)
            face_explorer.Next()
        if not faces:
            raise RuntimeError("CAD solid contains no triangles")
        points = (np.asarray(points) - origin_mm) * 0.001
        faces = np.asarray(faces, dtype=np.int32)
        triangles = points[faces]
        areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                       triangles[:, 2] - triangles[:, 0]), axis=1)
        faces = faces[areas > 1e-16]
        solids.append((points.astype(np.float32), faces))
        solid_explorer.Next()
    if not solids:
        raise RuntimeError("No solid geometry found in STEP file")
    return solids, bounds_mm, origin_mm
