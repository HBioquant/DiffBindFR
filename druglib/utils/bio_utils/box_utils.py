# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Union, Tuple
from pymol import cmd
import numpy as np


def getbox(
        selection: str = "sele",
        extending: Union[float, int] = 6.0,
        docking_software: str = "vina",
) -> dict:
    # check docking software available
    if docking_software not in ['vina', 'ledock', 'smina', 'linf9', 'rbdock', 'gnina']:
        raise ValueError(f"your desired {docking_software} can not be available in the present, "
                         f"while \'vina\', \'ledock\', \'smina\', "
                         f"\'linf9\', \'rbdock\', \'gnina\' is available now.")

    ([minX, minY, minZ], [maxX, maxY, maxZ]) = cmd.get_extent(selection=selection)

    # expanding the box boundary
    minX = minX - float(extending)
    minY = minY - float(extending)
    minZ = minZ - float(extending)
    maxX = maxX + float(extending)
    maxY = maxY + float(extending)
    maxZ = maxZ + float(extending)

    # get box size and center
    SizeX = maxX - minX
    SizeY = maxY - minY
    SizeZ = maxZ - minZ
    CenterX = (minX + maxX) / 2
    CenterY = (minY + maxY) / 2
    CenterZ = (minZ + maxZ) / 2

    cmd.delete("all")

    output1 = {
        'center_x': CenterX,
        'center_y': CenterY,
        'center_z': CenterZ
    }, {
       'size_x' : SizeX,
       'size_y' : SizeY,
       'size_z' : SizeZ
    }
    output2 = {
        'minX' : minX,
        'maxX' : maxX
    }, {
        'minY': minY,
        'maxY': maxY
    }, {
        'minZ': minZ,
        'maxZ': maxZ
    }
    if docking_software == "vina":
        return output1
    elif docking_software == 'ledock':
        return output2
    else:
        raise NotImplementedError

def compute_protein_bbox(
        prot_coords: np.ndarray,
) -> np.ndarray:
  """
  Compute the protein axis aligned bounding box
  Args:
      prot_coords: np.ndarray. A numpy array of shape `(N, 3)`,
        where `N` is the number of atoms.
  Returns:
      protein_range: np.ndarray. A numpy array of shape `(3,)`,
        where `3` is (x,y,z).
  """
  protein_max = np.max(prot_coords, axis=0)
  protein_min = np.min(prot_coords, axis=0)
  protein_bbox = protein_max - protein_min
  return protein_bbox

def compute_protein_bbox_open3d(
        prot_coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the protein axis aligned bounding box (aabb) and
        rotated bounding box (obb)
    Args:
      prot_coords: np.ndarray. A numpy array of shape `(N, 3)`,
        where `N` is the number of atoms.
    Returns:
      protein_range: np.ndarray. A numpy array of shape `(3,)`,
        where `3` is (x,y,z).
    """
    try:
        import open3D as o3d
    except:
        raise ImportError("Call :func:`compute_protein_bbox_open3d` needs `pip install open3d`")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(prot_coords)
    # get protein point clouds aabb box as the results from :func:`compute_protein_bbox`
    aabb = pcd.get_axis_aligned_bounding_box()
    # aabb = np.asarray(aabb.get_box_points())# shape (8, 3) 8 3D points
    aabb = aabb.get_extent()
    # get protein point clouds obb box
    obb = pcd.get_oriented_bounding_box()
    # obb = np.asarray(obb.get_box_points())# shape (8, 3) 8 3D points
    obb = obb.get_extent()

    return aabb, obb

