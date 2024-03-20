# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Optional, Union, Tuple
import math
import numpy as np
from scipy.spatial.transform import Rotation

import torch
from torch import Tensor

from ..torch_utils import global_mean_pool, maybe_batch
from .utils import quaternion_to_rot


class SVD_rigid_transform_np:
    """
    Implement rigid transformation by SVD decomposition between two sets
        of point cloud in 3D space.
    Referenced: https://blog.csdn.net/u012836279/article/details/80351462
    """
    def __init__(self):
        self._clear()

    def _clear(self):
        self._ref_coords:Optional[np.ndarray] = None
        self._coords:Optional[np.ndarray] = None
        self._trans_coords:[np.ndarray] = None
        self._rmsd:Optional[float] = None
        self._init_rmsd:Optional[float] = None
        self._rot: Optional[np.ndarray] = None
        self._trans:Optional[np.ndarray] = None

    @property
    def rot(self):
        return self._rot

    @property
    def trans(self):
        return self._trans

    @property
    def rmsd(self):
        return self._rmsd

    @property
    def init_rmsd(self):
        return self._init_rmsd

    @property
    def transformed(self):
        """Return the transformed coordinates."""
        return self._trans_coords

    def align(
            self,
            ref_coords: Optional[np.ndarray] = None,
            coords: Optional[np.ndarray] = None,
    ):
        self._clear()
        if ref_coords is None or coords is None:
            raise AttributeError(f":cls:`{self.__class__.__name__}` :attr:`ref_coords` or `coords` "
                                 f" is None. Please set them by calling the :func:`align`.")
        ref_shape = ref_coords.shape
        shape = coords.shape
        if ref_shape != shape:
            raise ValueError("Input `ref_coords` and `coords` must be the same shape, "
                             f"but got `ref_coords` shape {ref_shape}, `coords` shape {shape}")
        if ref_shape[1] == 3 and shape[1] == 3:
            raise ValueError("The two dimension of input `ref_coords` and `coords` must be 3 for "
                             f"Euclidean space, but get `ref_coords` shape {ref_shape}, `coords` shape {shape}")
        self._ref_coords = ref_coords # shape (N, 3)
        self._coords = coords # shape (N, 3)
        self._init_rmsd = self.calc_rmsd(ref_coords, coords)
        # get centroids for ref and coords and center on the centroids
        # centroid shape (1, 3)
        centroid_ref = np.mean(ref_coords, axis = 0, keepdims = True)
        centroid_coords = np.mean(coords, axis = 0, keepdims = True)
        ref_coords = ref_coords - centroid_ref
        coords = coords - centroid_coords
        # get correlation matrix
        C = coords.T @ ref_coords
        # SVG decomposition
        U, S, V_T = np.linalg.svd(C)
        # find rotation matrix
        self._rot = (V_T.T @ U.T).T
        # reflection
        if np.linalg.det(self._rot) < 0:
            V_T[2] = -V_T[2]
            self._rot = (V_T.T @ U.T).T
        # translation matrix shape (1, 3)
        self._trans = centroid_ref - centroid_coords @ self.rot
        self._trans_coords = self._coords @ self.rot + self.trans
        self._rmsd = self.calc_rmsd(self._ref_coords, self._trans_coords)

    def calc_rmsd(
            self,
            ref_coords: Optional[np.ndarray] = None,
            coords: Optional[np.ndarray] = None,
    ) -> float:
        if ref_coords is None or coords is None:
            raise AttributeError(f":cls:`{self.__class__.__name__}` :attr:`ref_coords` or `coords` "
                                 f" is None. Please set them by calling the :func:`align`.")
        return np.sqrt(((coords - ref_coords) ** 2).sum(axis = (-1, -2)) / ref_coords.shape[0])

    @staticmethod
    def get_random_rottrans(
            max_distance: Union[int, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a random initial rotation matrix and translation matrix based on max distance
        Args:
            max_distance, int or float. translation matirx distance upper bound.
        Returns:
            rotation matrix: shape (3, 3).
            translation matrix: shape (1, 3)
        """
        rot = Rotation.random(num = 1)
        rot = rot.as_matrix().squeeze()

        trans = np.random.randn(1, 3)
        trans = trans / np.linalg.norm(trans)
        distance = np.random.uniform(low = 0, high = max_distance)
        trans *= distance

        return rot.astype(np.float32), trans.astype(np.float32)


    @staticmethod
    def get_transformation_matrix(*args):
        assert len(args) > 6, "transformation matrix maybe require at least 6 inputs."
        x, y, z, tx, ty, tz = args[:6]
        return np.array(
            [[np.cos(z) * np.cos(y), np.cos(z) * np.sin(y) * np.sin(x) - np.sin(z) * np.cos(x), np.cos(z) * np.sin(y) * np.cos(x) + np.sin(z) * np.sin(x), tx],
             [np.sin(z) * np.cos(y), np.sin(z) * np.sin(y) * np.sin(x) + np.cos(z) * np.cos(x), np.sin(z) * np.sin(y) * np.cos(x) - np.cos(z) * np.sin(x), ty],
             [-np.sin(y),            np.cos(y) * np.sin(x),                                     np.cos(y) * np.cos(x),                                     tz],
             [0,                     0,                                                         0,                                                          1]],
            dtype = np.double,
        )

class SVD_rigid_transform_torch(SVD_rigid_transform_np):
    """Pytorch version of SVD rigid transformation"""
    def __init__(self):
        self._clear()

    def _clear(self):
        self._ref_coords: Optional[Tensor] = None
        self._coords: Optional[Tensor] = None
        self._trans_coords: [Tensor] = None
        self._rmsd: Optional[float] = None
        self._init_rmsd: Optional[float] = None
        self._rot: Optional[Tensor] = None
        self._trans: Optional[Tensor] = None

    def align(
            self,
            ref_coords: Optional[Tensor] = None,
            coords: Optional[Tensor] = None,
    ):
        self._clear()
        if ref_coords is None or coords is None:
            raise AttributeError(f":cls:`{self.__class__.__name__}` :attr:`ref_coords` or `coords` "
                                 f" is None. Please set them by calling the :func:`align`.")
        ref_shape = ref_coords.shape
        shape = coords.shape
        if ref_shape != shape:
            raise ValueError("Input `ref_coords` and `coords` must be the same shape, "
                             f"but got `ref_coords` shape {ref_shape}, `coords` shape {shape}")
        if ref_shape[1] == 3 and shape[1] == 3:
            raise ValueError("The two dimension of input `ref_coords` and `coords` must be 3 for "
                             f"Euclidean space, but get `ref_coords` shape {ref_shape}, `coords` shape {shape}")
        self._ref_coords = ref_coords # shape (N, 3)
        self._coords = coords # shape (N, 3)
        self._init_rmsd = self.calc_rmsd(ref_coords, coords)
        # get centroids for ref and coords and center on the centroids
        # centroid shape (1, 3)
        centroid_ref = torch.mean(ref_coords, dim = 0, keepdim = True)
        centroid_coords = torch.mean(coords, dim = 0, keepdim = True)
        ref_coords = ref_coords - centroid_ref
        coords = coords - centroid_coords
        # get correlation matrix
        C = coords.T @ ref_coords
        # SVG decomposition
        U, S, V_T = torch.linalg.svd(C)
        # find rotation matrix
        self._rot = (V_T.T @ U.T).T
        # reflection
        if torch.linalg.det(self._rot) < 0:
            V_T[2] = -V_T[2]
            self._rot = (V_T.T @ U.T).T
        # translation matrix shape (1, 3)
        self._trans = centroid_ref - centroid_coords @ self.rot
        self._trans_coords = self._coords @ self.rot + self.trans
        self._rmsd = self.calc_rmsd(self._ref_coords, self._trans_coords)

    def calc_rmsd(
            self,
            ref_coords: Optional[Tensor] = None,
            coords: Optional[Tensor] = None,
    ) -> float:
        if ref_coords is None or coords is None:
            raise AttributeError(f":cls:`{self.__class__.__name__}` :attr:`ref_coords` or `coords` "
                                 f" is None. Please set them by calling the :func:`align`.")
        return torch.sqrt((coords - ref_coords).square().sum(dim = (-2, -1)) / ref_coords.shape[0])

    @staticmethod
    def get_random_rottrans(
            max_distance: Union[int, float],
    ) -> Tuple[Tensor, Tensor]:
        rot, trans = SVD_rigid_transform_np.get_random_rottrans(max_distance)
        return torch.from_numpy(rot), torch.from_numpy(trans)

    @staticmethod
    def get_transformation_matrix(*args) -> torch.DoubleTensor:
        transformation = SVD_rigid_transform_np.get_transformation_matrix(*args)
        return torch.from_numpy(transformation)

def superimposition_multibatch_torch(
        ref_coords: Tensor,
        coords: Tensor,
        mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Superimposition coordinates as the format of Tensor
        referenced reference coordinates by SVD rigid transformation.
    Args:
        ref_coords: Tensor, shape (..., N, 3)
        coords: Tensor, shape (..., N, 3)
        mask: Tensor, optional. shape (..., N)
            Default to None
    Returns:
        A tuple of [..., N, 3] superimposition coordinates and [...] RMSD.
    """
    ref_shape = ref_coords.shape
    shape = coords.shape
    if ref_shape != shape:
        raise ValueError("Input `ref_coords` and `coords` must be the same shape, "
                         f"but got `ref_coords` shape {ref_shape}, `coords` shape {shape}")
    if ref_shape[-1] == 3 and shape[-1] == 3:
        raise ValueError("The last dimension of input `ref_coords` and `coords` must be 3 for "
                         f"Euclidean space, but get `ref_coords` shape {ref_shape}, `coords` shape {shape}")
    if mask is None:
        mask = coords.new_ones(shape[:-1], dtype = torch.long)

    # initialization of :cls: `SVD_rigid_transform_torch`
    svd_align = SVD_rigid_transform_torch()

    def _select_real_coords(
            coords: Tensor,
            mask: Tensor,
    ) -> Tensor:
        return torch.masked_select(
            input = coords,
            mask = (mask > 0.)[:, None]
        ).reshape(-1, 3)

    # flatten tensor with the batch size axis
    batch_axis = shape[:-2]
    flatten_ref = ref_coords.reshape((-1, ) + ref_shape[-2:])
    flatten_coords = coords.reshape((-1, ) + shape[-2:])
    flatten_mask = mask.reshape((-1, ) + mask.shape[-1:])
    spp_list = []
    rmsd_list = []
    for r, c, m in zip(flatten_ref, flatten_coords, flatten_mask):
        _r = _select_real_coords(r, m)
        _c = _select_real_coords(c, m)
        svd_align.align(_r, _c)
        _spp = _c.new_tensor(svd_align.transformed)
        rmsd = _c.new_tensor(svd_align.rmsd)
        spp = _c.new_zeros(c.shape)
        spp[m > 0.] = _spp
        spp_list.append(spp)
        rmsd_list.append(rmsd)
    spp = torch.stack(spp_list, dim = 0).reshape(batch_axis + shape[-2:])
    rmsd = torch.stack(rmsd_list, dim = 0).reshape(batch_axis)

    return spp, rmsd

def local_alignment_torch(
        ref_coords: Tensor,
        coords: Tensor,
        local_mask: Tensor,
        lig_coords: Optional[Tensor] = None,
        lig_mask: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
):
    """
    Local alignment and global transformation with ligand coordinates transform,
        such as motif alignment, pocket alignment
    Args:
        ref_coords: Tensor, shape (..., N, 3)
        coords: Tensor, shape (..., N, 3)
        local_mask: Tensor, shape (..., N)
        lig_coords: Tensor, optional. shape (..., M, 3)
            ligand positions of `coords` while reference coords
            ligand are non-movable, not required.
        lig_mask: Tensor, optional. shape (..., M)
        mask: Tensor, optional. shape (..., N)
            Default to None
    """
    ref_shape = ref_coords.shape
    shape = coords.shape
    batch_axis = shape[:-2]
    if ref_shape != shape:
        raise ValueError("Input `ref_coords` and `coords` must be the same shape, "
                         f"but got `ref_coords` shape {ref_shape}, `coords` shape {shape}")
    if ref_shape[-1] == 3 and shape[-1] == 3:
        raise ValueError("The last dimension of input `ref_coords` and `coords` must be 3 for "
                         f"Euclidean space, but get `ref_coords` shape {ref_shape}, `coords` shape {shape}")
    if lig_coords is not None and batch_axis != lig_coords.shape[:-2]:
        raise ValueError(f"ligand positions {lig_coords.shape} and receptor {shape} shape mismatch at batch dimension")
    if mask is None:
        mask = coords.new_ones(shape[:-1], dtype = torch.long)
    assert local_mask.shape == mask.shape, \
        f'local_mask {local_mask.shape} and mask {mask.shape} shape mismatch.'

    # initialization of :cls: `SVD_rigid_transform_torch`
    svd_align = SVD_rigid_transform_torch()

    def _select_real_coords(
            coords: Tensor,
            mask: Tensor,
    ) -> Tensor:
        return torch.masked_select(
            input = coords,
            mask = (mask > 0.0)[:, None]
        ).reshape(-1, 3)

    # flatten tensor with the batch size axis
    flatten_ref = ref_coords.reshape((-1, ) + ref_shape[-2:])
    flatten_coords = coords.reshape((-1, ) + shape[-2:])
    flatten_mask = mask.reshape((-1, ) + mask.shape[-1:])
    flatten_local_mask = local_mask.reshape((-1,) + mask.shape[-1:])
    spp_list = []
    global_rmsd_list = []
    local_rmsd_list = []
    if lig_coords is not None:
        assert lig_mask is not None, 'ligand mask must be input while lig_coords input.'
        flatten_lig_coords = lig_coords.reshape(
            (-1, ) + lig_coords.shape[-2:])
        flatten_lig_mask = lig_mask.reshape((-1,) + lig_mask.shape[-1:])
        spp_lig_list = []
    for idx, (r, c, m, lm) in enumerate(
            zip(flatten_ref, flatten_coords,
                flatten_mask, flatten_local_mask)):
        # local alignment
        _r = _select_real_coords(r, lm)
        _c = _select_real_coords(c, lm)
        svd_align.align(_r, _c)
        local_rmsd = _c.new_tensor(svd_align.rmsd)
        local_rmsd_list.append(local_rmsd)
        # global transformation
        _r = _select_real_coords(r, m)
        _c = _select_real_coords(c, m)
        spp = _c.new_zeros(c.shape)
        spp[m > 0.] = _c @ svd_align.rot + svd_align.trans
        spp_list.append(spp)
        global_rmsd = _c.new_tensor(svd_align.calc_rmsd(_r, _c))
        global_rmsd_list.append(global_rmsd)
        if lig_coords is not None:
            l = flatten_lig_coords[idx]
            lig_m = flatten_lig_mask[idx]
            _l = _select_real_coords(l, lig_m)
            spp_lig = _l.new_zeros(l.shape)
            spp_lig[lig_m > 0.] = _l @ svd_align.rot + svd_align.trans
            spp_lig_list.append(spp_lig)
    spp = torch.stack(spp_list, dim = 0).reshape(batch_axis + shape[-2:])
    global_rmsd = torch.stack(global_rmsd_list, dim = 0).reshape(batch_axis)
    local_rmsd = torch.stack(local_rmsd_list, dim = 0).reshape(batch_axis)
    if lig_coords is not None:
        spp_lig = torch.stack(spp_lig_list, dim = 0).reshape(
            batch_axis + lig_coords.shape[-2:])
        return spp, global_rmsd, local_rmsd, spp_lig
    return spp, global_rmsd, local_rmsd


def rigid_transform_Kabsch_3D_torch(
        A: Tensor, # shape (3, N)
        B: Tensor, # ref shape (3, N)
):
    # R = 3x3 rotation matrix, t = 3x1 column vector
    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise: 3 x 1
    centroid_A = torch.mean(A, dim = 1, keepdim = True)
    centroid_B = torch.mean(B, dim = 1, keepdim = True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    U, S, Vt = torch.linalg.svd(H)

    # Compared with SVD_rigid_transform_torch‘s implementation R^T because (N,3_a,3_b) \matmul (N, 3_b) -> (N, 3_b) @ (N, 3_b, 3_a)
    # Here the coordinates (3, N) so we can use R, so (N, 3_a, 3_b) @ (N, 3_b)
    R = Vt.T @ U.T # R \equiv (N, 3_a, 3_b)
    if torch.linalg.det(R) < 0:
        SS = torch.diag(torch.tensor([1.,1.,-1.], device = A.device))
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(torch.linalg.det(R) - 1) < 3e-3

    t = -R @ centroid_A + centroid_B
    return R, t

def batch_alignment_transform(
        ref_coords: Tensor,
        coords: Tensor,
        batch: Optional[Tensor] = None,
        num_nodes: Optional[Tensor] = None,
        batch_size: Optional[int] = None
) -> Tuple[Tensor, ...]:
    """
    Align coordinates of A to reference coordinates B based eigenvector solution.
        Support batch wise.
    Args:
        ref_coords: shape (N, 3). Reference coordinates.
        coords: shape (N, 3). Transformed coordinates.
        batch: shape (N,) or None.
        num_nodes: shape (B,) or None. B means batch size.
        batch_size: int or None.
    Returns:
        rotation matrix of coords transformation (N, 3, 3).
        Translation vector of coords transformation (N, 3).
        minimal RMSD value of aligned coords to reference coordinates (B,).
    """
    batch, num_nodes, batch_size = maybe_batch(
        ref_coords, batch, num_nodes, batch_size, 0,
    )

    total_nodes = ref_coords.shape[0]
    ref_centroids = global_mean_pool(ref_coords, batch)
    coords_centroids = global_mean_pool(coords, batch)
    ref_coords = ref_coords - torch.repeat_interleave(ref_centroids, num_nodes, dim = 0)
    coords = coords - torch.repeat_interleave(coords_centroids, num_nodes, dim = 0)
    pos_sum = (ref_coords + coords).view(total_nodes, 1, 3)
    pos_sub = (ref_coords - coords).view(total_nodes, 3, 1)
    # (N, 1, 4)
    tmp0 = torch.cat(
        [pos_sub.new_zeros((1, 1, 1)).expand(total_nodes, -1, -1),
         -pos_sub.permute(0, 2, 1)],
        dim = -1
    )
    # (N, 3, 3)
    identity = torch.eye(3).to(pos_sum).unsqueeze(0).expand(total_nodes, -1, -1)
    pos_sum = pos_sum.expand(-1, 3, -1)
    tmp1 = torch.cross(identity, pos_sum, dim = -1)
    # (N, 3, 4)
    tmp1 = torch.cat([pos_sub, tmp1], dim = -1)
    # (N, 4, 4)
    tmp = torch.cat([tmp0, tmp1], dim = -2)
    tmpb = torch.bmm(tmp.permute(0, 2, 1), tmp).view(total_nodes, -1)
    tmpb = global_mean_pool(tmpb, batch).view(batch_size, 4, 4)
    w, v = torch.linalg.eigh(tmpb)
    # (B,)
    min_rmsd = w[:, 0]
    # (B, 4)
    min_q = v[:, :, 0]
    # (B, 3, 3)
    rotation = quaternion_to_rot(min_q)
    t = ref_centroids - torch.einsum("kj,kij->ki", coords_centroids, rotation)
    rotation = torch.repeat_interleave(rotation, num_nodes, dim = 0)
    t = torch.repeat_interleave(t, num_nodes, dim = 0)

    return rotation, t, min_rmsd

def batch_alignment(
        ref_coords: Tensor,
        coords: Tensor,
        batch: Optional[Tensor] = None,
        num_nodes: Optional[Tensor] = None,
        batch_size: Optional[int] = None
) -> Tuple[Tensor, ...]:
    with torch.no_grad():
        rotation, t, min_rmsd = batch_alignment_transform(
                ref_coords = ref_coords,
                coords = coords,
                batch = batch,
                num_nodes = num_nodes,
                batch_size = batch_size,
        )
    coords = torch.einsum("kj,kij->ki", coords, rotation) + t
    return coords, min_rmsd

