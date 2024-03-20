# Copyright (c) MDLDrugLib. All rights reserved.
import warnings
from typing import Optional, List, Tuple
import itertools
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from functools import lru_cache

import torch
from torch import Tensor
import torch.nn.functional as F



def radian2sincos(
        radian: np.ndarray,
) -> np.ndarray:
    """radian: Shape (..., M)"""
    sin = np.sin(radian)
    cos = np.cos(radian)
    return np.stack([sin, cos], axis = -1)

def radian2sincos_torch(
        radian: Tensor,
) -> Tensor:
    """radian: Shape (..., M)"""
    sin = torch.sin(radian)
    cos = torch.cos(radian)
    return torch.stack([sin, cos], dim = -1)

def rot_vec_around_x_axis(
        x: np.ndarray,
        radian: np.ndarray,
) -> np.ndarray:
    """
    Args:
        x: Shape (N, 3) or (N, M, 3)
        radian: Shape (N)
    Returns:
        rotated vec with Shape (N, 3)
    """
    sin_cos = radian2sincos(radian)
    template_rot_mat = np.array(
        [
            [0., 0., 0., 0., 0., -1, 0., 1., 0.],  # sin
            [0., 0., 0., 0., 1., 0., 0., 0., 1.],  # cos
        ], dtype=sin_cos.dtype)
    rot_mat = np.matmul(sin_cos, template_rot_mat)
    rot_mat = rot_mat.reshape(-1, 3, 3)
    rot_mat[..., 0, 0] = 1

    if x.ndim == 2:
        pattern = 'ikl,il->ik'
    elif x.ndim == 3:
        pattern = 'ikl,iml->imk'
    else:
        raise ValueError(f'Expect input `x` shape (N, 3) or (N, M, 3), but got {x.shape}')

    return np.einsum(pattern, rot_mat, x)

def parse_xrot_angle(
        x: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    """
    Args:
        x: Shape (N, 3)
    Returns:
        Projected over xy-plane vector (N, 3) and
        angle (radian) between x_axis-x_vector and xy-plane
        with Shape (N, )
    """
    # 1. rot x (x, y, z) to xy-plane
    n = x.shape[0]
    x_axis = np.zeros((n, 3), dtype = np.float32)
    x_axis[:, 0] = 1.
    # get x projection on the x axis: positive or negative
    x_vec = np.sum(x * x_axis, axis=-1, keepdims = True) * x_axis

    # get y projection on the y axis: only positive
    yz_vec = x - x_vec
    yz_norm = np.linalg.norm(yz_vec, axis = -1)

    xy_plane_proj = np.zeros((n, 3), dtype = np.float32)
    xy_plane_proj[:, 0] = x_vec[:, 0]
    xy_plane_proj[:, 1] = yz_norm

    # 2. get dihedral angle between x_vec-axis and xy-plane
    angle_radian = np.arctan2(yz_vec[:, -1], yz_vec[:, -2])

    return xy_plane_proj, angle_radian

def make_rigid_transformation_4x4(
        ex: np.ndarray,
        ey: np.ndarray,
        translation: np.ndarray,
        eps: float = 1e-6,
) -> np.ndarray:
    """
    Create a rigid 4x4 transformation matrix from two batched axes and transl.
    Args:
        ex: Shape (N, 3)
        ey: Shape (N, 3)
        translation: Shape (N, 3)
    Returns:
        transformation matrix 4x4 (N, 4, 4)
    """
    # Normalize ex.
    ex_normalized = ex / (np.linalg.norm(ex, axis = -1, keepdims = True) + eps)

    # make ey perpendicular to ex
    ey_normalized = ey - np.sum(ey * ex_normalized, axis = -1, keepdims = True) * ex_normalized
    ey_normalized /= np.linalg.norm(ey_normalized, axis = -1, keepdims = True)

    # compute ez as cross product
    eznorm = np.cross(ex_normalized, ey_normalized)
    m = np.zeros((ex.shape[0], 4, 4), dtype = np.float32)
    m[:, :, :3] = np.stack(
        [ex_normalized, ey_normalized, eznorm, translation], axis = 1
    )
    m[:, 3, 3] = 1.
    return m.transpose(0, 2, 1).astype(np.float32)


def residue_frame(
        origin,
        x_axis,
        xy_plane,
        eps: float = 1e-20,
):
    """
    Use global frame coordinates to get R and T for local frame to global transformation
    use :func:`apply_euclidean` x_global = R \maltul x_local + T
    So, we can transform the any global point to
    """
    e0 = x_axis - origin
    e1 = xy_plane - origin
    denom = np.sqrt(np.sum(e0 ** 2, axis = -1, keepdims = True) + eps)
    e0 = e0 / denom
    dot = np.sum(e0 * e1, axis = -1, keepdims = True)
    e1 = e1 - e0 * dot
    denom = np.sqrt(np.sum(e1 ** 2, axis = -1, keepdims = True) + eps)
    e1 = e1 / denom
    e2 = np.cross(e0, e1)
    rots: np.ndarray = np.stack([e0, e1, e2], axis = -1)
    translation = origin

    return rots, translation

def residue_frame_torch(
        origin: torch.Tensor,
        x_axis: torch.Tensor,
        xy_plane: torch.Tensor,
        eps: float = 1e-20,
):
    """
    Use global frame coordinates to get R and T for local frame to global transformation
    use :func:`apply_euclidean` x_global = R \maltul x_local + T
    So, we can transform the any global point to local point by inverse euclidean
    """
    e0 = x_axis - origin
    e1 = xy_plane - origin
    denom = torch.sqrt(torch.sum(e0 ** 2, dim = -1, keepdim = True) + eps)
    e0 = e0 / denom
    dot = torch.sum(e0 * e1, dim = -1, keepdim = True)
    e1 = e1 - e0 * dot
    denom = torch.sqrt(torch.sum(e1 ** 2, dim = -1, keepdim = True) + eps)
    e1 = e1 / denom
    e2 = torch.cross(e0, e1)
    rots = torch.stack([e0, e1, e2], dim = -1)
    translation = origin

    return rots, translation

def apply_euclidean(
        x: np.ndarray,
        R: np.ndarray,
        T: np.ndarray,
) -> np.ndarray:
    """
    Apply the rotation operator and translation operator to x.
    Args:
        x: Shape (N, M, 3)
        R: Shape (N, 3, 3)
        T: Shape (N, 3)
    Returns:
        Transformed x with shape (N, M, 3)
    """
    return np.einsum('ikl,iml->imk', R, x) + T.reshape(-1, 1, 3)

def apply_inv_euclidean(
        x: np.ndarray,
        R: np.ndarray,
        T: np.ndarray,
) -> np.ndarray:
    """
    Apply the inverse euclidean operators to x
    Args:
        x: Shape (N, M, 3)
        R: Shape (N, 3, 3)
        T: Shape (N, 3)
    Returns:
        Transformed x with shape (N, M, 3)
    """
    return np.einsum('ilk,iml->imk', R, x - T.reshape(-1, 1, 3))

def calc_euclidean_distance_np(
        X: np.ndarray,
        Y: np.ndarray,
) -> np.ndarray:
    """
    Computes pairwise distances between two sets of molecules point clouds
    Takes an input (m, 3) and (n, 3) numpy arrays of 3D coords of
        two molecules respectively, and outputs an m x n numpy
        array of pairwise distances in Angstroms between the first and
        second molecule. entry (i,j) is dist between the i"th
        atom of first molecule and the j"th atom of second molecule.
    Args:
        X: np.ndarray. A numpy array of shape `(m, 3)`, where `m` is
            the number of atoms.
        Y: np.ndarray. A numpy array of shape `(n, 3)`, where `n` is
            the number of atoms.
    Returns:
        pairwise_distances: np.ndarray. A numpy array of shape `(m, n)`.
    """
    pairwise_distances = cdist(X, Y, metric='euclidean')
    return pairwise_distances

def calc_euclidean_distance_torch(
        X: torch.Tensor,
        Y: torch.Tensor,
) -> torch.Tensor:
    """
    Computes pairwise distances between two sets of molecules point clouds
    Takes an input (m, 3) and (n, 3) torch.Tensor of 3D coords of
        two molecules respectively, and outputs an m x n torch.Tensor
        of pairwise distances in Angstroms between the first and
        second molecule. entry (i,j) is dist between the i"th
        atom of first molecule and the j"th atom of second molecule.
    Args:
        X: torch.Tensor . A torch.Tensor  of shape `(m, 3)`, where `m` is
            the number of atoms.
        Y: torch.Tensor . A torch.Tensor  of shape `(n, 3)`, where `n` is
            the number of atoms.
    Returns:
        pairwise_distances: torch.Tensor . A torch.Tensor  of shape `(m, n)`.
    """
    pairwise_distances = torch.cdist(X, Y, p = 2)
    return pairwise_distances

def unit_vector_np(
        vector: np.ndarray,
) -> np.ndarray:
    """
    Generate the unit vector of input vector.
    Args:
        vector: np.ndarray shape (3, ) in 3D space.
    Return:
        unit vector: np.ndarray shape (3, ) in 3D space.
    """
    return vector / np.linalg.norm(vector)

def angle_between_np(
        vector_i: np.ndarray,
        vector_j: np.ndarray,
) -> float:
    """
    Calculate the angle of two vector i and j
    Args:
        vector_i: np.ndarray shape (3, ) in 3D space.
        vector_j: np.ndarray shape (3, ) in 3D space.
    Returns:
        The angle in radians between the two vectors.
    """
    unit_i = unit_vector_np(vector_i)
    unit_j = unit_vector_np(vector_j)
    angle = np.arccos(np.dot(unit_i, unit_j))
    # avoiding nan
    if np.isnan(angle):
        if np.allclose(vector_i, vector_j):
            return 0.0
        else:
            return np.pi
    return angle

def unit_vector_torch(
        vector: torch.Tensor,
) -> torch.Tensor:
    """
    Generate the unit vector of input vector.
    Args:
        vector: torch.Tensor shape (3, ) in 3D space.
    Return:
        unit vector: torch.Tensor shape (3, ) in 3D space.
    """
    return vector / torch.linalg.norm(vector)

def angle_between_torch(
        vector_i: torch.Tensor,
        vector_j: torch.Tensor,
) -> float:
    """
    Calculate the angle of two vector i and j
    Args:
        vector_i: np.ndarray shape (3, ) in 3D space.
        vector_j: np.ndarray shape (3, ) in 3D space.
    Returns:
        The angle in radians between the two vectors.
    """
    unit_i = unit_vector_torch(vector_i)
    unit_j = unit_vector_torch(vector_j)
    angle = torch.arccos(torch.dot(unit_i, unit_j))
    # avoiding nan
    if torch.isnan(angle):
        if torch.allclose(vector_i, vector_j):
            return 0.0
        else:
            return torch.pi
    return angle

def uniform_unit_s2(*size):
    """
    Generate a random unit vector on the sphere S^2.
    Citation: http://mathworld.wolfram.com/SpherePointPicking.html
    Pseudocode:
        a. Choose random theta \element [0, 2*pi]
        b. Choose random z \element [-1, 1]
        c. Compute output vector u: (x,y,z) = (sqrt(1-z^2)*cos(theta), sqrt(1-z^2)*sin(theta),z)
    Returns:
    u: np.ndarray: A numpy array of shape `(*size, 3)`. u is an unit vector
    """
    theta = np.random.uniform(low = 0.0, high = 2 * np.pi, size = size)
    z = np.random.uniform(low = -1., high = 1., size = size)
    u = np.stack(
        [np.sqrt(1 - z ** 2) * np.cos(theta),
         np.sqrt(1 - z ** 2) * np.sin(theta), z],
        axis = -1,
    )
    return u

def rots_matmul(
        rot_i: torch.Tensor,
        rot_j: torch.Tensor,
) -> torch.Tensor:
    """
    Performs matrix multiplication of two rotation matrix tensors.
    Args:
        rot_i: shape (..., 3, 3) left multiplicand
        rot_j: shape (..., 3, 3) right multiplicand
    Returns:
        The matmul rotation matrix, shape (..., 3, 3).
    """
    return torch.einsum("...nj,...jm->...nm", rot_i, rot_j)

def rot_vec_matmul(
        rot: torch.Tensor,
        vec: torch.Tensor,
) -> torch.Tensor:
    """
    Performs matrix multiplication of rotation matrix tensor and vector tensor.
    Args:
        rot: shape (..., 3, 3) left multiplicand
        vec: shape (..., 3) right multiplicand
    Returns:
        The rotated vector matrix, shape (..., 3).
    """
    return torch.einsum("...nj,...j->...n", rot, vec)

@lru_cache(maxsize = None)
def identity_rot_mats(
    batch_dims: Tuple[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    """
    Get [*(batch_dims), 3, 3] identity rotation matrix
    Args:
        batch_dims: Tuple[int]. The batch axis if required rotation matrix.
        ...
    Returns:
        shape (*(batch_dims), 3, 3) ratation matrix.
    """
    rots = torch.eye(
        3, dtype=dtype, device=device, requires_grad=requires_grad
    )

    return rots.repeat(batch_dims + (1, 1))

@lru_cache(maxsize = None)
def identity_trans(
    batch_dims: Tuple[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    """Get [*(batch_dims), 3] identity translation matrix"""
    trans = torch.zeros(
        (*batch_dims, 3),
        dtype = dtype,
        device = device,
        requires_grad = requires_grad
    )

    return trans

def rot_inv(
        rot: torch.Tensor,
) -> torch.Tensor:
    """
    Get rotation matrix invert.
    Args:
        rot: Tensor. Shape (..., 3, 3)
    Returns:
        rot_inv: Tensor. Shape (..., 3, 3)
    """
    return rot.transpose(-1, -2)

#################### Euler Angle doc
# rotate around the X-axis  called roll
# Rx = [[ 1       0            0      ]
#       [ 0  cos(\alpha) -sin(\alpha) ]
#       [ 0  sin(\alpha)  cos(\alpha) ]]

# rotate around the Y-axis  called pitch
# Ry = [[ cos(\beta)  0  sin(\beta) ]
#       [     0       1      0      ]
#       [-sin(\beta)  0  cos(\beta) ]]

# rotate around the Z-axis  called yaw
# Rz = [[ cos(\theta)   -sin(\theta)   0 ]
#       [ sin(\theta)    cos(\theta)   0 ]
#       [      0              0        1 ]]

# P' = R × P = (Rz × Ry × Rx) × P    that has three degrees of freedom \alpha  \beta \theta
# When any one axis is rotated by 90 degrees, the rotation of another two axis will have the same effect
# on the overall rotation. This is called a deadlock.

def _euler_axis(
        axis: str,
        angle: torch.Tensor,
) -> torch.Tensor:
    """
    Given the input axis and angle, return the corresponding
        rotation matrix around the axis.
    Args:
        axis: str. 'X', 'Y', 'Z' options.
        angle: Tensor. Any shape in radians.
    Returns:
        Rotation matrix. Shape (..., 3, 3)
    """
    assert isinstance(axis, str), f'input `axis` must be string, but got {type(axis)}'
    axis = axis.upper()

    _cos = torch.cos(angle)
    _sin = torch.sin(angle)
    _zero = torch.zeros_like(angle)
    _one = torch.ones_like(angle)
    if axis == 'X':
        mat = (_one, _zero, _zero, _zero, _cos, -_sin, _zero, _sin, _cos)
    elif axis == 'Y':
        mat = (_cos, _zero, _sin, _zero, _one, _zero, -_sin, _zero, _cos)
    elif axis == 'Z':
        mat = (_cos, -_sin, _zero, _sin, _cos, _zero, _zero, _zero, _one)
    else:
        raise ValueError(f"input `axis` must be chosen from 'X', 'Y', 'Z', but got {axis}")

    return torch.stack(mat, dim = -1).view(angle.shape + (3, 3))

def euler_to_rot(
        euler_angle: torch.Tensor,
        order: List[str] = ['Z', 'Y', 'X'],# left-multiplication, equiv with external rotation (x -> y -> z)
) -> torch.Tensor:
    """
    Handle any orders from {'X', 'Y', 'Z'} to rotation matrix,
        not only the external rotation (x -> y -> z) and
        internal rotation (z -> y -> x).
    Args:
        euler_angle: Tensor. Shape (..., 3) in radians.
        order: a list of string. order must follow the last dim of
            input `euler_angle`.
    Returns:
        rotation matrix: Tensor. Shape (..., 3, 3)
    """
    if len(order) != 3:
        raise ValueError("input args `order` must be 3 letters from {'X', 'Y', 'Z'}")
    if order[1] in (order[0], order[2]):
        raise ValueError(f'Middle one must be the only, but got {order}')
    if all([ord in ['X', 'Y', 'Z'] for ord in order]):
        raise ValueError("Invalid input args `order`, any the element of `order` "
                         "must be chosen from {'X', 'Y', 'Z'}, " + f'but got {order}')
    if euler_angle.dim == 0 or euler_angle.size(-1) != 3:
        raise ValueError(f"input args `euler_angle` shape (..., 3). Check your input tensor with shape {euler_angle.shape}")
    # multiply the axis-wise rotation matrix by ele0_mat × ele1_mat × ele2_mat
    mat = [
        _euler_axis(which_order, ang) for which_order, ang in zip(order, torch.unbind(euler_angle, dim = -1))
    ]
    return torch.matmul(torch.matmul(mat[0], mat[1]), mat[2])

def _angle_from_tan(
        axis: str,
        other_axis: str,
        data: torch.Tensor,
        horizontal: bool,
        tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.
    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.
    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """
    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])

def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")

def rot_to_euler_angles(
        rot: torch.Tensor,
        order: List[str],
) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.
    Args:
        rot: Rotation matrices as tensor of shape (..., 3, 3).
        order: ordered string of three uppercase letters from {'X', 'Y', 'Z'}.
    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    Modified from https:/github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L266
    """
    if len(order) != 3:
        raise ValueError("Convention must have 3 letters.")
    if order[1] in (order[0], order[2]):
        raise ValueError(f"Invalid convention {order}.")
    if all([ord in ['X', 'Y', 'Z'] for ord in order]):
        raise ValueError("Invalid input args `order`, any the element of `order` "
                         "must be chosen from {'X', 'Y', 'Z'}, " + f'but got {order}')
    if rot.size(-1) != 3 or rot.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {rot.shape}.")
    i0 = _index_from_letter(order[0])
    i2 = _index_from_letter(order[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            rot[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(rot[..., i0, i0])

    o = (
        _angle_from_tan(
            order[0], order[1], rot[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            order[2], order[1], rot[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)



#################### quaternion doc
# quaternion q = w + xi + yi + zi, where i^2 = -1 (complex) and w^{2} + x^{2} + y^{2} + z^{2} = 1
# qPq^{-1} = R_{q}P =
# [[  1 - 2y^{2} - 2z^{2}    2xy - 2wz       2xz + 2wy]
#     2xy + 2wz        1 - 2x^{2} - 2z^{2}   2yz + 2wx]
#     2xz - 2wy              2yz + 2wx  1 - 2x^{2} - 2y^{2} ]]
# The physical picture is rotating \theta degree around A = (m, n, k), where m^{2} + n^{2} + k^{2} = 1
# q = s + tA = cos(\theta / 2) + Asin(\theta / 2) = cos(\theta / 2) + (mi + ni + ki)sin(\theta / 2)
# So you can randomly sample the rotated axis A and \theta and calculate the (w, x, y, z)
# quaternion can convert to rotation matrix as below

def standardize_quaternion(
        x: torch.Tensor
) -> torch.Tensor:
    """
    Standardize the quaternion to a non-negative real part quaternion
    Args:
        quaternion: Tensor. shape (..., 4).
    Returns:
        quaternion: Tensor. shape (..., 4)
    """
    return torch.where(x[..., 0:1] < 0, -x, x)

# Define some quaternion operators
_quaternion_entry = ['w', 'x', 'y', 'z']
_entry_to_idx = {''.join(value): ind for ind, value in enumerate(itertools.product(_quaternion_entry, repeat = 2))}

def _to_mat(
        pair: List[Tuple[str, int]],
) -> np.ndarray:
    mat = np.zeros(shape = (4, 4))
    for key, value in pair:
        ind = _entry_to_idx[key]
        mat[ind // 4][ind % 4] = value

    return mat

_QUATERNION_MAT = np.zeros(shape = (4, 4, 3, 3))
_QUATERNION_MAT[..., 0, 0] = _to_mat([('ww', 1), ('xx', 1), ('yy', -1), ('zz', -1)])
_QUATERNION_MAT[..., 1, 1] = _to_mat([('ww', 1), ('xx', -1), ('yy', 1), ('zz', -1)])
_QUATERNION_MAT[..., 2, 2] = _to_mat([('ww', 1), ('xx', -1), ('yy', -1), ('zz', 1)])
_QUATERNION_MAT[..., 0, 1] = _to_mat([('xy', 2), ('wz', -2)])
_QUATERNION_MAT[..., 1, 0] = _to_mat([('xy', 2), ('wz', 2)])
_QUATERNION_MAT[..., 2, 0] = _to_mat([('xz', 2), ('wy', -2)])
_QUATERNION_MAT[..., 0, 2] = _to_mat([('xz', 2), ('wy', 2)])
_QUATERNION_MAT[..., 1, 2] = _to_mat([('yz', 2), ('wx', -2)])
_QUATERNION_MAT[..., 2, 1] = _to_mat([('yz', 2), ('wx', 2)])

_QUAT_MUL = np.zeros(shape = (4, 4, 4))
_QUAT_MUL[..., 0] = np.array(
    [[ 1, 0, 0, 0 ],
     [ 0,-1, 0, 0 ],
     [ 0, 0,-1, 0 ],
     [ 0, 0, 0,-1 ]]
)
_QUAT_MUL[..., 1] = np.array(
    [[ 0, 1, 0, 0 ],
     [ 1, 0, 0, 0 ],
     [ 0, 0, 0, 1 ],
     [ 0, 0,-1, 0 ]]
)
_QUAT_MUL[..., 2] = np.array(
    [[ 0, 0, 1, 0 ],
     [ 0, 0, 0,-1 ],
     [ 1, 0, 0, 0 ],
     [ 0, 1, 0, 0 ]]
)
_QUAT_MUL[..., 3] = np.array(
    [[ 0, 0, 0, 1 ],
     [ 0, 0, 1, 0 ],
     [ 0,-1, 0, 0 ],
     [ 1, 0, 0, 0 ]]
)
_QUAT_MUL_VEC = _QUAT_MUL[:, 1:, :]

_CACHED_QUATS = {
    "_QTR_MAT": _QUATERNION_MAT,
    "_QUAT_MUL": _QUAT_MUL,
    "_QUAT_MUL_VEC": _QUAT_MUL_VEC,
}

@lru_cache(maxsize = None)
def _get_quat(quat_key, dtype, device):
    return torch.tensor(_CACHED_QUATS[quat_key], dtype=dtype, device=device)

def normalised_quaternion(
        quaternion: torch.Tensor,
        eps: float = 1e-6,
) -> torch.Tensor:
    """
    Get a unit quaternion. A unit quaternion has a `norm` of 1.0.
    Args:
        quaternion: Tensor. Shape (..., 4) with the first element real.
        eps: float, optional. Numerical patience. Default to 1e-6.
    Returns:
        A new unit quaternion Tensor. Shape (..., 4) with the first element real.
    """
    square = (quaternion ** 2).sum(-1, keepdim = True)
    zero = (square - eps) < 0
    if zero.all():
        raise ZeroDivisionError("Quaternion ortho-normalization ZeroDivisionError.")
    return quaternion / torch.sqrt(square)

def quaternion_to_rot(
        quaternion: torch.Tensor,
        normalise: bool = True,
) -> torch.Tensor:
    """
    Convert a quaternion to rotation matrix
    Args:
        quaternion: Tensor. Shape (..., 4) with the first element real.
        normalise: bool. Whether to normalize quats. If normalize is not True,
            must be a unit quaternion.
    Returns:
        rotation matrix: Tensor. shape (..., 3, 3)
    """
    if normalise:
        # guarantee it is a unit quaternion / orthonormalization
        quaternion = normalised_quaternion(quaternion)

    # shape (..., 4) -> (..., 4, 4)
    quaternion = quaternion[..., None] * quaternion[..., None, :]

    # get coefficient matrix shape (4, 4, 3, 3)
    mat = _get_quat("_QTR_MAT", dtype=quaternion.dtype, device=quaternion.device)

    # to shape (...1*, 4, 4, 3, 3)
    mat = mat.view((1,) * len(quaternion.shape[:-2]) + mat.shape)

    # broadcast to (..., 4, 4, 3, 3)
    rot = quaternion[..., None, None] * mat

    # shape (..., 3, 3)
    return torch.sum(rot, dim = (-3, -4))

@lru_cache(maxsize = None)
def identity_quats(
    batch_dims: Tuple[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    """Generate a (..., 4) identity quaternion with the real part = 1"""
    quat = torch.zeros(
        (*batch_dims, 4),
        dtype = dtype,
        device = device,
        requires_grad = requires_grad
    )

    with torch.no_grad():
        quat[..., 0] = 1

    return quat

def _sqrt_positive_part(
        x: torch.Tensor,
) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def rot_to_quaternion(
        rot: torch.Tensor
) -> torch.Tensor:
    """
    Convert a rotation matrix to quaternion
    Args:
        rot: Tensor. shape (..., 3, 3)
    Returns:
        quaternion: Tensor. shape (..., 4) with the first element real.
    This code uses a modification of the algorithm described in:
    https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    which is itself based on the method described here:
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    """
    if rot.shape[-2:] != (3, 3):
        raise ValueError(f'Input rotation matrix is expected to shape with (..., 3, 3), but got {rot.shape}')

    batch_dim = rot.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        rot.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
           F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
           ].reshape(batch_dim + (4,))

def rot_to_quaternion_openfold(
        rot: torch.Tensor
) -> torch.Tensor:
    """Code from https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/rigid_utils.py#L191"""
    if (rot.shape[-2:] != (3, 3)):
        raise ValueError("Input rotation is incorrectly shaped")

    rot = [[rot[..., i, j] for j in range(3)] for i in range(3)]
    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot

    k = [
        [xx + yy + zz, zy - yz, xz - zx, yx - xy, ],
        [zy - yz, xx - yy - zz, xy + yx, xz + zx, ],
        [xz - zx, xy + yx, yy - xx - zz, yz + zy, ],
        [yx - xy, xz + zx, yz + zy, zz - xx - yy, ]
    ]

    k = (1. / 3.) * torch.stack([torch.stack(t, dim=-1) for t in k], dim=-2)

    _, vectors = torch.linalg.eigh(k)
    return vectors[..., -1]

def quaternions_multiply(
        q: torch.Tensor,
        p: torch.Tensor
) -> torch.Tensor:
    """
    Multiply two quaternions by broadcasting apply.
    Args:
        q: Tensor. Shape (..., 4), real part first.
        p: Tensor. Shape (..., 4), real part first.
    Returns:
        The product of q and p, a tensor of quaternions shape (..., 4).
    """
    assert (q.shape[-1] == 4) and (p.shape[-1] == 4)
    # get quaternion multiply operator
    op = _get_quat("_QUAT_MUL", dtype=q.dtype, device=q.device)
    # view shape -> (..., 4, 4, 4)
    op = op.view((1, ) * len(q.shape[:-1]) + op.shape)
    return torch.sum(
        op * q[..., :, None, None] * p[..., None, :, None],
        dim = (-3, -2),
    )

def quaternion_vec_multiply(
        q: torch.Tensor,
        p: torch.Tensor,
) -> torch.Tensor:
    """
    Multiply one quaternions and one 3D vector by broadcasting apply.
    Args:
        q: Tensor. Shape (..., 4), real part first.
        p: Tensor. Shape (..., 3), 3D vector, as like quaternions = 0 + xi + yj + zk.
    Returns:
        The product of q and p, a tensor of quaternions shape (..., 4).
    """
    assert (q.shape[-1] == 4) and (p.shape[-1] == 3)
    # get quaternion multiply operator
    op = _get_quat("_QUAT_MUL_VEC", dtype=q.dtype, device=q.device)
    # view shape -> (..., 4, 3, 4)
    op = op.view((1,) * len(q.shape[:-1]) + op.shape)
    return torch.sum(
        op * q[..., :, None, None] * p[..., None, :, None],
        dim = (-3, -2),
    )

def quaternions_multiply_rot(
        q: torch.Tensor,
        p: torch.Tensor
) -> torch.Tensor:
    """
    Multiply two quaternions by broadcasting apply for representing rotation.
    Args:
        q: Tensor. Shape (..., 4), real part first.
        p: Tensor. Shape (..., 4), real part first.
    Returns:
        The product of q and p, a tensor of quaternions shape (..., 4) with
            non-negative real part.
    """
    qp = quaternions_multiply(q, p)
    return standardize_quaternion(qp)

def quaternion_invert(
        quat: torch.Tensor,
) -> torch.Tensor:
    """
    Get quaternion invert q^{-1} or q_hat.
    Args:
        quat: Tensor. Shape (..., 4)
    Returns:
        quat_inv: Tensor. Shape (..., 4)
    """
    q_hat = quat.clone()
    q_hat[..., 1:] *= -1
    # orthonormalization
    inv = q_hat / (q_hat ** 2).sum(-1, keepdim = True)
    return inv

def random_quaternions(
        n: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate a batch of random unit quaternions.
    Uniform distributed across the rotation space,
        with non-negative real part.
    Args:
        n: int. The number of quaternion for a batch.
        dtype: torch.dtype, optional. Default to None.
        device: torch.device, optional. Default to None.
    Returns:
        A batch of quaternions with non-negative real part.
            Shape (n, 4)
    Reference code: https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L261
    Reference source: http://planning.cs.uiuc.edu/node198.html
    """
    r1, r2, r3 = torch.unbind(torch.rand((n, 3), dtype = dtype, device = device), dim = -1)
    w = torch.sqrt(1.0 - r1) * torch.sin(2.0 * torch.pi * r2)
    x = torch.sqrt(1.0 - r1) * torch.cos(2.0 * torch.pi * r2)
    y = torch.sqrt(r1) * torch.sin(2.0 * torch.pi * r3)
    z = torch.sqrt(r1) * torch.cos(2.0 * torch.pi * r3)
    # this quaternion is orthogonal
    quat = torch.stack([w, x, y, z], dim = -1)
    # non-negative real part
    return standardize_quaternion(quat)

def random_quaternion(
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate a random unit quaternion.
    Uniform distributed across the rotation space,
        with non-negative real part.
    Args:
        dtype: torch.dtype, optional. Default to None.
        device: torch.device, optional. Default to None.
    Returns:
        A batch of quaternions with non-negative real part.
            Shape (4,)
    """
    return random_quaternions(1, dtype, device)[0]

def random_rotations(
        n: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate a batch of random rotation matrix from the random quaternions.
    Uniform distributed across the rotation space.
    Args:
        n: int. The number of rotation matrix for a batch.
        dtype: torch.dtype, optional. Default to None.
        device: torch.device, optional. Default to None.
    Returns:
        A batch of rotation matrix. Shape (n, 3, 3)
    """
    quat = random_quaternions(n, dtype = dtype, device = device)
    return quaternion_to_rot(quat)

def random_rotation(
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate a random rotation matrix from the random quaternion.
    Args:
        dtype: torch.dtype, optional. Default to None.
        device: torch.device, optional. Default to None.
    Returns:
        A batch of rotation matrix. Shape (3, 3)
    """
    return random_rotations(1, dtype, device)[0]

def apply_quaternion(
        quat: torch.Tensor,
        point_clouds: torch.Tensor,
) -> torch.Tensor:
    """
    Apply the rotation operator by a quaternion to 3D object.
    Args:
        quat: Tensor. Shape (..., 4), representing the rotation.
        point_clouds: Tensor. Shape (..., 3), 3D object.
    Returns:
        Rotated 3D object: Tensor. Shape (..., 3).
    """
    if point_clouds.size(-1) != 3:
        raise ValueError("input args `point_clouds` must be 3D object with the last dimension 3, "
                         f"but got shape {point_clouds.shape}")
    if quat.size(-1) != 4:
        raise ValueError("input args `point_clouds` must be normal quaternion with the last dimension 4, "
                         f"but got shape {quat.shape}")
    transformed = quaternions_multiply(
        quaternion_vec_multiply(quat, point_clouds),
        quaternion_invert(quat)
    )
    return transformed

def apply_rotmat(
        rot: torch.Tensor,
        point_clouds: torch.Tensor,
) -> torch.Tensor:
    """
    Apply the rotation operator by a rotation matrix to 3D object.
    Args:
        rot: Tensor. Shape (..., 3, 3), representing the rotation.
        point_clouds: Tensor. Shape (..., 3), 3D object.
    Returns:
        Rotated 3D object: Tensor. Shape (..., 3).
    """
    if point_clouds.size(-1) != 3:
        raise ValueError("input args `point_clouds` must be 3D object with the last dimension 3, "
                         f"but got shape {point_clouds.shape}")
    if rot.shape[-2:] != (3, 3):
        raise ValueError("input args `point_clouds` must be normal rotation matrix with the "
                         f"last two dimension (3, 3), but got shape {rot.shape}")
    transformed = rot_vec_matmul(rot, point_clouds)
    return transformed

def apply_euler(
        euler_angle: torch.Tensor,
        order: List[str],
        point_clouds: torch.Tensor,
) -> torch.Tensor:
    """
    Apply the specified euler angle to 3D object.
    Args:
        euler_angle: Tensor. Shape (..., 3) in radians.
        order: a list of string. order must follow the last dim of
            input `euler_angle`.
        point_clouds: Tensor. Shape (..., 3), 3D object.
    Returns:
        Rotated 3D object: Tensor. Shape (..., 3).
    """
    if point_clouds.size(-1) != 3:
        raise ValueError("input args `point_clouds` must be 3D object with the last dimension 3, "
                         f"but got shape {point_clouds.shape}")
    if euler_angle.size(-1) != 3:
        raise ValueError("input args `euler_angle` must be normal euler matrix with the "
                         f"last dimension 3, but got shape {euler_angle.shape}")
    rot = euler_to_rot(euler_angle, order)
    return apply_rotmat(rot, point_clouds)

def axis_angle_to_quaternion(
        axis_angle: torch.Tensor
) -> torch.Tensor:
    """
    Convert axis angle to quaternion
    Note that simple axis angle is (ax, ay, az)
        with anticlockwise angle \theta in radians
        around the unit vector direction.
    But the other representation is (ax * \theta,
        ay * \theta, az * \theta).
    Args:
        axis_angle: Tensor. Shape (..., 3).
    Returns:
        quaternion: Tensor. Shape (..., 4).
    """
    # calculate 2-norm to get rotation angle
    ang: torch.Tensor = torch.norm(axis_angle, p = 2, dim = -1, keepdim = True)
    half_ang: torch.Tensor = ang * 0.5
    sin_term_divide_angle = torch.empty_like(ang)
    eps = 1e-6
    # solve the numerical stability problem
    small_ang_flag = ang.abs() < eps
    # q = cos(\theta / 2) + Asin(\theta / 2), where A = T/||T||
    # stable case
    stable = ~small_ang_flag
    sin_term_divide_angle[stable] = (torch.sin(half_ang[stable]) / ang[stable])
    # unstable case, dealing with the sin(\theta / 2) taylor expansion
    # sin(x / 2) = x / 2 - \frac{1}{6}(x / 2)^3
    # sin(x / 2) / x = 1 / 2 - x^2 / 48
    sin_term_divide_angle[small_ang_flag] = (
        0.5 - (ang[small_ang_flag] * ang[small_ang_flag]) / 48
    )

    return torch.cat(
        [torch.cos(half_ang), axis_angle * sin_term_divide_angle],
        dim = -1,
    )

def quaternion_to_axis_angle(
        quat: torch.Tensor,
) -> torch.Tensor:
    """
    Convert quaternion to axis angle
    Note that simple axis angle is (ax, ay, az)
        with anticlockwise angle \theta in radians
        around the unit vector direction.
    But the other representation is (ax * \theta,
        ay * \theta, az * \theta).
    Args:
        quaternion: Tensor. Shape (..., 4).
    Returns:
        axis_angle: Tensor. Shape (..., 3).
    """
    quat = normalised_quaternion(quat)
    sin = torch.norm(quat[..., 1:], p = 2, dim = -1, keepdim = True)
    half_ang: torch.Tensor = torch.atan2(sin, quat[..., :1])
    ang: torch.Tensor = half_ang * 2.
    sin_term_divide_angle = torch.empty_like(ang)
    eps = 1e-6
    # solve the numerical stability problem
    small_ang_flag = ang.abs() < eps
    # q = cos(\theta / 2) + Asin(\theta / 2), where A = T/||T||
    # stable case
    stable = ~small_ang_flag
    sin_term_divide_angle[stable] = (torch.sin(half_ang[stable]) / ang[stable])
    # unstable case, dealing with the sin(\theta / 2) taylor expansion
    # sin(x / 2) = x / 2 - \frac{1}{6}(x / 2)^3
    # sin(x / 2) / x = 1 / 2 - x^2 / 48
    sin_term_divide_angle[small_ang_flag] = (
            0.5 - (ang[small_ang_flag] * ang[small_ang_flag]) / 48
    )

    return quat[..., 1:] / sin_term_divide_angle

def rigid_to_se3_vec(
        frame: np.ndarray,
        scaling_factor: float = 1.0,
) -> np.ndarray:
    """
    Convert x7 (quaternion, translation) to
        x6 (axis angle, translation) formatting
        np.ndarry.
    Args:
        frame: np.ndarray with shape (N, 7).
    Returns:
        frame: np.ndarray with shape (N, 6)
    """
    translation = frame[:, 4:] * scaling_factor
    axis_angle = Rotation.from_quat(frame[:, :4]).as_rotvec()
    return np.concatenate([axis_angle, translation], axis = -1)

def _copysign(
        input: torch.Tensor,
        other: torch.Tensor
) -> torch.Tensor:
    """
    Return tensor where each element has the absolute value taken from
        the corresponding element of `input` with sign taken from the
        corresponding element of `other`. This is like the standard copysign
        floating-point operation, but is not careful about negative 0 and NaN.
    Args:
        input: source tensor.
        other: the same shape with input tensor, sign provider.
    Returns:
        Tensor `input` with updated sign, the same as other sign.
    """
    sign_differ = (input < 0) != (other < 0)
    return torch.where(sign_differ, -input, input)

def quaternion_to_euler(
        quat: torch.Tensor,
) -> torch.Tensor:
    """
    Convert quaternion to euler angle.
    Args:
        quaternion: Tensor. Shape (..., 4).
    Returns:
        euler angle: Tensor. Shape (..., 3).
    """
    quat = normalised_quaternion(quat)
    w, x, y, z = torch.unbind(quat, dim = -1)
    # x rotation
    roll = torch.atan2(2.*(w*x + y*z), 1.- 2.*(x**2 + y**2))
    # y rotation
    sinp = 2.*(w*y - z*x)
    mask = sinp.abs() >= 1
    pitch = torch.empty_like(sinp)
    pitch[~mask] = torch.asin(sinp[~mask])
    M_PI: torch.Tensor = torch.ones_like(sinp[mask]) * (torch.pi / 2)
    pitch[mask] = _copysign(M_PI, sinp[mask])
    # z rotation
    yaw = torch.atan2(2.*(w*z + x*y), 1.-2.*(z**2 + y**2))

    return torch.stack([roll, pitch, yaw], dim = -1)


def euler_to_quaternion(
        euler: torch.Tensor,
) -> torch.Tensor:
    """
    Convert euler angle to quaternion.
    Args:
        euler angle: Tensor. Shape (..., 3).
    Returns:
        quaternion: Tensor. Shape (..., 4).
    """
    r, p, y = torch.unbind(euler, dim = -1)
    cr = torch.cos(r * 0.5)
    sr = torch.sin(r * 0.5)
    cp = torch.cos(p * 0.5)
    sp = torch.sin(p * 0.5)
    cy = torch.cos(y * 0.5)
    sy = torch.sin(y * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr + sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    return torch.stack([w, x, y, z], dim = -1)

def rot_to_axis_angle(
        rot: torch.Tensor,
) -> torch.Tensor:
    """
    Convert rotation matrix to axis angle
    Args:
        rot: Tensor. Shape (..., 3, 3).
    Returns:
        axis angle: Tensor. Shape (..., 3).
    """
    return quaternion_to_axis_angle(rot_to_quaternion(rot))

def axis_angle_to_rot(
        axis_angle: torch.Tensor,
) -> torch.Tensor:
    """
    Convert axis angle to rotation matrix.
    Args:
        axis_angle: Tensor. Shape (..., 3).
    Returns:
        rotation matrix: Tensor. Shape (..., 3, 3).
    """
    return quaternion_to_rot(axis_angle_to_quaternion(axis_angle))

def _fast_3x3_det(
        mat: torch.Tensor
) -> torch.Tensor:
    """
    Fast calculate a batch of 3x3 matrix, such as rotation matrix,
        determinant
    Args:
        mat: Tensor. Shape (..., 3, 3).
    Returns:
        matrix determinnant. Shape (...)
    """
    det = (
        mat[..., 0, 0] * (mat[..., 1, 1] * mat[..., 2, 2] - mat[..., 1, 2] * mat[..., 2, 1]) +
        mat[..., 0, 2] * (mat[..., 1, 0] * mat[..., 2, 1] - mat[..., 1, 1] * mat[..., 2, 0]) +
        mat[..., 0, 1] * (mat[..., 1, 2] * mat[..., 2, 0] - mat[..., 1, 0] * mat[..., 2, 2])
    )
    return det

@torch.no_grad()
def check_rotation_matrix(
        rot: torch.Tensor,
        atol: float = 1e-6,
):
    """
    Check the rotation matrix whether meets the followed property:
     ... RR^T = I and det(R) = 1
    Args:
        rot: rotation matrix. Shape (N, 3, 3)
        atol: float, optional. allclose tolerance. Default to 1e-6.
    """
    identity = torch.eye(3, dtype = rot.dtype, device = rot.device)
    identity = identity.view(1, 3, 3).expend(rot.shape[0], -1, -1)
    # check whether orthogonal
    orthogonal_flag = torch.allclose(rot.bmm(rot.transpose(2, 1)), identity, atol = atol)
    det = _fast_3x3_det(rot)
    det_flag = torch.allclose(det, torch.ones(det), atol = atol)
    if not (det_flag and orthogonal_flag):
        warnings.warn(f'The rotation matrix shape {rot.shape} is not a valid orthogonal matrix.')

    return



