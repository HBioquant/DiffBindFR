# Copyright (c) MDLDrugLib. All rights reserved.
import torch
from torch import Tensor


def calc_lig_centroid(
    pred_pos: Tensor, # (N_pose, N_traj, N_node, 3)
    target_pos: Tensor, # (N_node, 3)
) -> Tensor:
    pred_pos_mean = torch.mean(pred_pos, dim = -2)
    target_pos_mean = torch.mean(target_pos, dim = -2)
    target_pos_mean = target_pos_mean.view(*((1, ) * (pred_pos.dim() - 2) + target_pos_mean.shape))
    dist = (pred_pos_mean - target_pos_mean).norm(dim = -1)
    return dist