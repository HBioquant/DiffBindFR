# Copyright (c) MDLDrugLib. All rights reserved.
import copy
from typing import (
    Union, Optional, List,
)
from easydict import EasyDict as ed
import numpy as np

import torch
from torch import Tensor, BoolTensor

from druglib.utils.bio_utils import update_batchlig_pos
from druglib.utils.geometry_utils import radian2sincos_torch
from druglib.utils.obj import (
    build_pdb_from_template,
)
from druglib.utils.torch_utils import (
    ptr_and_batch, slice_tensor_batch
)
from .base import BaseMLDocker
from .default_MLDockBuilder import MLDOCK_BUILDER
from ..builder import build_interaction, build_energy



@MLDOCK_BUILDER.register_module()
class DiffBindFR(BaseMLDocker):
    def __init__(
            self,
            diffusion_model: Optional[dict] = None,
            scoring_model: Optional[dict] = None,
            train_cfg: dict = {},
            test_cfg: dict = {},
            pretrained = None,
            init_cfg: dict = {},
            **kwargs,
    ):
        super(DiffBindFR, self).__init__(init_cfg = init_cfg)
        if diffusion_model is not None:
            print('Initializing diffusion model...')
            self.diffusion_model_cfg = copy.deepcopy(diffusion_model.cfg)
            self.diffusion_model = build_interaction(diffusion_model)
        if scoring_model is not None:
            print('Initializing scoring model...')
            self.scoring_model_cfg = copy.deepcopy(scoring_model.cfg)
            self.scoring_model = build_energy(scoring_model)

        # phase (train or test) config
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # Weights initialization
        self.pretrain = pretrained
        self.init_cfg = init_cfg

        # built-in debug parameters
        self.debug = False

    def forward_train(
            self,
            **kwargs,
    ):
        if hasattr(self, 'diffusion_model') and hasattr(self, 'scoring_model'):
            raise RuntimeError('When training mode, cannot train diffusion model and scoring model at the same time.')

    def forward_test(
            self,
            data,
            *args,
            **kwargs
    ):
        data = data.to_dict(
            decode = True,
            drop_meta = False,
        )

        return self.sample(
            ed(**data),
            *args,
            **kwargs
        )

    def t_schedule(self):
        cfg = self.test_cfg.sample_cfg
        time_schedule = cfg.time_schedule
        inference_steps = cfg.inference_steps
        eps = cfg.eps
        if time_schedule == 'linear':
            return torch.linspace(1, eps, inference_steps + 1)
        else:
            raise NotImplementedError('Current time schedule only supports `linear`.')

    def sigma_fn(self, t_tr, t_rot, t_tor, t_sc_tor):
        """Synchronous sigma function given time t"""
        cfg = self.test_cfg.sample_cfg
        tr_sigma = cfg.tr_sigma_min ** (1 - t_tr) * cfg.tr_sigma_max ** t_tr
        rot_sigma = cfg.rot_sigma_min ** (1 - t_rot) * cfg.rot_sigma_max ** t_rot
        tor_sigma = cfg.tor_sigma_min ** (1 - t_tor) * cfg.tor_sigma_max ** t_tor
        sc_tor_sigma = None
        if not self.diffusion_model_cfg.no_sc_torsion:
            sc_tor_sigma = cfg.sc_tor_sigma_min ** (1 - t_sc_tor) * cfg.sc_tor_sigma_max ** t_sc_tor
        return tr_sigma, rot_sigma, tor_sigma, sc_tor_sigma

    def set_time(self, data, t: float):
        from druglib.utils.geometry_utils import so3, torus
        device = data.batch.device
        num_graphs = data.lig_node_batch.max().item() + 1
        data.t = torch.tensor(
            [t] * num_graphs,
            device = device,
            dtype = torch.float32)
        tr_sigma, rot_sigma, tor_sigma, sc_tor_sigma = self.sigma_fn(t, t, t, t)
        data.tr_sigma = torch.tensor([tr_sigma] * num_graphs, device = device, dtype = torch.float32)
        data.rot_score_norm = so3.score_norm(np.array([rot_sigma])).to(device).repeat(num_graphs, 1)
        num_torsion_bonds = data.tor_edge_mask.sum()
        tor_sigma_edge = torch.ones(num_torsion_bonds, dtype = torch.float32) * sc_tor_sigma
        data.tor_score_norm2 = torch.from_numpy(torus.score_norm(tor_sigma_edge)).to(device).float()
        sc_torsion_edge_mask = data.sc_torsion_edge_mask
        chi4_sigma = torch.ones(sc_torsion_edge_mask.shape, dtype = torch.float32) * sc_tor_sigma
        data.sc_tor_score_norm2 = torch.from_numpy(torus.score_norm(chi4_sigma)).to(device).float() * sc_torsion_edge_mask

        return data, tr_sigma, rot_sigma, tor_sigma, sc_tor_sigma

    @torch.no_grad()
    def sample(
            self,
            data,
            visualize: bool = False,
    ):
        """Output predicted ligand pose and protein positions"""
        cfg = self.test_cfg.sample_cfg
        device = data.batch.device
        num_graphs = data.lig_node_batch.max().item() + 1
        actual_steps = cfg.actual_steps
        assert actual_steps <= cfg.inference_steps, 'actual steps should <= inference steps'
        t_schedule = self.t_schedule()
        rot_node_mask_list = data.metastore['rot_node_mask']
        if isinstance(rot_node_mask_list[0], np.ndarray):
            fn = lambda x: torch.from_numpy(x).to(device)
            data.metastore['rot_node_mask'] = [fn(m) for m in rot_node_mask_list]

        lig_pos_out_list = []
        if not self.diffusion_model_cfg.no_sc_torsion:
            atom14_pos_out_list = []

        for t_idx in range(actual_steps):
            t = t_schedule[t_idx]
            dt = t_schedule[t_idx] - t_schedule[t_idx + 1]

            _data, tr_sigma, rot_sigma, tor_sigma, sc_tor_sigma = self.set_time(
                copy.deepcopy(data), t)
            tr_score, rot_score, tor_score, sc_tor_score = self.diffusion_model(_data)

            tr_g = tr_sigma * np.sqrt(
                2 * np.log(cfg.tr_sigma_max / cfg.tr_sigma_min)
            )
            rot_g = 2 * rot_sigma * np.sqrt(
                np.log(cfg.rot_sigma_max / cfg.rot_sigma_min)
            )
            tor_g = tor_sigma * np.sqrt(
                2 * np.log(cfg.tor_sigma_max / cfg.tor_sigma_min))
            if cfg.type == 'ode':
                tr_perturb = 0.5 * tr_g ** 2 * tr_score * dt
                rot_perturb = 0.5 * rot_g ** 2 * rot_score * dt
                tor_perturb = 0.5 * tor_g ** 2 * tor_score * dt
            else:
                tr_z = torch.zeros((num_graphs, 3)) if cfg.no_random or (
                            cfg.no_final_step_noise and t_idx == actual_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(num_graphs, 3))
                tr_z = tr_z.to(device=device, dtype=torch.float32)
                tr_perturb = tr_g ** 2 * tr_score * dt + tr_g * np.sqrt(dt) * tr_z

                rot_z = torch.zeros((num_graphs, 3)) if cfg.no_random or (
                            cfg.no_final_step_noise and t_idx == actual_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(num_graphs, 3))
                rot_z = rot_z.to(device=device, dtype=torch.float32)
                rot_perturb = rot_g ** 2 * rot_score * dt + rot_g * np.sqrt(dt) * rot_z

                tor_z = torch.zeros(tor_score.shape) if cfg.no_random or (
                            cfg.no_final_step_noise and t_idx == actual_steps - 1) \
                    else torch.normal(mean=0, std=1, size=tor_score.shape)
                tor_z = tor_z.to(device=device, dtype=torch.float32)
                tor_perturb = tor_g ** 2 * tor_score * dt + tor_g * np.sqrt(dt) * tor_z

            # update ligand pose
            _update_lig_pos = update_batchlig_pos(
                tr_perturb, rot_perturb, tor_perturb,
                data.lig_pos, data.lig_edge_index,
                data.tor_edge_mask, data.metastore['rot_node_mask'],
                batch = data.lig_node_batch,
            )
            data.lig_pos = _update_lig_pos
            if visualize or t_idx == actual_steps - 1:
                lig_pos_out_list.append(_update_lig_pos.detach().cpu())

            if not self.diffusion_model_cfg.no_sc_torsion:
                sc_tor_g = sc_tor_sigma * np.sqrt(
                    2 * np.log(cfg.sc_tor_sigma_max / cfg.sc_tor_sigma_min))
                if cfg.type == 'ode':
                    sc_tor_perturb = 0.5 * sc_tor_g ** 2 * sc_tor_score * dt
                else:
                    sc_tor_z = torch.zeros(sc_tor_score.shape) if cfg.no_random or (cfg.no_final_step_noise and t_idx == actual_steps - 1) \
                        else torch.normal(mean = 0, std = 1, size = sc_tor_score.shape)
                    sc_tor_z = sc_tor_z.to(device = device, dtype = torch.float32)
                    sc_tor_perturb = sc_tor_g ** 2 * sc_tor_score * dt + sc_tor_g * np.sqrt(dt) * sc_tor_z

                # update pocket side chain positions
                chi_angles = data.torsion_angle[:, 1:]
                chi_angles[data.sc_torsion_edge_mask] = chi_angles[data.sc_torsion_edge_mask] + sc_tor_perturb
                data.torsion_angle[:, 1:] = chi_angles

                template = ed(
                    sequence = data.sequence,
                    backbone_transl = data.backbone_transl,
                    backbone_rots = data.backbone_rots,
                    default_frame = data.default_frame,
                    rigid_group_positions = data.rigid_group_positions,
                    torsion_angle = radian2sincos_torch(data.torsion_angle),
                )
                atom14_mask = data.atom14_mask.bool()
                _update_atom14_pos, _ = build_pdb_from_template(
                    template = template,
                    device = device,
                )
                _update_atom14_pos = _update_atom14_pos * atom14_mask.unsqueeze(-1)
                data.rec_atm_pos = _update_atom14_pos[atom14_mask]
                if not self.diffusion_model_cfg.no_sc_torsion and (visualize or t_idx == actual_steps - 1):
                    atom14_pos_out_list.append(
                        _update_atom14_pos.detach().cpu(),
                    )

            del _data

        lig_pos_traj = torch.stack(lig_pos_out_list, dim = 1)
        lig_pos_traj = [_pos.transpose(1, 0) for _pos in slice_lig(
            lig_pos = lig_pos_traj,
            lig_batch = data.lig_node_batch.detach().cpu(),
        )]
        # return a list of cpu Tensor (tuple (Tensor)) with shape (N_traj, N_subset, ..., 3)
        if self.diffusion_model_cfg.no_sc_torsion:
            return lig_pos_traj
        else:
            atom14_pos_traj = torch.stack(atom14_pos_out_list, dim = 1)
            atom14_pos_traj = [_pos.transpose(1, 0) for _pos in slice_protein(
                atom14_pos = atom14_pos_traj,
                atom14_mask = data.atom14_mask.detach().cpu(),
                rec_atm_pos_batch = data.rec_atm_pos_batch.detach().cpu(),
            )]
            assert len(atom14_pos_traj) == len(lig_pos_traj), 'ligand and protein batch size mismatches'
            return list(zip(lig_pos_traj, atom14_pos_traj))


def slice_lig(
        lig_pos: Tensor,
        lig_ptr: Optional[Tensor] = None,
        lig_batch: Optional[Tensor] = None,
        protein_center: Optional[Union[Tensor, List[Tensor]]] = None,
):
    lig_batch, lig_ptr = ptr_and_batch(lig_batch, lig_ptr)
    lig_pos_list = slice_tensor_batch(lig_pos, lig_batch)

    if protein_center is not None:
        lig_pos_list = add_protein_center(lig_pos_list, protein_center)

    return lig_pos_list


def add_protein_center(
        pos_list: List[Tensor],
        protein_center: Union[Tensor, List[Tensor]],
) -> List[Tensor]:
    """
    Args:
        pos_list: A list of Tensor with shape (*, 3) with length M
        protein_center: Tensor with shape (M, 3) or a list of Tensor with shape (3,)
            with length M.
    """
    if isinstance(protein_center, Tensor):
        protein_center = torch.unbind(protein_center, dim = 0)
    assert len(pos_list) == len(protein_center), 'protein_center batch size mismatches'
    shape = pos_list[0].shape
    assert shape[-1] == 3, f'The last dimension of input pos must be 3, but got {shape[-1]}'
    pos_list = [_pos + _center.view(*((1, ) * len(shape[:-1]) + (3,)))
                for _pos, _center in zip(pos_list, protein_center)]
    return pos_list

def slice_protein(
        atom14_pos: Tensor,
        atom14_mask: Tensor,
        rec_atm_pos_ptr: Optional[Tensor] = None,
        rec_atm_pos_batch: Optional[Tensor] = None,
        protein_center: Optional[Union[Tensor, List[Tensor]]] = None,
):
    """
    atom14_pos: shape with (N_res, N_traj, 14, 3),
    atom14_mask: shape with (N_res, 14, 3)
    Returns:
        A list of Tensor with shape (N_traj, N_res_subset, 14, 3)
    """
    assert atom14_pos.ndim == 4, 'atom14_pos shap must be (N_res, N_traj, 14, 3)'
    rec_atm_pos_batch, rec_atm_pos_ptr = ptr_and_batch(
        rec_atm_pos_batch, rec_atm_pos_ptr)

    if not isinstance(atom14_mask, BoolTensor):
        atom14_mask = atom14_mask.bool()
    # flatten batch to atom14 batch
    batch14 = torch.zeros(
        atom14_mask.shape,
        dtype = torch.long,
        device = atom14_mask.device,
    )
    batch14[atom14_mask] = rec_atm_pos_batch
    batch14 = torch.amax(batch14, dim = -1)

    atom14_pos_list = slice_tensor_batch(atom14_pos, batch14)
    if protein_center is not None:
        atom14_pos_list = add_protein_center(atom14_pos_list, protein_center)
        atom14_mask_list = slice_tensor_batch(atom14_mask, batch14)
        atom14_pos_list = [_atom * _mask.unsqueeze(1) for _atom, _mask in
                           zip(atom14_pos_list, atom14_mask_list)]

    return atom14_pos_list
