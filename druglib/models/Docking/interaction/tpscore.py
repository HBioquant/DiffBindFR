# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Optional, Union, Callable

import torch
from torch import nn
from torch.nn import functional as F
from e3nn import o3
from torch_cluster import radius, radius_graph
from torch_scatter import scatter, scatter_mean

import druglib.apis as apis
from druglib.utils.obj import protein_constants as pc
from druglib.utils.torch_utils import get_complete_bipartite_graph
from .schnet import GaussianSmearing
from ..encoder.equibind_encoder import AtomEncoder
from ...builder import INTERACTION
from ...Base.diffusion.time_emb import get_timestep_embfunc


class LayerNorm(nn.Module):
    """V3 + Learnable mean shift"""
    def __init__(self, irreps, eps=1e-5, affine=True, normalization='component'):
        super().__init__()

        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps

        mean_shift = []
        for mul, ir in self.irreps:
            if ir.l == 0 and ir.p == 1:
                mean_shift.append(torch.ones(1, mul, 1))
            else:
                mean_shift.append(torch.zeros(1, mul, 1))
        mean_shift = torch.cat(mean_shift, dim=1)
        self.mean_shift = nn.Parameter(mean_shift)
        #self.register_parameter()
        
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    # @torch.autocast(device_type='cuda', enabled=False)
    def forward(self, node_input, **kwargs):
        
        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0
        i_mean_shift = 0

        for mul, ir in self.irreps:  
            d = ir.dim
            field = node_input.narrow(1, ix, mul * d)
            ix += mul * d

            field = field.reshape(-1, mul, d) # [batch * sample, mul, repr]
            
            field_mean = torch.mean(field, dim=1, keepdim=True) # [batch, 1, repr]
            field_mean = field_mean.expand(-1, mul, -1)
            mean_shift = self.mean_shift.narrow(1, i_mean_shift, mul)
            field = field - field_mean * mean_shift
            i_mean_shift += mul
                
            if self.normalization == 'norm':
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == 'component':
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            field_norm = torch.mean(field_norm, dim=1, keepdim=True)    
            field_norm = (field_norm + self.eps).pow(-0.5)  # [batch, mul]

            if self.affine:
                weight = self.affine_weight[None, iw: iw + mul]  # [batch, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch, mul]

            field = field * field_norm.reshape(-1, mul, 1)  # [batch * sample, mul, repr]

            if self.affine and d == 1 and ir.p == 1:  # scalars
                bias = self.affine_bias[ib: ib + mul]  # [batch, mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch * sample, mul, repr]

            # Save the result, to be stacked later with the rest
            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        if ix != dim:
            fmt = "`ix` should have reached node_input.size(-1) ({}), but it ended at {}"
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        output = torch.cat(fields, dim=-1)  # [batch * sample, stacked features]
        return output

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps})"

class SimpleLinear(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            hidden_dim: Optional[int] = None,
            dropout: float = 0.0,
            use_bias: bool = True,
            act: Union[str, Callable, None] = 'relu',
            **kwargs
    ):
        super(SimpleLinear, self).__init__()
        if isinstance(act, str):
            act = apis.get_activation(act)(**kwargs)
        elif act is None:
            act = nn.Identity()
        if hidden_dim is None:
            hidden_dim = out_dim

        self.lin = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias = use_bias),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim, bias = use_bias),
        )

    def init_weights(self):
        for l in self.lin:
            if isinstance(l, nn.Linear):
                apis.xavier_init(l, distribution = 'uniform')

    def forward(self, x):
        return self.lin(x)

class TensorProductConvLayer(nn.Module):
    def __init__(
            self,
            in_irreps: Union[str, o3.Irreps],
            sh_irreps: Union[str, o3.Irreps],
            out_irreps: Union[str, o3.Irreps],
            n_edge_features: int,
            residual: bool = True,
            batch_norm: bool = True,
            dropout: float = 0.0,
            hidden_features: Optional[int] = None,
    ):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if hidden_features is None:
            hidden_features = n_edge_features

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights = False)

        self.fc = SimpleLinear(
            in_dim = n_edge_features,
            out_dim = tp.weight_numel,
            hidden_dim = hidden_features,
            dropout = dropout,
        )

        self.batch_norm = LayerNorm(out_irreps) if batch_norm else None

    def init_weights(self):
        self.fc.init_weights()

    def forward(
            self,
            node_attr,
            edge_index,
            edge_attr,
            edge_sh,
            out_nodes = None,
            reduce = 'mean',
    ):
        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim = 0, dim_size = out_nodes, reduce = reduce)

        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)

        return out


@INTERACTION.register_module()
class TensorProductModel(nn.Module):
    """
    :Model:`TensorProductScoreMatching` is modified from 'DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking'
        from https://arxiv.org/abs/2210.01776.pdf
    Here we add side chain diffusion based on original rigid-protein blind docking task.
    So, the model can support ligand-flexible docking over the full protein (CA) or pocket (CA or all-atoms)
        or flexible docking in the pocket (all-atoms).
    When args `no_sc_torsion` is set to True, the model supports rigid receptor docking; otherwise,
        the model supports flexible docking
    We suggest the users train a CA model if the full protein input,
        while pocket input with all-atoms representation if users focus on the pocket docking.
    """
    def __init__(
            self,
            cfg,
    ):
        super(TensorProductModel, self).__init__()
        self.cfg = cfg
        self.no_sc_torsion = cfg.no_sc_torsion

        self.in_lig_edge_dim = cfg.features_dim.ligand_atom.edge_features
        self.in_lig_node_dim = cfg.features_dim.ligand_atom.node_features

        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax = cfg.sh_lmax)
        self.scale_by_sigma = cfg.scale_by_sigma

        self.num_conv_layers = cfg.num_conv_layers

        self.timestep_emb_func = get_timestep_embfunc(
            emb_type = cfg.time_emb_type,
            emb_dim = cfg.sigma_embed_dim,
            emb_scale = cfg.emb_scale,
        )

        self.lig_cutoff = cfg.lig_cutoff
        self.atom_cutoff = cfg.atom_cutoff
        self.cross_cutoff = cfg.cross_cutoff
        self.dynamic_max_cross = cfg.dynamic_max_cross
        self.atom_max_neighbors = cfg.atom_max_neighbors

        # embedding layers
        atom_emb_features = cfg.features_dim.protein_atom.feature_list
        ns, nv = cfg.ns, cfg.nv
        self.ns = ns
        sigma_embed_dim = cfg.sigma_embed_dim
        distance_embed_dim = cfg.distance_embed_dim
        dropout = cfg.dropout
        batch_norm = cfg.batch_norm
        self.lig_node_embedding = SimpleLinear(
            in_dim = self.in_lig_node_dim + sigma_embed_dim,
            out_dim = ns, dropout = dropout,
        )
        self.lig_edge_embedding = SimpleLinear(
            in_dim = self.in_lig_edge_dim + sigma_embed_dim + distance_embed_dim,
            out_dim = ns, dropout = dropout,
        )

        self.atom_node_embedding = AtomEncoder(ns, atom_emb_features, sigma_embed_dim,
                                               lm_embed_type = None, use_bias = False)
        self.atom_edge_embedding = SimpleLinear(
            in_dim = sigma_embed_dim + distance_embed_dim,
            out_dim = ns, dropout = dropout,
        )
        self.la_edge_embedding = SimpleLinear(
            in_dim = sigma_embed_dim + distance_embed_dim,
            out_dim = ns, dropout = dropout,
        )

        self.lig_distance_expansion = GaussianSmearing(0.0, self.lig_cutoff, distance_embed_dim)
        self.atom_distance_expansion = GaussianSmearing(0.0, self.atom_cutoff, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, self.cross_cutoff, distance_embed_dim)

        if cfg.use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]
        self.lig_conv_layers = nn.ModuleList()
        self.atom_conv_layers = nn.ModuleList()
        self.cross_al_conv_layers = nn.ModuleList()
        self.cross_la_conv_layers = nn.ModuleList()
        for i in range(self.num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout,
            }
            self.lig_conv_layers.append(TensorProductConvLayer(**parameters))
            self.atom_conv_layers.append(TensorProductConvLayer(**parameters))
            self.cross_al_conv_layers.append(TensorProductConvLayer(**parameters))
            self.cross_la_conv_layers.append(TensorProductConvLayer(**parameters))

        self.TaskModelInit()

    def TaskModelInit(self):
        """
        A lazy hook to endow the TP model for different task:
            1. score matching for binding structure generation;
            2. confidence model for binding structure RMSD prediction or classification;
            3. binding affinity prediction of binding structure.
        """
        tasks = ['struct_gen', 'RMSD_reg', 'RMSD_cls', 'affinity']
        assert self.cfg.task in tasks
        if self.cfg.task == 'struct_gen':
            self._ScoreMatchingInit()
        elif self.cfg.task == 'RMSD_reg':
            self._RMSDRegInit()
        elif self.cfg.task == 'RMSD_cls':
            self._RMSDClsInit()
        elif self.cfg.task == 'affinity':
            self._AffinityPred()
        else:
            raise NotImplementedError(f'Current model only support "{", ".join(tasks)}", but got {self.cfg.task}')

    def _ScoreMatchingInit(self):
        """score matching for binding structure generation"""
        cfg = self.cfg
        ns, nv = cfg.ns, cfg.nv
        center_max_distance = cfg.center_max_distance
        sigma_embed_dim = cfg.sigma_embed_dim
        distance_embed_dim = cfg.distance_embed_dim
        dropout = cfg.dropout
        batch_norm = cfg.batch_norm
        # convolution for translational, rotational, torsional scores of ligand
        self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance, distance_embed_dim)
        self.center_edge_embedding = SimpleLinear(
            in_dim = distance_embed_dim + sigma_embed_dim,
            out_dim = ns, dropout = dropout,
        )

        self.final_conv = TensorProductConvLayer(
            in_irreps = self.lig_conv_layers[-1].out_irreps,
            sh_irreps = self.sh_irreps,
            out_irreps = f'2x1o + 2x1e',
            n_edge_features = 2 * ns,
            residual = False,
            dropout = dropout,
            batch_norm = batch_norm
        )

        self.tr_final_layer = SimpleLinear(
            in_dim = 1 + sigma_embed_dim,
            out_dim = 1, hidden_dim = ns,
            dropout = dropout,
        )
        self.rot_final_layer = SimpleLinear(
            in_dim = 1 + sigma_embed_dim,
            out_dim = 1, hidden_dim = ns,
            dropout = dropout,
        )

        self.tor_edge_embedding = SimpleLinear(
            in_dim = distance_embed_dim,
            out_dim = ns, dropout = dropout,
        )
        self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
        self.tor_bond_conv = TensorProductConvLayer(
            in_irreps = self.lig_conv_layers[-1].out_irreps,
            sh_irreps = self.final_tp_tor.irreps_out,
            out_irreps = f'{ns}x0o + {ns}x0e',
            n_edge_features = 3 * ns,
            residual = False,
            dropout = dropout,
            batch_norm = batch_norm,
        )
        self.tor_final_layer = SimpleLinear(
            in_dim = 2 * ns,
            out_dim = 1, hidden_dim = ns,
            dropout = dropout, use_bias = False,
            act = 'tanh',
        )

        if not self.no_sc_torsion:
            # convolution for torsional score of protein side chain
            self.sc_edge_embedding = SimpleLinear(
                in_dim = distance_embed_dim,
                out_dim = ns, dropout = dropout,
            )
            self.sc_tor_bond_conv = TensorProductConvLayer(
                in_irreps = self.atom_conv_layers[-1].out_irreps,
                sh_irreps = self.final_tp_tor.irreps_out,
                out_irreps = f'{ns}x0o + {ns}x0e',
                n_edge_features = 3 * ns,
                residual = False,
                dropout = dropout,
                batch_norm = batch_norm
            )
            self.sc_tor_final_layer = SimpleLinear(
                in_dim = 2 * ns,
                out_dim = 1, hidden_dim = ns,
                dropout = dropout, use_bias = False,
                act = 'tanh',
            )

    def _RMSDRegInit(self):
        """
        Complex encoding for RMSD Regression.
        """
        cfg = self.cfg
        ns, nv = cfg.ns, cfg.nv
        dropout = cfg.dropout
        batch_norm = cfg.batch_norm
        self.Predictor = nn.Sequential(
            nn.Linear(2 * self.ns if self.num_conv_layers >= 3 else self.ns, ns),
            nn.BatchNorm1d(ns) if not batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
            nn.BatchNorm1d(ns) if not batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, 1)
        )

    def _RMSDClsInit(self, num_labels: int = 2):
        """
        Complex encoding for RMSD Regression.
        """
        assert num_labels > 1
        cfg = self.cfg
        ns, nv = cfg.ns, cfg.nv
        dropout = cfg.dropout
        batch_norm = cfg.batch_norm
        self.Predictor = nn.Sequential(
            nn.Linear(2 * self.ns if self.num_conv_layers >= 3 else self.ns, ns),
            nn.BatchNorm1d(ns) if not batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
            nn.BatchNorm1d(ns) if not batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, num_labels)
        )

    def _AffinityPred(self):
        """The same network with RMSD Reg"""
        self._RMSDRegInit()

    def init_weights(self):
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights()

    def forward(self, data):
        data.num_graphs = data.lig_node_batch.max().item() + 1
        data.time_emb = self.timestep_emb_func(data.t)

        # build ligand graph
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh = self.build_lig_conv_graph(data)
        lig_node_attr = self.lig_node_embedding(lig_node_attr)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)

        # build atom graph
        atom_node_attr, atom_edge_index, atom_edge_attr, atom_edge_sh = self.build_atom_conv_graph(data)
        atom_node_attr = self.atom_node_embedding(atom_node_attr)
        atom_edge_attr = self.atom_edge_embedding(atom_edge_attr)

        # build cross graph: ligand to receptor
        data.tr_sigma = data.tr_sigma.unsqueeze(1)
        la_edge_index, la_edge_attr, la_edge_sh = self.build_cross_conv_graph(data)
        la_edge_attr = self.la_edge_embedding(la_edge_attr)

        if not self.no_sc_torsion:
            # build side chain edge i->j->k<-l, only j-k
            data.sc_torsion_edge_index = data.pop('torsion_edge_index')[data.sc_torsion_edge_mask].T

        for l in range(self.num_conv_layers):
            # LIGAND updates
            lig_edge_attr_ = torch.cat(
                [lig_edge_attr, lig_node_attr[lig_edge_index[0], :self.ns],
                 lig_node_attr[lig_edge_index[1], :self.ns]], -1)
            lig_update = self.lig_conv_layers[l](lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh)

            # RECEPTOR TO LIGAND
            al_edge_attr_ = torch.cat(
                [la_edge_attr, lig_node_attr[la_edge_index[0], :self.ns],
                 atom_node_attr[la_edge_index[1], :self.ns]], -1)
            al_update = self.cross_al_conv_layers[l](atom_node_attr, la_edge_index, al_edge_attr_, la_edge_sh,
                                                     out_nodes = lig_node_attr.shape[0])

            # ATOM UPDATES
            atom_edge_attr_ = torch.cat(
                [atom_edge_attr, atom_node_attr[atom_edge_index[0], :self.ns],
                 atom_node_attr[atom_edge_index[1], :self.ns]], -1)
            atom_update = self.atom_conv_layers[l](atom_node_attr, atom_edge_index, atom_edge_attr_, atom_edge_sh)

            # LIGAND TO RECEPTOR
            la_edge_attr_ = torch.cat(
                [la_edge_attr, atom_node_attr[la_edge_index[1], :self.ns],
                 lig_node_attr[la_edge_index[0], :self.ns]], -1)
            la_update = self.cross_la_conv_layers[l](lig_node_attr, torch.flip(la_edge_index, dims = [0]), la_edge_attr_,
                                                     la_edge_sh, out_nodes = atom_node_attr.shape[0])

            # padding original features and update features with residual updates
            lig_node_attr = F.pad(lig_node_attr, (0, lig_update.shape[-1] - lig_node_attr.shape[-1]))
            lig_node_attr = lig_node_attr + lig_update + al_update
            atom_node_attr = F.pad(atom_node_attr, (0, atom_update.shape[-1] - atom_node_attr.shape[-1]))
            atom_node_attr = atom_node_attr + atom_update + la_update

        # task-wise head
        # confidence and affinity prediction
        if self.cfg.task != 'struct_gen':
            scalar_lig_attr = torch.cat(
                [lig_node_attr[:, :self.ns], lig_node_attr[:, -self.ns:]],
                dim = 1) if self.num_conv_layers >= 3 else lig_node_attr[:, :self.ns]
            pred = self.Predictor(
                scatter_mean(scalar_lig_attr, data.lig_node_batch, dim = 0))
            return pred

        # compute translational and rotational score vectors
        center_edge_index, center_edge_attr, center_edge_sh = self.build_center_conv_graph(data)
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_index[1], :self.ns]], -1)
        global_pred = self.final_conv(lig_node_attr, center_edge_index, center_edge_attr, center_edge_sh,
                                      out_nodes = data.num_graphs)

        tr_pred = global_pred[:, :3] + global_pred[:, 6:9]
        rot_pred = global_pred[:, 3:6] + global_pred[:, 9:]

        # adjust the magniture of the score vectors
        tr_norm = torch.linalg.vector_norm(tr_pred, dim = 1).unsqueeze(1)
        tr_pred = tr_pred / tr_norm * self.tr_final_layer(torch.cat([tr_norm, data.time_emb], dim = 1))

        rot_norm = torch.linalg.vector_norm(rot_pred, dim = 1).unsqueeze(1)
        rot_pred = rot_pred / rot_norm * self.rot_final_layer(torch.cat([rot_norm, data.time_emb], dim = 1))

        # torsional components
        if data.tor_edge_mask.sum() > 0:
            tor_edge_index, tor_edge_attr, tor_edge_sh = self.build_lig_bond_conv_graph(data, lig_node_attr)
            tor_pred = self.tor_bond_conv(lig_node_attr, tor_edge_index, tor_edge_attr, tor_edge_sh,
                                          out_nodes = data.tor_edge_mask.sum(), reduce = 'mean')
            tor_pred = self.tor_final_layer(tor_pred).squeeze(1)
        else:
            tor_pred = torch.empty(0, device = tr_pred.device)

        if self.scale_by_sigma:
            tr_pred = tr_pred / data.tr_sigma
            rot_pred = rot_pred * data.rot_score_norm
            if data.tor_edge_mask.sum() > 0:
                tor_pred = tor_pred * torch.sqrt(data.tor_score_norm2)

        if not self.no_sc_torsion:
            # side chain torsional components
            sc_tor_edge_index, sc_tor_edge_attr, sc_tor_edge_sh = self.build_sc_bond_conv_graph(data, atom_node_attr)
            sc_tor_pred = self.sc_tor_bond_conv(atom_node_attr, sc_tor_edge_index, sc_tor_edge_attr, sc_tor_edge_sh,
                                                out_nodes = data.sc_torsion_edge_mask.sum(), reduce = 'mean')
            sc_tor_pred = self.sc_tor_final_layer(sc_tor_pred).squeeze(1)

            data.sc_tor_score_norm2 = data.sc_tor_score_norm2[data.sc_torsion_edge_mask]
            if self.scale_by_sigma:
                sc_tor_pred = sc_tor_pred * torch.sqrt(data.sc_tor_score_norm2)

            return tr_pred, rot_pred, tor_pred, sc_tor_pred

        return tr_pred, rot_pred, tor_pred, None

    def build_lig_conv_graph(self, data):
        """
        Build the graph between ligand atoms
        Build dynamic ligand graph:
            1) update node attr: input lig node + time emb;
            2) additional radius edge index;
            3) edge_attr: additional radius edge attr + time emb + GS emb;
            4) source to target edge spherical harmonics;
        """
        data.lig_node_sigma_emb = data.time_emb[data.lig_node_batch]
        node_attr = torch.cat([data.lig_node, data.lig_node_sigma_emb], 1)
        radius_edges = radius_graph(data.lig_pos, self.lig_cutoff, data.lig_node_batch)
        edge_index = torch.cat([data.lig_edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data.lig_edge_feat,
            torch.zeros(radius_edges.shape[-1], self.in_lig_edge_dim,
            device = data.lig_pos.device)], 0)

        src, dst = edge_index
        edge_sigma_emb = data.lig_node_sigma_emb[src]
        edge_vec = data.lig_pos[dst] - data.lig_pos[src]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim = -1))
        edge_attr = torch.cat([edge_attr, edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize = True, normalization = 'component')

        return node_attr, edge_index, edge_attr, edge_sh

    def build_atom_conv_graph(self, data):
        """
        Build the graph between receptor atoms
        The graph is dynamic if the torsional side chains are movable in the semi-flexible docking
            1) update node attr: input prot atoms node + time emb;
            2) radius edge index;
            3) edge_attr: time emb + GS emb;
            4) source to target edge spherical harmonics;
        """
        atm_node_sigma_emb = data.time_emb[data.rec_atm_pos_batch]
        node_attr = torch.cat([data.pocket_node_feature, atm_node_sigma_emb], 1)
        edge_index = radius_graph(data.rec_atm_pos, self.atom_cutoff, data.rec_atm_pos_batch,
                                  max_num_neighbors = self.atom_max_neighbors)
        src, dst = edge_index
        edge_vec = data.rec_atm_pos[dst] - data.rec_atm_pos[src]
        edge_length_emb = self.atom_distance_expansion(edge_vec.norm(dim = -1))
        edge_sigma_emb = atm_node_sigma_emb[src]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize = True, normalization = 'component')

        return node_attr, edge_index, edge_attr, edge_sh

    def _build_cross_conv_graph(self, data):
        """
        Build receptor CA-CB and ligand atoms full-connected graph.
        Other atoms at receptor and ligand atoms graph will be dynamic scheduled
            by tr-sigma using radius graph.
        """
        # build full-connected graph for lig-ca and lig-cb
        device = data.pocket_node_feature.device
        atom37_id = data.pocket_node_feature[:, 0].long()
        cab_mask = torch.logical_or(
            atom37_id == pc.atom_order['CA'],
            atom37_id == pc.atom_order['CB'],
        )
        atomids = torch.arange(
            data.pocket_node_feature.size(0),
            device = device,
        )
        cab_indx = atomids[cab_mask]
        cab_batch = data.rec_atm_pos_batch[cab_indx]
        cab_bincount = torch.bincount(cab_batch)
        lig_bincount = torch.bincount(data.lig_node_batch)
        lrab_edge_index, _ = get_complete_bipartite_graph(lig_bincount, cab_bincount)
        lrab_edge_index = torch.stack([lrab_edge_index[0], cab_indx[lrab_edge_index[1]]], dim = 0)

        # different cutoff for every non-ca-cb-lig bigraph (depends on the diffusion time)
        nab_mask = torch.logical_not(cab_mask)
        nab_indx = atomids[nab_mask]
        nab_batch = data.rec_atm_pos_batch[nab_indx]
        nab_pos = data.rec_atm_pos[nab_mask]
        if self.dynamic_max_cross:
            cross_distance_cutoff = data.tr_sigma * 0.2 + 5
            lnab_edge_index = radius(nab_pos / cross_distance_cutoff[nab_batch],
                                     data.lig_pos / cross_distance_cutoff[data.lig_node_batch], 1,
                                     nab_batch, data.lig_node_batch, max_num_neighbors = 10000)
        else:
            lnab_edge_index = radius(nab_pos, data.lig_pos, self.cross_cutoff,
                                     nab_batch, data.lig_node_batch, max_num_neighbors = 10000)
        lnab_edge_index = torch.stack([lnab_edge_index[0], nab_indx[lnab_edge_index[1]]], dim = 0)
        la_edge_index = torch.cat([lrab_edge_index, lnab_edge_index], dim = 1).long()

        return la_edge_index

    def build_cross_conv_graph(self, data):
        """
        Build the cross edges between ligand atoms and receptor atoms.
        Message flow ligand (source) to receptor (target)
        Build dynamic cross graph:
            1) cross radius edge index;
            2) edge_attr: time emb + GS emb;
            3) source to target edge spherical harmonics;
        """
        la_edge_index = self._build_cross_conv_graph(data)
        la_edge_vec = data.rec_atm_pos[la_edge_index[1]] - data.lig_pos[la_edge_index[0]]
        la_edge_length_emb = self.cross_distance_expansion(la_edge_vec.norm(dim = -1))
        la_edge_sigma_emb = data.lig_node_sigma_emb[la_edge_index[0]]
        la_edge_attr = torch.cat([la_edge_sigma_emb, la_edge_length_emb], 1)
        la_edge_sh = o3.spherical_harmonics(self.sh_irreps, la_edge_vec, normalize = True, normalization = 'component')

        return la_edge_index, la_edge_attr, la_edge_sh

    def build_center_conv_graph(self, data):
        """
        Build the filter for the convolution of the center with the ligand atoms
            for translational and rotational score.
        Build dynamic cross graph:
            1) centroid to ligand atoms edge index;
            2) edge_attr: time emb + GS emb;
            3) source to target edge spherical harmonics;
        """
        device = data.lig_pos.device
        edge_index = torch.cat(
            [data.lig_node_batch.unsqueeze(0),
             torch.arange(len(data.lig_node_batch)).to(device).unsqueeze(0),
             ], dim = 0)
        edge_index = edge_index.long()

        center_pos = torch.zeros((data.num_graphs, 3)).to(device)
        center_pos.index_add_(0, index = data.lig_node_batch, source = data.lig_pos)
        center_pos = center_pos / torch.bincount(data.lig_node_batch).unsqueeze(1)

        edge_vec = data.lig_pos[edge_index[1]] - center_pos[edge_index[0]]
        edge_length_emb = self.center_distance_expansion(edge_vec.norm(dim = -1))
        edge_sigma_emb = data.lig_node_sigma_emb[edge_index[1]]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize = True, normalization = 'component')

        return edge_index, edge_attr, edge_sh

    def build_lig_bond_conv_graph(self, data, lig_node_attr):
        """Build graph for the pseudotorque layer"""
        bonds = data.lig_edge_index[:, data.tor_edge_mask.bool()]
        tor_bond_vec = data.lig_pos[bonds[1]] - data.lig_pos[bonds[0]]
        tor_bond_attr = lig_node_attr[bonds[0]] + lig_node_attr[bonds[1]]
        tor_bond_sh = o3.spherical_harmonics("2e", tor_bond_vec, normalize = True, normalization = 'component')

        bond_pos = (data.lig_pos[bonds[0]] + data.lig_pos[bonds[1]]) / 2
        bond_batch = data.lig_node_batch[bonds[0]]
        edge_index = radius(
            data.lig_pos, bond_pos, self.lig_cutoff,
            batch_x = data.lig_node_batch, batch_y = bond_batch)
        edge_vec = data.lig_pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim = -1))
        edge_attr = self.tor_edge_embedding(edge_attr)

        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize = True, normalization = 'component')
        tor_edge_sh = self.final_tp_tor(edge_sh, tor_bond_sh[edge_index[0]])

        tor_edge_attr = torch.cat([edge_attr, lig_node_attr[edge_index[1], :self.ns],
                                   tor_bond_attr[edge_index[0], :self.ns]], -1)

        return edge_index, tor_edge_attr, tor_edge_sh

    def build_sc_bond_conv_graph(self, data, atom_node_attr):
        """Build graph for the pseudotorque layer of receptor side chain"""
        # TODO: undirected graph!
        bonds = data.sc_torsion_edge_index
        sc_tor_bond_vec = data.rec_atm_pos[bonds[1]] - data.rec_atm_pos[bonds[0]]
        sc_tor_bond_attr = atom_node_attr[bonds[0]] + atom_node_attr[bonds[1]]
        sc_tor_bonds_sh = o3.spherical_harmonics("2e", sc_tor_bond_vec, normalize = True,
                                                 normalization = 'component')

        bond_pos = (data.rec_atm_pos[bonds[0]] + data.rec_atm_pos[bonds[1]]) / 2
        bond_batch = data.rec_atm_pos_batch[bonds[0]]
        edge_index = radius(
            data.rec_atm_pos, bond_pos, self.atom_cutoff,
            batch_x = data.rec_atm_pos_batch, batch_y = bond_batch)
        edge_vec = data.rec_atm_pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.atom_distance_expansion(edge_vec.norm(dim = -1))
        edge_attr = self.sc_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize = True, normalization = 'component')

        sc_tor_edge_sh = self.final_tp_tor(edge_sh, sc_tor_bonds_sh[edge_index[0]])
        sc_tor_edge_attr = torch.cat([edge_attr, atom_node_attr[edge_index[1], :self.ns],
                                      sc_tor_bond_attr[edge_index[0], :self.ns]], -1)

        return edge_index, sc_tor_edge_attr, sc_tor_edge_sh