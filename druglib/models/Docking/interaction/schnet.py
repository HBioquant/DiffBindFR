# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Union, Callable, Optional, Tuple
from math import pi as PI
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter
from torch_geometric.nn import radius_graph

from druglib.apis import glorot_init, xavier_init, get_activation

class NeighborEmbedding(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_rbf: int,
            start: float,
            stop: float,
            max_z: int = 100,
            aggr: str = "add"
    ):
        super(NeighborEmbedding, self).__init__()
        self.aggr = aggr
        self.embedding = nn.Embedding(max_z, hidden_dim)
        self.distance_proj = nn.Linear(num_rbf, hidden_dim)
        self.cat = nn.Linear(hidden_dim * 2, hidden_dim)
        self.cutoff = CosineCutoff(start, stop)

    def init_weight(self):
        self.embedding.reset_parameters()
        xavier_init(self.distance_proj, distribution = 'uniform')
        xavier_init(self.cat, distribution = 'uniform')

    def forward(
            self,
            z: Tensor, # shape (num_nodes) # node atom elemnet
            x: Tensor, # shape (num_nodes, hidden_dim)
            edge_index: Tensor, # shape (2, num_edges)
            edge_weight: Tensor, # shape (num_edges)
            edge_attr: Tensor, # (num_edges, num_rbf)
    ):
        # remove self loops
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)
        x_neighbors = self.embedding(z)
        # propagate_type: (x: Tensor, W: Tensor)
        row, col = edge_index
        x_neighbors = x_neighbors[row]
        x_neighbors *= W
        num_nodes = x.size(0)
        x_neighbors = scatter(x_neighbors, col, dim = 0, dim_size = num_nodes, reduce = self.aggr)
        x_neighbors = self.cat(torch.cat(x, x_neighbors), dim = -1)

        return x_neighbors

class Distance(nn.Module):
    def __init__(
        self,
        start: float,
        stop: float,
        max_num_neighbors: int = 32,
        return_vecs: bool = False,
        loop: bool = False,
    ):
        super(Distance, self).__init__()
        self.start = start
        self.stop = stop
        self.max_num_neighbors = max_num_neighbors
        self.return_vecs = return_vecs
        self.loop = loop

    def forward(
            self,
            pos, # shape (num_nodes, 3)
            batch, # shape (num_nodes)
    ):
        edge_index = radius_graph(
            pos,
            r = self.stop,
            batch = batch,
            loop = self.loop,
            max_num_neighbors = self.max_num_neighbors + 1,
        )
        # make sure we didn't miss any neighbors due to max_num_neighbors
        assert not (torch.unique(edge_index[0], return_counts = True)[1] > self.max_num_neighbors).any(), \
            ("The neighbor search missed some atoms due to upper cutoff args `stop` being too low. "
            "Please increase this parameter to include the maximum number of atoms within the cutoff.")

        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        mask: Optional[torch.Tensor] = None
        if self.loop:
            # mask out self loops when computing distances because
            # the norm of 0 produces NaN gradients
            # NOTE: might influence force predictions as self loop gradients are ignored
            mask = (edge_index[0] != edge_index[1])
            edge_weight = torch.zeros(edge_vec.size(0), device = edge_vec.device)
            edge_weight[mask] = torch.norm(edge_vec[mask], dim = -1)
        else:
            edge_weight = torch.norm(edge_vec, dim = -1)

        lower_mask = edge_weight >= self.start
        if self.loop and mask is not None:
            # keep self loops even though they might be below the lower cutoff
            lower_mask = lower_mask | ~mask
        edge_index = edge_index[:, lower_mask]
        edge_weight = edge_weight[lower_mask]

        if self.return_vecs:
            edge_vec = edge_vec[lower_mask]
            return edge_index, edge_weight, edge_vec
        # Union typing works with TorchScript (https://github.com/pytorch/pytorch/pull/53180)
        return edge_index, edge_weight, None

class EdgeExpansion(nn.Module):
    def __init__(
            self,
            edge_dim: int,
            eps: float = 1e-9,
    ):
        super(EdgeExpansion, self).__init__()
        self.lin = nn.Linear(1, edge_dim, bias = False)
        self.eps = eps

    def init_weight(self):
        glorot_init(self.lin)

    def forward(self, edge_vector: Tensor): # shape (num_edges, 3)
        edge_vector = edge_vector / torch.linalg.norm(edge_vector,
                      dim = 1, keepdim = True).clamp_min(self.eps)
        expansion = self.lin(edge_vector.unsqueeze(-1)).transpose(1, -1)

        return expansion

class GaussianSmearing(nn.Module):
    def __init__(
            self,
            start: float = 0.0,
            stop: float = 10.0,
            num_gaussians: int = 50,
            trainable: bool = False,
    ):
        super(GaussianSmearing, self).__init__()
        self.trainable = trainable
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians

        offset, coeff = self._init_params()
        if trainable:
            self.register_parameter('coeff', coeff)
            self.register_parameter('offset', offset)
        else:
            self.register_buffer('coeff', coeff)
            self.register_buffer('offset', offset)

    def _init_params(self):
        offset = torch.linspace(self.start, self.stop, self.num_gaussians)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def init_weight(self):
        if self.trainable:
            offset, coeff = self._init_params()
            self.offset.data.copy_(offset)
            self.coeff.data.copy_(coeff)

    def forward(self, dist: Tensor): # shape (num_edges)
        dist = dist.clamp_max(self.stop) # shape (num_edges)
        dist = dist.unsqueeze(-1) - self.offset # shape (num_edges, num_gaussians)

        return torch.exp(self.coeff * torch.pow(dist, 2))

class CosineCutoff(nn.Module):
    def __init__(
            self,
            start: float = 0.0,
            stop: float = 5.0,
    ):
        super(CosineCutoff, self).__init__()
        self.start = start
        self.stop = stop

    def forward(self, dist: Tensor):
        if self.start > 0:
            cutoffs = 0.5 * (torch.cos(PI * (2 * (dist - self.start)
                      / (self.stop - self.start) + 1.0)) + 1.0)
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (dist < self.stop).float()
            cutoffs = cutoffs * (dist > self.start).float()

            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(dist * PI / self.stop) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (dist < self.stop).float()

            return cutoffs

class ExpNormalSmearing(nn.Module):
    def __init__(
            self,
            start: float = 0.0,
            stop: float = 10.0,
            num_gaussians: int = 50,
            trainable: bool = False,
    ):
        super(ExpNormalSmearing, self).__init__()
        self.trainable = trainable
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians

        self.cutoff_fn = CosineCutoff(0, stop)
        self.alpha = 5.0 / (stop - start)

        means, betas = self._init_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _init_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(self.start - self.stop)
        )
        means = torch.linspace(start_value, 1, self.num_gaussians)
        betas = torch.tensor(
            [(2 / self.num_gaussians * (1 - start_value)) ** -2] * self.num_gaussians
        )
        return means, betas

    def init_weight(self):
        if self.trainable:
            means, betas = self._initial_params()
            self.means.data.copy_(means)
            self.betas.data.copy_(betas)

    def forward(self, dist: Tensor): # shape (num_edges)
        dist = dist.unsqueeze(-1)

        return self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (self.start - dist)) - self.means) ** 2
        )


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift

class CFConv(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_filters: int,
        net: nn.Module,
        start: float,
        stop: float,
        aggr: str = "add",
    ):
        super(CFConv, self).__init__()
        self.lin1 = nn.Linear(in_dim, num_filters, bias = False)
        self.lin2 = nn.Linear(num_filters, out_dim)
        self.net = net
        self.cutoff = CosineCutoff(start, stop)
        self.aggr = aggr

    def init_weight(self):
        xavier_init(self.lin1, distribution = 'uniform')
        xavier_init(self.lin2, distribution = 'uniform')

    def forward(
            self,
            x, # shape (num_nodes, in_dim)
            edge_index, # shape (2, num_edges)
            edge_weight, # shape (num_edges)
            edge_attr, # shape (num_edges, rbf)
    ):
        num_nodes = x.size(0)
        C = self.cutoff(edge_weight)
        W = self.net(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        row, col = edge_index
        x = x[row] * W
        x = scatter(x, col, dim = 0, dim_size = num_nodes, reduce = self.aggr)
        x = self.lin2(x)

        return x

class InteractionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_rbf: int,
        num_filters: int,
        start: float,
        stop: float,
        act: Union[str, Callable, None] = None,
        aggr: str = "add",
    ):
        super(InteractionBlock, self).__init__()
        if isinstance(act, str):
            act = get_activation(act)()
        self.act = act

        self.mlp = nn.Sequential(
            nn.Linear(num_rbf, num_filters),
            act,
            nn.Linear(num_filters, num_filters),
        )
        self.conv = CFConv(
            hidden_dim,
            hidden_dim,
            num_filters,
            self.mlp,
            start, stop, aggr,
        ).jittable()
        self.lin = nn.Linear(hidden_dim, hidden_dim)

    def init_weight(self):
        xavier_init(self.mlp[0], distribution = 'uniform')
        xavier_init(self.mlp[2], distribution = 'uniform')
        self.conv.init_weight()
        xavier_init(self.lin, distribution = 'uniform')

    def forward(
            self,
            x, # shape (num_nodes, in_dim)
            edge_index, # shape (2, num_edges)
            edge_weight, # shape (num_edges)
            edge_attr, # shape (num_edges, rbf)
    ):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x

class SchNet(nn.Module):
    """
    Also called TorchMD Graph Network.
    The continuous-filter convolutional neural network SchNet from the
        'SchNet: A Continuous-filter Convolutional Neural Network for Modeling
        Quantum Interactions' from https://arxiv.org/abs/1706.08566 paper that uses
        the interactions blocks of the form:
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),
        where :math:`h_{\mathbf{\Theta}}` denotes an MLP and
        :math:`\mathbf{e}_{j,i}` denotes the interatomic distances between atoms.
    """
    def __init__(
            self,
            hidden_dim: int = 128,
            num_filters: int = 128,
            num_layers: int = 6,
            num_rbf: int = 50,
            rbf_type: str = "expnorm",
            trainable_rbf: bool = True,
            act: str = "silu",
            attn_act: str = "silu",
            neighbor_embedding: bool = True,
            start: float = 0.0,
            stop: float = 5.0,
            max_z: int = 100,
            max_num_neighbors: int = 32,
            aggr: str = "add",
    ):
        super(SchNet, self).__init__()
        assert rbf_type in rbf_class_mapping, (
            f'Unknown RBF type "{rbf_type}". '
            f'Choose from {", ".join(rbf_class_mapping.keys())}.')
        assert aggr in ["add", "mean", "max", ], \
        'Argument aggr must be one of: "add", "mean", or "max"'

        act = get_activation(act)
        self.embedding = nn.Embedding(max_z, hidden_dim)

        self.distance = Distance(
            start, stop,
            max_num_neighbors = max_num_neighbors,
            loop = True,
        )
        self.distance_expansion = rbf_class_mapping[rbf_type](
            start, stop, num_rbf, trainable_rbf)
        self.neighbor_embedding = (
            NeighborEmbedding(
            hidden_dim, num_rbf, start, stop, max_z
            ).jittable()
            if neighbor_embedding
            else None
        )

        self.interactions = nn.ModuleList()
        for _ in range(num_layers):
            block = InteractionBlock(
                hidden_dim,
                num_rbf,
                num_filters,
                start, stop,
                act, aggr,
            )
            self.interactions.append(block)

    def init_weight(self):
        self.embedding.reset_parameters()
        self.distance_expansion.init_weight()
        if self.neighbor_embedding is not None:
            self.neighbor_embedding.init_weight()
        for interaction in self.interactions:
            interaction.init_weight()

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor, Tensor]:

        x = self.embedding(z)
        edge_index, edge_weight, _ = self.distance(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)
        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)
        for interaction in self.interactions:
            x = x + interaction(x, edge_index, edge_weight, edge_attr)

        return x, None, z, pos, batch


rbf_class_mapping = {
    'gauss': GaussianSmearing,
    'expnorm': ExpNormalSmearing
}