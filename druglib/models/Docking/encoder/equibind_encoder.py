# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Optional, Tuple
import torch
from torch import Tensor
import torch.nn as nn

from druglib.apis import xavier_init, kaiming_init


class AtomEncoder(nn.Module):
    """
    This :cls:`AtomEncoder` is from 'EQUIBIND: Geometric Deep Learning for Drug Binding Structure Prediction'
        from https://arxiv.org/pdf/2202.05146.pdf
    Merged modeified :cls:`AtomEncoder` from 'DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking'
        from https://arxiv.org/abs/2210.01776.pdf
    """
    def __init__(
            self,
            emb_dim: int,
            feature_dims: Tuple[Tuple[int], int],
            sigma_dim: Optional[int] = None,
            lm_embed_type: Optional[int] = None,
            n_feats_to_use: Optional[int] = None,
            use_bias: bool = False,
    ):
        super(AtomEncoder, self).__init__()
        self.n_feats_to_use = n_feats_to_use

        self.atom_emb_list = nn.ModuleList()
        # the first element of feature_dims tuple is a list
        # with the lenght of each categorical feature
        # the second is the number of scalar features
        self.num_onehot = len(feature_dims[0])
        self.scalar_dim = feature_dims[1]
        self.sigma_dim = sigma_dim
        if sigma_dim is not None:
            self.scalar_dim = feature_dims[1] + sigma_dim
        self.lm_embed_type = lm_embed_type
        for i, dim in enumerate(feature_dims[0]):
            emb = nn.Embedding(dim, emb_dim)
            self.atom_emb_list.append(emb)
            if i + 1 == self.n_feats_to_use:
                break
        if self.scalar_dim > 0:
            self.scalar_lin = nn.Linear(self.scalar_dim + emb_dim,
                                    emb_dim, bias = use_bias)

        if self.lm_embed_type is not None:
            if self.lm_embed_type == 'esm':
                self.lm_embed_dim = 1280
            else:
                raise ValueError(
                    'LM Embedding type was not correctly determined. LM embedding type: ',
                    self.lm_embed_type)
            self.lm_lin = nn.Linear(self.lm_embed_dim + emb_dim,
                                emb_dim, bias = use_bias)

    def init_weights(self):
        for layer in self.atom_emb_list:
            xavier_init(layer, distribution = 'uniform')
        if self.scalar_dim > 0:
            kaiming_init(self.scalar_lin, distribution = 'uniform')
        if self.lm_embed_type is not None:
            kaiming_init(self.lm_lin, distribution = 'uniform')

    def forward(self, x: Tensor):
        x_emb = 0
        if self.lm_embed_type is not None:
            assert x.shape[1] == self.num_onehot + self.scalar_dim + self.lm_embed_dim
        else:
            assert x.shape[1] == self.num_onehot + self.scalar_dim

        for i in range(self.num_onehot):
            x_emb += self.atom_emb_list[i](x[:, i].long())
            if i + 1 == self.n_feats_to_use:
                break

        if self.scalar_dim > 0:
            x_emb += self.scalar_lin(
                torch.cat([x_emb, x[:, self.num_onehot:self.num_onehot + self.scalar_dim]],
                          dim = -1))

        if self.lm_embed_type is not None:
            x_emb += self.lm_lin(
                torch.cat([x_emb, x[:, -self.lm_embed_dim:]],
                          dim = -1))

        return x_emb

