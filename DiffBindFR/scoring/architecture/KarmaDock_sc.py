import torch
from torch import nn
from torch_scatter import scatter
from torch_geometric.nn import GraphNorm
from .GVP_Block import GVP_embedding
from .GraphTransformer_Block import GraghTransformer
from .MDN_Block import MDN_Block
from .EGNN_Block import EGNN
from .Gate_Block import Gate_Block
from .Angle_ResNet import AngleResnet


class KarmaDock(nn.Module):
    def __init__(self):
        super(KarmaDock, self).__init__()
        # encoders
        self.lig_encoder = GraghTransformer(
            in_channels=89, 
            edge_features=20, 
            num_hidden_channels=128,
            activ_fn=torch.nn.SiLU(),
            transformer_residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.15,
            num_layers=6
        )
        self.pro_encoder = GVP_embedding(
            (9, 3), (128, 16), (21, 1), (32, 1), seq_in=True) 
        self.gn = GraphNorm(128)
        # pose prediction
        self.egnn_layers = nn.ModuleList( 
            [EGNN(dim_in=128, dim_tmp=128, edge_in=128, edge_out=128, num_head=4, drop_rate=0.15) for i in range(8)]
        )
        self.edge_init_layer = nn.Linear(6, 128)
        self.node_gate_layer = Gate_Block(dim_tmp=128, 
                                          drop_rate=0.15
                                          )
        self.edge_gate_layer = Gate_Block(dim_tmp=128, 
                                          drop_rate=0.15
                                          )
        # scoring 
        self.mdn_layer = MDN_Block(
            hidden_dim=128,
            n_gaussians=10,
            dropout_rate=0.10,
            dist_threhold=7.
        )
        self.torsion_sin_cos_layer = AngleResnet(
            c_in=128,
            c_hidden=32,
            no_blocks=2,
            no_angles=4,
            epsilon=1e-6
        )

    def forward(self, data):
        batch_size = data['ligand'].batch[-1] + 1
        pro_node_s, lig_node_s = self.encoding(data)
        lig_pos = data['ligand'].xyz
        mdn_score_pred = self.scoring(
            lig_s=lig_node_s,
            lig_pos=lig_pos,
            pro_s=pro_node_s,
            data=data,
            dist_threhold=5.,
            batch_size=batch_size,
        )
        return mdn_score_pred

    def encoding(self, data):
        """get ligand & protein embeddings"""
        pro_node_s = self.pro_encoder(
            (
                data['protein']['node_s'],
                data['protein']['node_v']
            ),
            data[("protein", "p2p", "protein")]["edge_index"],
            (
                data[("protein", "p2p", "protein")]["edge_s"],
                data[("protein", "p2p", "protein")]["edge_v"]
            ),
            data['protein'].seq
        )
        lig_node_s = self.lig_encoder(data['ligand'].node_s.to(torch.float32), data['ligand', 'l2l', 'ligand'].edge_s[data['ligand'].cov_edge_mask].to(torch.float32), data['ligand', 'l2l', 'ligand'].edge_index[:,data['ligand'].cov_edge_mask])
        return pro_node_s, lig_node_s
    
    def scoring(self, lig_s, lig_pos, pro_s, data, dist_threhold, batch_size):
        """scoring the protein-ligand binding strength"""
        pi, sigma, mu, dist, c_batch, _, _ = self.mdn_layer(
            lig_s=lig_s,
            lig_pos=lig_pos,
            lig_batch=data['ligand'].batch,
            pro_s=pro_s,
            pro_pos=data['protein'].xyz_full,
            pro_batch=data['protein'].batch,
            edge_index=data['ligand', 'l2l', 'ligand'].edge_index[:, data['ligand'].cov_edge_mask]
        )
        mdn_score = self.mdn_layer.calculate_probablity(pi, sigma, mu, dist)
        mdn_score[torch.where(dist > dist_threhold)[0]] = 0.
        mdn_score = scatter(mdn_score, index=c_batch, dim=0, reduce='sum', dim_size=batch_size).float()
        return mdn_score