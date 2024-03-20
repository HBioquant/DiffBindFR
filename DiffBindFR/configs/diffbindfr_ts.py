batch_size = 8
inference_steps = 22
tr_sigma_min = 0.1
tr_sigma_max = 6
rot_sigma_min = 0.03
rot_sigma_max = 1.55
tor_sigma_min = 0.0314
tor_sigma_max = 3.14
sc_tor_sigma_min = 0.0314
sc_tor_sigma_max = 3.14

# Here, the test transform is specific to enable the separated ligand and protein pipeline
# Ligand part
# 'rot_node_mask' already in metastore
test_pre_transform_lig = [
    dict(
        type = 'LoadLigand',
        sanitize = True,
        calc_charges = True,
        remove_hs = True,
        assign_chirality = False,
        allow_genconf = False,
        # set int (>= 1), if users would like to the multiple rdkit conformer initialization
        emb_multiple_3d = None,
    ),
    dict(type = 'LigandFeaturizer'),
    dict(type = 'TorsionFactory'),
    dict(type = 'LigandGrapher'),
]
cystal_ligand_loader = test_pre_transform_lig[0:1]

# Protein part
test_pre_transform_prot = [
    dict(type = 'LoadProtein'),
    dict(
        type = 'SCPocketFinderDefault',
        by_ligand3d = True,
        point3d_obj = 'all',
        selection_mode = 'any',
        cutoff = 12,
    ),
    dict(type = 'PocketGraphBuilder'),
    dict(type = 'PocketFeaturizer'),
    # generate pocket_center_pos, move to origin
    dict(type = 'Decentration'),
]

# Filter (or called Collector) part
collect_data = [
    'lig_pos', 'lig_edge_index', 'lig_node', 'lig_edge_feat', 'tor_edge_mask',
    'atom14_position', 'atom14_mask', 'sequence', 'backbone_transl', 'backbone_rots',
    'default_frame', 'rigid_group_positions', 'torsion_angle', 'torsion_edge_index',
    'sc_torsion_edge_mask', 'pocket_node_feature', 'pocket_center_pos'
]
collect_transform = [
    dict(
        type = 'Collect',
        keys = collect_data,
        meta_keys = ['metastore']
    )
]

# Merge the above three pipelines
# test_pre_transform = test_pre_transform_lig + test_pre_transform_prot + collect_transform

# Real-time transform function
test_transform = [
    dict(
        type = 'LigInit',
        tr_sigma_max = 10, # tr_sigma_max,
    ),
    dict(type = 'SCFixer'),
    dict(type = 'SCProtInit'),
    dict(type = 'Atom14ToAllAtomsRepr'),
    dict(
        type = 'ToDataContainer',
        fields = (
            'lig_node', 'lig_pos', 'lig_edge_index', 'lig_edge_feat', 'tor_edge_mask',
            'pocket_node_feature', 'rec_atm_pos', 'sc_torsion_edge_mask', 'torsion_edge_index',
            'backbone_transl', 'sequence', 'atom14_mask', 'backbone_rots', 'default_frame',
            'rigid_group_positions', 'torsion_angle',
        ),
        excluded_keys = ('lig_node_type', 'ligand_edge_type')
    ),
    dict(type = 'ToPLData'),
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=8,
    # test=dict(
    #     pre_transform=test_pre_transform,
    #     transform=test_transform,
    # ),
    test_dataloader=dict(
        follow_batch=['lig_node', 'rec_atm_pos'],
        exclude_keys=None,
        use_bgg=True,
    ),
)

model = dict(
    task = 'mldock',
    type = 'DiffBindFR',
    diffusion_model = dict(
        type = 'TensorProductModel',
        cfg = dict(
            task = 'struct_gen', # 'struct_gen', 'RMSD_reg', 'RMSD_cls', 'affinity'
            no_sc_torsion = False,
            features_dim = {
                "protein_atom": {
                    'atom37_label': 37,
                    'atomcoarse22_label': 22,
                    'atom4_label': 4,
                    'aatype21_label': 21,
                    'is_backbone': 2,
                    'feature_list': ((37, 22, 4, 21, 2), 0)
                },
                "ligand_atom": {
                    'node_features': 27,
                    'edge_features': 10,
                }
            },
            ns = 48,
            nv = 12,
            sh_lmax = 2,
            lig_cutoff = 5,
            atom_cutoff = 4,
            cross_cutoff = 32,
            dynamic_max_cross = True,
            center_max_distance = 32,
            atom_max_neighbors = 1000,
            distance_embed_dim = 32,
            time_emb_type = "sinusoidal",
            sigma_embed_dim = 32,
            emb_scale = 1000,
            num_conv_layers = 6,
            use_second_order_repr = False,
            dropout = 0.1,
            batch_norm = True,
            scale_by_sigma = True,
        )
    ),
    test_cfg = dict(
        sample_cfg = dict(
            type = 'sde',
            batch_size = 32,
            time_schedule = 'linear',
            inference_steps = inference_steps,
            actual_steps = inference_steps - 2,
            eps = 1e-5,
            no_final_step_noise = True,
            no_random = False,
            tr_sigma_min = tr_sigma_min,
            tr_sigma_max = tr_sigma_max,
            rot_sigma_min = rot_sigma_min,
            rot_sigma_max = rot_sigma_max,
            tor_sigma_min = tor_sigma_min,
            tor_sigma_max = tor_sigma_max,
            sc_tor_sigma_min = sc_tor_sigma_min,
            sc_tor_sigma_max = sc_tor_sigma_max,
        )
    ),
)

log_level = 'INFO'
load_from = None
resume_from = None