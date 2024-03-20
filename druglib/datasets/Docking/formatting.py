# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Any
from druglib.data import Data, DataContainer
from ..builder import PIPELINES

class PLData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __cat_dim__(self, key: str, value: DataContainer, *args, **kwargs) -> Any:
        if key in ['torsion_edge_index']:
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key: str, value: DataContainer, *args, **kwargs) -> Any:
        if key in ['lig_edge_index']:
            return self['lig_node'].size(0)
        elif key in ['torsion_edge_index']:
            # consider missing atoms
            if 'rec_atm_pos' in self:
                return self['rec_atm_pos'].size(0)
            elif 'atom14_position' in self:
                return self['atom14_position'][self['atom14_mask']].size(0)

        return super().__inc__(key, value, *args, **kwargs)


@PIPELINES.register_module()
class ToPLData:
    """
    Use `PLData` encapsulate data for easy collation.
    """
    def __call__(self, data) -> Data:
        return PLData(**data)

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f')')

@PIPELINES.register_module()
class Atom14ToAllAtomsRepr:
    """
    Cancel out atom14 repr to all atoms repr,
        so that no worries about mask.
    """
    def __call__(self, data):
        data['rec_atm_pos'] = data['atom14_position'][data['atom14_mask']]
        del data['atom14_position']
        data['pocket_node_feature'] = data['pocket_node_feature'][data['atom14_mask']]

        return data
