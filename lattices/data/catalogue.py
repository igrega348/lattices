import os
import warnings
from typing import Dict, Iterable, List, Union
import numpy as np
try:
    from tqdm import trange
except ImportError:
    warnings.warn('tqdm not found, progress bar will not be shown')
    def trange(*args, **kwargs):
        return range(*args, **kwargs)
import re

class CfgDict(dict):
    """dot.notation access to dictionary attributes"""
    # based on https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    # but allowing for implicit nesting
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __getattr__(self, key):
        val = self.get(key)
        if isinstance(val, dict):
            return CfgDict(val)
        return val
    
class Catalogue:
    """Unit cell catalogue object.

    Note:
        Two class methods are available to initialise the object:
            - :func:`from_file`
                Used to read the catalogue from a file

                >>> cat = Catalogue.from_file('Unit_Cell_Catalog.txt', 1)

            - :func:`from_dict`
                Used to create the catalogue from unit cells either from scratch
                or when unit cells from a file are modified

                >>> nodes = [[0,0,0],[1,0,0],[0.5,1,0],[0.5,1,1]]
                >>> edges = [[0,1],[1,2],[0,2],[0,3],[1,3],[2,3]]
                >>> lat = Lattice(nodal_positions=nodes, edge_adjacency=edges)
                >>> lat
                {'num_nodes': 4, 'num_edges': 6}
                >>> cat_dict = {'pyramid':lat.to_dict()}
                >>> cat = Catalogue.from_dict(cat_dict)


                See Also:
                    :func:`Lattice.to_dict`
    """
    names: List
    lines: dict
    INDEXING: int
    iter: int

    def __init__(self, data: dict, indexing: int, **attrs) -> None:
        self.lines = data
        self.names = list(data.keys())
        self.INDEXING = indexing
        for key in attrs:
            setattr(self, key, attrs[key])

    def __len__(self) -> int:
        return len(self.names)

    def __repr__(self) -> str:
        desc = "Unit cell catalogue "\
            f"with {len(self.names)} entries"
        return desc

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        if self.iter<len(self):
            data = self.get_unit_cell(self.names[self.iter])
            self.iter += 1
            return data
        else:
            raise StopIteration

    def __getitem__(self, ind: Union[int, slice]):
        if isinstance(ind, int):
            return self.get_unit_cell(self.names[ind])
        elif isinstance(ind, str):
            return self.get_unit_cell(ind)
        elif isinstance(ind, slice):
            selected_names = self.names[ind]
            selected_data = {name: self.lines[name] for name in selected_names}
            return Catalogue(data=selected_data, indexing=self.INDEXING)
        else:
            raise NotImplementedError

    @classmethod
    def get_names(cls, fn: str) -> List[str]:
        """Retrieve names from catalogue without reading it into memory.

        Args:
            fn (str): path to input file
            
        Returns:
            List[str]: list of names
        """
        names = []
        with open(fn, 'r') as fin:
            for line in fin:
                if line.startswith('Name:'):
                    phrases = line.split()
                    names.append(phrases[1])
        return names
         
    @classmethod
    def from_file(
        cls, fn: str, indexing: int, progress: bool = False, regex: str = None
    ) -> "Catalogue":
        """Read catalogue from a file.

        Args:
            fn (str): path to input file
            indexing (int): 0 or 1 as the basis of edge indexing
            progress (bool, optional): whether to show progress bar. Defaults to False.
            regex (str, optional): regular expression to filter names. Defaults to None.

        Returns:
            Catalogue
        """
        with open(fn, 'r') as fin:
            lines = fin.readlines()
        lines = [line.rstrip() for line in lines]
        line_ranges = dict()
        data = dict()

        name = None
        start_line = None
        end_line = None

        if progress:
            itr = trange(len(lines), desc=f'Loading catalogue {fn}')
        else:
            itr = range(len(lines))
    
        for i_line in itr:
            line = lines[i_line]

            if line.startswith('Name:'):
                if isinstance(name, str) and (not isinstance(end_line, int)):
                    assert isinstance(start_line, int)
                    end_line = i_line
                    line_ranges[name] = slice(start_line, end_line)
                    name = None
                    start_line = None
                    end_line = None

                phrases = line.split()
                name = phrases[1]
                if isinstance(regex, str):
                    if not re.match(regex, name):
                        name = None
                        continue
                start_line = i_line
                end_line = None
            elif 'lattice_transition' in line:
                if not isinstance(name, str): continue
                end_line = i_line
                line_ranges[name] = slice(start_line, end_line)
                name = None
                start_line = None
                end_line = None
            else:
                pass
            
        if isinstance(name, str) and (not isinstance(end_line, int)):
            assert isinstance(start_line, int)
            end_line = i_line+1
            line_ranges[name] = slice(start_line, end_line)

        for name in line_ranges.keys():
            text = lines[line_ranges[name]]
            data[name] = text
        
        attrs = {'fn':fn}
        return cls(data=data, indexing=indexing, **attrs)

    @classmethod
    def from_dict(cls, lattice_dicts: Dict[str, Dict]) -> "Catalogue":
        """Generate unit cell catalogue from dictionary representation

        Args:
            lattice_dicts (Dict[str, Dict]): dictionary of lattice dictionaries

        Returns:
            Catalogue

        Note:
            Lattice dictionaries must contain the following keys:
                - `edge_adjacency`
                - `nodal_positions` or `reduced_node_coordinates`
            They can contain also:
                - `lattice_constants`
                - `compliance_tensors`   # assuming Voigt notation, legacy support
                - `compliance_tensors_V` # Voigt notation
                - `compliance_tensors_M` # Mandel notation
                - `fundamental_edge_adjacency`
                - `fundamental_tesselation_vecs`
                - `fundamental_edge_radii`
                - `base_name`
                - `imperfection_kind`
                - `imperfection_level`
                - `nodal_hash`

        """
        data = dict()

        for name in lattice_dicts:
            lat_dict = lattice_dicts[name]
            lines = []

            if 'name' in lat_dict:
                assert lat_dict['name']==name

            lines.append(f'Name: {name}')
            lines.append('')

            if 'base_name' in lat_dict:
                lines.append(f"Base name: {lat_dict['base_name']}")
            if 'imperfection_level' in lat_dict:
                lines.append(f"Imperfection level: {lat_dict['imperfection_level']}")
            if 'imperfection_kind' in lat_dict:
                lines.append(f"Imperfection kind: {lat_dict['imperfection_kind']}")
            if 'nodal_hash' in lat_dict:
                lines.append(f"Nodal hash: {lat_dict['nodal_hash']}")

            if 'lattice_constants' in lat_dict:
                lattice_constants = lat_dict['lattice_constants']
                assert len(lattice_constants)==6
                lines.append(
                    'Normalized unit cell parameters (a,b,c,alpha,beta,gamma):'
                )
                line = ''
                for i, x in enumerate(lattice_constants):
                    line = line + f'{x:.5g}'
                    if i!=5:
                        line = line + ', '
                lines.append(line)
                lines.append('')

            if (('compliance_tensors_M' in lat_dict) 
            or ('compliance_tensors_V' in lat_dict) 
            or ('compliance_tensors' in lat_dict)):
                if 'compliance_tensors_M' in lat_dict:
                    compliance_tensors = lat_dict['compliance_tensors_M']
                    compl_format = 'Mandel'
                elif 'compliance_tensors_V' in lat_dict:
                    compliance_tensors = lat_dict['compliance_tensors_V']
                    compl_format = 'Voigt'
                else:
                    compliance_tensors = lat_dict['compliance_tensors']
                    compl_format = 'Voigt'
                lines.append('')
                lines.append(f'Compliance tensors ({compl_format}) start (flattened upper triangular)')
                for rel_dens in sorted(compliance_tensors.keys()):
                    lines.append(f'-> at relative density {rel_dens}:')
                    S = compliance_tensors[rel_dens]
                    assert S.shape==(6,6)
                    nums = S[np.triu_indices(6)].tolist()
                    nums = [f'{x:.5g}' for x in nums]
                    line = ','.join(nums)
                    lines.append(line)
                lines.append(f'Compliance tensors ({compl_format}) end')
                lines.append('')

            
            if 'fundamental_edge_radii' in lat_dict:
                assert 'fundamental_edge_adjacency' in lat_dict
                lines.append('Fundamental edge radii start')
                fundamental_edge_radii = lat_dict['fundamental_edge_radii']
                for rel_dens in sorted(fundamental_edge_radii.keys()):
                    lines.append(f'-> at relative density {rel_dens}:')
                    r = fundamental_edge_radii[rel_dens]
                    assert len(r)==len(lat_dict['fundamental_edge_adjacency'])
                    nums = [f'{x:.5g}' for x in r]
                    line = ','.join(nums)
                    lines.append(line)
                lines.append('Fundamental edge radii end')
                lines.append('')
            
            assert ('reduced_node_coordinates' in lat_dict) or ('nodal_positions' in lat_dict)
            assert 'edge_adjacency' in lat_dict
            
            if 'reduced_node_coordinates' in lat_dict:
                nodal_positions = lat_dict['reduced_node_coordinates']
            else:
                nodal_positions = lat_dict['nodal_positions']

            lines.append('Nodal positions:')
            for x,y,z in nodal_positions:
                lines.append(f'{x:.5g} {y:.5g} {z:.5g}')

            lines.append('')

            lines.append('Bar connectivities:')
            for e in lat_dict['edge_adjacency']:
                lines.append(f'{int(e[0])} {int(e[1])}')

            lines.append('')

            if 'fundamental_edge_adjacency' in lat_dict:
                assert 'fundamental_tesselation_vecs' in lat_dict
                
                lines.append('Fundamental edge adjacency')
                for e in lat_dict['fundamental_edge_adjacency']:
                    lines.append(f'{int(e[0])} {int(e[1])}')

                lines.append('')

                lines.append('Fundamental tesselation vectors')
                for v in lat_dict['fundamental_tesselation_vecs']:
                    line = ' '.join([f'{x:.5g}' for x in v])
                    lines.append(line)

                lines.append('')
        
            data.update({name:lines})

        return cls(data=data, indexing=0)

    def get_unit_cell(self, name: str) -> dict:
        """Return a dictionary which represents unit cell.

        Args:
            name (str): Name of the unit cell from the catalogue that 
                will be returned
            
        Returns:
            dict: Dictionary describing the unit cell

        Note:
            Returned dictionary contains all available keys from the following:
                - `name`
                - `lattice constants`: [a,b,c,alpha,beta,gamma]
                - `average connectivity`
                - `compliance_tensors_M`: dictionary {rel_dens: compliance tensor}
                - `compliance_tensors_V`: dictionary {rel_dens: compliance tensor}
                - `nodal_positions`: nested list of shape (num_nodes, 3)
                - `edge_adjacency`: nested list of shape (num_edges, 2) (0-indexed)
                - `fundamental_edge_adjacency`: nested list of shape (num_fundamental_edges, 2) (0-indexed)
                - `fundamental_tesselation_vecs`: nested list of shape (num_fundamental_edges, 6) or (num_fundamental_edges, 3) (0-indexed)
                - `fundamental_edge_radii`: dictionary {rel_dens: list of edge radii}

            The dictionary can be unpacked in the creation of a `Lattice` object

                >>> from data import Lattice, Catalogue
                >>> cat = Catalogue.from_file('Unit_Cell_Catalog.txt', 1)
                >>> lat = Lattice(**cat.get_unit_cell(cat.names[0]))
                >>> lat
                {'name': 'cub_Z06.0_E1', 'num_nodes': 8, 'num_edges': 12}

        See Also:
            :func:`data.Lattice.__init__`
        """
        lines = self.lines[name]

        uc_dict = CfgDict({})
        assert name in lines[0]
        uc_dict['name'] = name
        
        compl_start = compl_end = None
        fund_ea_start = fund_er_start = fund_tessvec_start = None
        nodal_hash = None

        for i_line, line in enumerate(lines):
            if 'unit cell parameters' in line:
                l_1 = lines[i_line+1]
                lat_params = [float(w) for w in l_1.split(',')]
                uc_dict['lattice_constants'] = lat_params
            elif 'connectivity' in line:
                z = float(line.split('Z_avg = ')[1])
                uc_dict['average_connectivity'] = z
            elif 'Base name' in line:
                base_name = line.split(':')[1].lstrip()
                uc_dict['base_name'] = base_name
            elif 'Imperfection level' in line:
                imp_level = line.split(':')[1].lstrip()
                uc_dict['imperfection_level'] = float(imp_level)
            elif 'Imperfection kind' in line:
                imp_kind = line.split(':')[1].lstrip()
                uc_dict['imperfection_kind'] = imp_kind
            elif 'Nodal hash' in line:
                nodal_hash = line.split(':')[1].lstrip()
                uc_dict['nodal_hash'] = nodal_hash
            elif 'Compliance tensors' in line:
                if 'start' in line:
                    compl_start = i_line
                elif 'end' in line:
                    compl_end = i_line
                if 'Mandel' in line:
                    compl_format = 'M'
                else:
                    compl_format = 'V'
            elif 'Nodal positions' in line:
                nod_pos_start = i_line
            elif 'Bar connectivities' in line:
                edge_adj_start = i_line
            elif 'Fundamental edge adjacency' in line:
                fund_ea_start = i_line
            elif 'Fundamental tesselation vectors' in line:
                fund_tessvec_start = i_line
            elif 'Fundamental edge radii start' in line:
                fund_er_start = i_line
            elif 'Fundamental edge radii end' in line:
                fund_er_end = i_line
            

        nodal_coords = []
        for i_line in range(nod_pos_start+1, len(lines)):
            line = lines[i_line]
            if len(line)>1:
                nc = [float(w) for w in line.split()]
                nodal_coords.append(nc)
            else:
                break
        uc_dict['nodal_positions'] = nodal_coords


        edge_adjacency = []
        for i_line in range(edge_adj_start+1, len(lines)):
            line = lines[i_line]
            if len(line)>1:
                ea = [int(w)-self.INDEXING for w in line.split()]
                edge_adjacency.append(ea)
            else:
                break
        uc_dict['edge_adjacency'] = edge_adjacency

        if isinstance(fund_ea_start, int):
            fund_edge_adjacency = []
            for i_line in range(fund_ea_start+1, len(lines)):
                line = lines[i_line]
                if len(line)>1:
                    ea = [int(w)-self.INDEXING for w in line.split()]
                    fund_edge_adjacency.append(ea)
                else:
                    break
            uc_dict['fundamental_edge_adjacency'] = fund_edge_adjacency

            fund_tess_vec = []
            for i_line in range(fund_tessvec_start+1, len(lines)):
                line = lines[i_line]
                if len(line)>1:
                    nc = [float(w) for w in line.split()]
                    fund_tess_vec.append(nc)
                else:
                    break
            uc_dict['fundamental_tesselation_vecs'] = fund_tess_vec

        if isinstance(compl_start, int):
            assert isinstance(compl_end, int)
            compliance_tensors = dict()
            for i_line in range(compl_start+1, compl_end):
                line = lines[i_line]
                if 'at relative density' in line:
                    rel_dens = float(line.rstrip(':').split('density')[1])
                    line = lines[i_line+1]
                    nums = []
                    for num in line.split(','):
                        try:
                            s = float(num)
                        except ValueError:
                            continue
                        nums.append(s)
                    assert len(nums)==21
                    S = np.zeros((6,6))
                    S[np.triu_indices(6)] = nums
                    S[np.triu_indices(6)[::-1]] = nums
                    compliance_tensors[rel_dens] = S
            uc_dict[f'compliance_tensors_{compl_format}'] = compliance_tensors

        if isinstance(fund_er_start, int):
            assert isinstance(fund_er_end, int)
            fund_edge_radii = dict()
            for i_line in range(fund_er_start+1, fund_er_end):
                line = lines[i_line]
                if 'at relative density' in line:
                    rel_dens = float(line.rstrip(':').split('density')[1])
                    line = lines[i_line+1]
                    nums = [float(x) for x in line.split(',')]
                    fund_edge_radii[rel_dens] = nums
            uc_dict['fundamental_edge_radii'] = fund_edge_radii

        return uc_dict

    def to_file(self, fn: str) -> None:
        """Export unit cell catalogue to file.

        Args:
            fn (str): output file path
        """
        outlines = []

        for name in self.lines.keys():
            outlines.append('----- lattice_transition -----')
            outlines.extend(self.lines[name])
        
        outlines = [line + '\n' for line in outlines]
        if len(os.path.dirname(fn))>0:
            os.makedirs(os.path.dirname(fn), exist_ok=True)
        with open(fn, 'w') as fout:
            fout.writelines(outlines)

    @staticmethod
    def n_2_bn(name: str) -> str:
        """Convert name of a unit cell to base name.

        Relies on the following naming convention:
            [base_name]_p_[imperfection_level]_[nodal_hash]
        where `base_name` is of the form:
            [symmetry]_[connectivity]_[code]
        For instance:
            cub_Z12.0_E19_p_0.0_9113732828474860344

        Args:
            name (str): Name of the unit cell

        Returns:
            str: Base name of the unit cell
        """
        fields = name.split('_')
        assert len(fields)>=3
        return '_'.join(fields[:3])