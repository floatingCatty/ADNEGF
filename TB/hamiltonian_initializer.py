"""
The module contains functions facilitating setting tight-binding parameters and
initializing Hamiltonian objects from a Python dictionary.
"""

from __future__ import absolute_import
import sys
import numpy as np
from TB.orbitals import Orbitals
from TB import tb_params as dme
from TB.hamiltonian import Hamiltonian
from TB.hamiltonian_sparse import HamiltonianSp
import torch
import torch.nn as nn


def set_tb_params(orb, **kwargs):
    """Initialize a set of the user-defined tight-binding parameters.
    Parameters
    ----------
    **kwargs :

    Returns
    -------

    """
    for item in kwargs:
        if item.startswith('PARAMS_') or item.startswith('OV_'):
            for key in kwargs[item]:
                kwargs[item][key] = torch.tensor(kwargs[item][key], dtype=torch.float)
                # kwargs[item][key] = nn.Parameter(torch.tensor(kwargs[item][key], dtype=torch.float))
                # orb.params.append(kwargs[item][key])
            setattr(dme, item, kwargs[item])



def set_tb_params_bond_length(orb, **kwargs):
    """Initialize a set of the user-defined tight-binding parameters.
    Parameters
    ----------
    **kwargs :

    Returns
    -------

    """
    for item in kwargs:
        if item.startswith('BL'):
            for key in kwargs[item]:
                kwargs[item][key] = torch.tensor(kwargs[item][key], dtype=torch.float)
                # kwargs[item][key] = nn.Parameter(torch.tensor(kwargs[item][key], dtype=torch.float))
                # orb.params.append(kwargs[item][key])
            setattr(dme, item, kwargs[item])




def initializer(**kwargs):
    """Creates a Hamiltonian object from a set of parameters stored in a Python dictionary.

    This functions is used by CLI scripts to create Hamiltonian objects
    from a configuration file (normally in a yaml format) which is previously
    parsed into a Python dictionary data structure.
    Parameters
    ----------
    kwargs : dict
        Dictionary of parameters needed to make a Hamiltonian object.
    **kwargs :

    Returns
    -------

    """
    set_tb_params(**kwargs)
    Orbitals.orbital_sets = kwargs.get('orbital_sets', {'Si': 'SiliconSP3D5S', 'H': 'HydrogenS'})
    sys.modules[__name__].VERBOSITY = kwargs.get('VERBOSITY', 1)

    xyz = kwargs.get('xyz', {})
    nn_distance = kwargs.get('nn_distance', 2.7)
    sparse = kwargs.get('sparse', 0)
    sigma = kwargs.get('sigma', 1.1)
    num_eigs = kwargs.get('num_eigs', 14)

    if sparse:
        h = HamiltonianSp(xyz=xyz, nn_distance=nn_distance, sigma=sigma, num_eigs=num_eigs)
    else:
        h = Hamiltonian(xyz=xyz, nn_distance=nn_distance)

    h.initialize()

    primitive_cell = kwargs.get('primitive_cell', [0, 0, 0])

    if np.sum(np.abs(np.array(primitive_cell))) > 0:
        h.set_periodic_bc(primitive_cell=primitive_cell)

    return h