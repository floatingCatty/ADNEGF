import ase
import torch
from ase.lattice.cubic import FaceCenteredCubic
from ase.visualize.plot import plot_atoms
import matplotlib.pyplot as plt
import numpy as np
from TB.orbitals import Orbitals
from TB.hamiltonian_initializer import set_tb_params, set_tb_params_bond_length
from transport import Transport_tb

if __name__ == '__main__':
    atoms = FaceCenteredCubic(directions=[[1, -1, 0], [1, 1, -2], [1, 1, 1]],
                              size=(3, 3, 3), symbol='Cu')
    print(atoms.get_cell())

    ax1 = plot_atoms(atoms, show_unit_cell=2, rotation='90x,0y,270z')
    plt.tight_layout()
    plt.show()

    period = np.array([list(atoms.get_cell()[2])])

    period[:, [1, 2]] = period[:, [2, 1]]
    coord = atoms.get_positions()

    coord[:, [1, 2]] = coord[:, [2, 1]]
    coords = []
    coords.append(str(len(coord)))
    coords.append('Copper')

    print(period)



    for j, item in enumerate(coord):
        coords.append('Cu' + str(j + 1) + ' ' + str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]))

    coords = '\n'.join(coords)
    print(coords)

    s_orb = Orbitals('Cu')
    s_orb.add_orbital("s", energy=-20.14, spin=0)
    s_orb.add_orbital("px", energy=100.00, orbital=1, magnetic=-1, spin=0)
    s_orb.add_orbital("py", energy=100.00, orbital=1, magnetic=1, spin=0)
    s_orb.add_orbital("pz", energy=100.00, orbital=1, magnetic=0, spin=0)
    s_orb.add_orbital("dz2", energy=-20.14, orbital=2, magnetic=-1, spin=0)
    s_orb.add_orbital("dxz", energy=-20.14, orbital=2, magnetic=-2, spin=0)
    s_orb.add_orbital("dyz", energy=-20.14, orbital=2, magnetic=2, spin=0)
    s_orb.add_orbital("dxy", energy=-20.14, orbital=2, magnetic=1, spin=0)
    s_orb.add_orbital("dx2my2", energy=-20.14, orbital=2, magnetic=0, spin=0)

    set_tb_params(
        s_orb,
        PARAMS_CU_CU={
            'ss_sigma':-0.48,
            'sp_sigma':1.84,
            'pp_sigma':3.24,
            'pp_pi':-0.81,
            'sd_sigma':-3.16,
            'pd_sigma':-2.95,
            'pd_pi':1.36,
            'dd_sigma':-16.20,
            'dd_pi':8.75,
            'dd_delta':0.00
        })
    # set_tb_params_bond_length(s_orb, BL_C_C1={'bl': c_c1, 'pp_pi': c_c1_pow})

    def radial_dependence_func(bond_length, ne_bond_length, param):
        return torch.exp(-param*(ne_bond_length/bond_length-1))

    def sorting(coords, **kwargs):
        return np.argsort(coords[:, 1], kind='mergesort')

    transport_tb = Transport_tb(comp_overlap=False, radial_dep=None, xyz=coords, xyz_new=coords, nn_distance=2.6, sort_func=sorting, period=period)
    transport_tb.refresh()
    ee = torch.linspace(-3, 3, 50)
    TT = transport_tb.negf.calGreen(
        ee=ee,
        ul=0,
        ur=0,
        atom_coord=transport_tb.h.get_site_coordinates()[transport_tb.h._offsets],
        d_trains=1,
        left_pos=period[0][0],
        right_pos=period[0][1],
        offset=transport_tb.h._offsets,
        calDOS=False,
        calTT=True,
        calSeebeck=False,
        etaLead=1e-5,
        etaDevice=0.,
        ifSCF=True,
        n_int_neq=100,
        cutoff=False,
        sgfMethod='Lopez-Schro'
    )['TT']

    plt.plot(ee.detach(), TT.detach())
    plt.show()