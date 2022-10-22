import numpy as np
import matplotlib.pyplot as plt
from TB.hamiltonian import Hamiltonian
from TB.orbitals import Orbitals
from calc import NEGF
from torch.autograd.functional import hessian
import torch
import seaborn as sns
from TB.hamiltonian_initializer import set_tb_params, set_tb_params_bond_length
from TB.aux_functions import get_k_coords
from tqdm import tqdm
from torch.optim import Adam, SGD, RMSprop, LBFGS

from ase.build.ribbon import graphene_nanoribbon
from ase.visualize.plot import plot_atoms

if __name__ == '__main__':
    atoms = graphene_nanoribbon(1.5, 1, type='armchair', saturated=False)
    ax1 = plot_atoms(atoms, show_unit_cell=2, rotation='90x,0y,270z')
    plt.tight_layout()
    plt.show()

    period = np.array([list(atoms.get_cell()[2])])
    period[:, [1, 2]] = period[:, [2, 1]]
    coord = atoms.get_positions()

    coord[:, [1, 2]] = coord[:, [2, 1]]
    coords = []
    coords.append(str(len(coord)))
    coords.append('Nanoribbon')

    for j, item in enumerate(coord):
        coords.append('C' + str(j + 1) + ' ' + str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]))

    coords = '\n'.join(coords)
    print(coords)

    s_orb = Orbitals('C')
    s_orb.add_orbital("pz", energy=-0.28, orbital=1, magnetic=0, spin=0)

    gamma0 = -2.97
    gamma1 = -0.073
    gamma2 = -0.33
    s0 = 0.073
    s1 = 0.018
    s2 = 0.026

    set_tb_params(s_orb, PARAMS_C_C1={'pp_pi': gamma0},
                  PARAMS_C_C2={'pp_pi': gamma1},
                  PARAMS_C_C3={'pp_pi': gamma2},
                  OV_C_C1={'pp_pi': s0},
                  OV_C_C2={'pp_pi': s1},
                  OV_C_C3={'pp_pi': s2})

    c_c1 = 1.42
    c_c2 = 2.45951214
    c_c3 = 2.84
    c_c1_pow = 3.84
    c_c2_pow = 3.84
    c_c3_pow = 3.84

    set_tb_params_bond_length(s_orb, BL_C_C1={'bl': c_c1, 'pp_pi': c_c1_pow},
                              BL_C_C2={'bl': c_c2, 'pp_pi': c_c2_pow},
                              BL_C_C3={'bl': c_c3, 'pp_pi': c_c3_pow}
                              )


    def radial_dependence_func(bond_length, ne_bond_length, param):
        return torch.exp(-param * (ne_bond_length / bond_length - 1))


    def sorting(coords, **kwargs):
        return np.argsort(coords[:, 1], kind='mergesort')


    h = Hamiltonian(radial_dep=radial_dependence_func,xyz=coords, xyz_new=coords, nn_distance=[1.6, 2.70, 3.2], sort_func=sorting, comp_overlap=True, period=period)


    def pack(**options):
        return options


    x = torch.stack([h.atom_list[k] for k in h.atom_list.keys()])
    # with torch.enable_grad():
    #     plt.matshow(torch.autograd.grad(func(x, e), x)[0].detach())
    #     plt.show()

    for e in torch.linspace(0, 1, steps=50):
        def func(x):
            for i, k in enumerate(h.atom_list.keys()):
                h.atom_list[k] = x[i]

            h.initialize()
            h.set_periodic_bc(period)
            hL, hD, hR, sL, sD, sR = h.get_hamiltonians()
            hl_list, hd_list, hr_list, sl_list, sd_list, sr_list, subblocks = \
                h.get_hamiltonians_block_tridiagonal(optimized=True)
            hmt_ovp = pack(hd=hd_list,
                           hu=hr_list,
                           hl=hl_list,
                           sd=sd_list,
                           su=sr_list,
                           sl=sl_list,

                           lhd=hD,
                           lhu=hL.conj().T,
                           lsd=sD,
                           lsu=sL.conj().T,

                           rhd=hD,
                           rhu=hR,
                           rsd=sD,
                           rsu=sR)

            negf = NEGF(hmt_ovp, dtype=torch.complex128)

            TT = negf.calGreen(
                ee=e,
                ul=0,
                ur=0,
                atom_coord=h.get_site_coordinates()[h._offsets],
                d_trains=1,
                left_pos=period[0][0],
                right_pos=period[0][1],
                offset=h._offsets,
                calDOS=True,
                calTT=True,
                calSeebeck=False,
                etaLead=1e-5,
                etaDevice=0.,
                ifSCF=True,
                n_int_neq=100,
                cutoff=False,
                sgfMethod='Lopez-Schro'
            )['TT']

            return TT

        torch.autograd.gradgradcheck(func, x, eps=1e-6)


