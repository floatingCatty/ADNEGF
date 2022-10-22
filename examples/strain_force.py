import numpy
import torch

from calc.NEGF import NEGF

from ase.build.ribbon import graphene_nanoribbon
from ase.visualize.plot import plot_atoms
import seaborn as sns
from TB import *
import matplotlib.pyplot as plt

d0 = 1.42 # bond length of C-C


def radial_dep(norm, d0=d0):
    """
        Bound length dependence
    """
    return torch.exp(-3.37*(norm/d0 - 1))



def sorting(coords, **kwargs):
    return np.argsort(coords[:, 1], kind='mergesort')

def create_graphene_nanoribbon(w=3.5, l=5):
    atoms = graphene_nanoribbon(w, l, type='armchair', saturated=True, C_C=d0)

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

    return period, coords

def rebuild_hamiltonian(coords, period, require_gradXYZ=True):
    s_orb = Orbitals('C')
    s_orb.add_orbital("pz", energy=-0.28, orbital=1, magnetic=0, spin=0)

    gamma0 = -2.97
    gamma1 = -0.073
    gamma2 = -0.33
    s0 = 0.073
    s1 = 0.018
    s2 = 0.026

    set_tb_params(PARAMS_C_C1={'pp_pi': gamma0},
                  PARAMS_C_C2={'pp_pi': gamma1},
                  PARAMS_C_C3={'pp_pi': gamma2},
                  OV_C_C1={'pp_pi': s0},
                  OV_C_C2={'pp_pi': s1},
                  OV_C_C3={'pp_pi': s2})

    h = Hamiltonian(xyz=coords, nn_distance=[1.5, 2.5, 3.1], comp_overlap=True, sort_func=sorting).initialize()
    h.set_periodic_bc(period)

    hL, hD, hR, sL, sD, sR = h.get_hamiltonians()
    hl_list, hd_list, hr_list, sl_list, sd_list, sr_list, subblocks = \
        h.get_hamiltonians_block_tridiagonal(optimized=True)

    def pack(**options):
        return options

    hmt_ovp = pack(hd=hd_list,hu=hr_list,hl=hl_list,sd=sd_list,su=sr_list,sl=sl_list,lhd=hD,lhu=hL.conj().T,lsd=sD,lsu=sL.conj().T,rhd=hD,rhu=hR,rsd=sD,rsu=sR)
    negf = NEGF(hmt_ovp)

    if require_gradXYZ:
        for i in list(h.atom_list.keys()):
            h.atom_list[i].requires_grad_()

    return negf, h

def recreate_GNR(coord):
    coords = []
    coords.append(str(len(coord)))
    coords.append('Nanoribbon')
    for j, item in enumerate(coord):
        coords.append('C' + str(j + 1) + ' ' + str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]))

    coords = '\n'.join(coords)

    return coords

def refresh_h_negf(h):
    h.initialize()
    h.set_periodic_bc(period)

    hL, hD, hR, sL, sD, sR = h.get_hamiltonians()
    hl_list, hd_list, hr_list, sl_list, sd_list, sr_list, subblocks = \
        h.get_hamiltonians_block_tridiagonal(optimized=True)

    def pack(**options):
        return options

    hmt_ovp = pack(hd=hd_list, hu=hr_list, hl=hl_list, sd=sd_list, su=sr_list, sl=sl_list, lhd=hD, lhu=hL.conj().T,
                   lsd=sD, lsu=sL.conj().T, rhd=hD, rhu=hR, rsd=sD, rsu=sR)
    negf = NEGF(hmt_ovp)

    return negf, h

def plotPos(coord):

    plt.scatter(x=coord[:,0], y=coord[:,1])
    plt.xlim((-2,4))
    plt.ylim((-1,12))
    plt.show()



if __name__ == '__main__':
    ur = 1.2
    step = 5

    s_orb = Orbitals('C')
    s_orb.add_orbital("pz", energy=-0.28, orbital=1, magnetic=0, spin=0)

    gamma0 = -2.97
    gamma1 = -0.073
    gamma2 = -0.33
    s0 = 0.073
    s1 = 0.018
    s2 = 0.026

    set_tb_params(PARAMS_C_C1={'pp_pi': gamma0},
                  PARAMS_C_C2={'pp_pi': gamma1},
                  PARAMS_C_C3={'pp_pi': gamma2},
                  OV_C_C1={'pp_pi': s0},
                  OV_C_C2={'pp_pi': s1},
                  OV_C_C3={'pp_pi': s2})


    # Current = []
    # for i in range(10):
    #     I = negf.calCurrent_NUM(ul=0, ur=i*0.3, atom_coord=h.get_site_coordinates()[h._offsets],
    #                             d_trains=1,
    #                             left_pos=period[0][0],
    #                             right_pos=period[0][1],
    #                             offset=h._offsets,
    #                             etaLead=1e-5,
    #                             etaDevice=0.,
    #                             ifSCF=True,
    #                             n_int_neq=100,
    #                             cutoff=True)
    #     Current.append(I)
    #
    # Current = torch.stack(Current)
    # plt.plot([i * 0.3 for i in range(10)], Current.detach())
    # plt.show()

    period, coords = create_graphene_nanoribbon(w=1.5, l=3)
    h = Hamiltonian(xyz=coords, nn_distance=[1.5, 2.5, 3.1], comp_overlap=True, sort_func=sorting, radial_dep=radial_dep)
    for i in range(step):
        negf, h = refresh_h_negf(h)
        for i in list(h.atom_list.keys()):
            h.atom_list[i].requires_grad_()

        # modify the Current
        I = negf.calCurrent_NUM(ul=0, ur=ur, atom_coord=h.get_site_coordinates()[h._offsets],
                                d_trains=1,
                                left_pos=period[0][0],
                                right_pos=period[0][1],
                                offset=h._offsets,
                                etaLead=1e-5,
                                etaDevice=0.,
                                ifSCF=True,
                                n_int_neq=100,
                                cutoff=True)
        print(I)
        atom_ids = list(h.atom_list.keys())

        XYZ_grad = torch.autograd.grad(I, list(h.atom_list.values()))

        # update h
        for i in range(len(atom_ids)):
            h.atom_list[atom_ids[i]] = h.atom_list[atom_ids[i]].detach() + XYZ_grad[i]
        atom_Coord = torch.stack(list(h.atom_list.values())).detach()
        plotPos(atom_Coord)
        period = [[0, torch.max(atom_Coord[:,1]), 0]]



    torch.save(obj=(atom_ids, atom_Coord), f="strain_force.pth")



