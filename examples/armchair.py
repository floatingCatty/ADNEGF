import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import torch
from calc.SCF import SCF_with_hTB
from calc.transport import *
from TB import *
from TB.block_tridiagonalization import cut_in_blocks

def sorting(coords, **kwargs):
    return np.argsort(coords[:, 1], kind='mergesort')

def graphene_nanoribbons_armchair():

    from ase.build.ribbon import graphene_nanoribbon
    from ase.visualize.plot import plot_atoms

    atoms = graphene_nanoribbon(1.5, 3, type='armchair', saturated=True)

    period = np.array([list(atoms.get_cell()[2])])
    period[:, [1, 2]] = period[:, [2, 1]]
    coord = atoms.get_positions()

    coord[:, [1, 2]] = coord[:, [2, 1]]
    coords = []
    coords.append(str(len(coord)))
    coords.append('Nanoribbon')

    for j, item in enumerate(coord):
        coords.append('C' + str(j+1) + ' ' + str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]))

    coords = '\n'.join(coords)


    s_orb = Orbitals('C')
    s_orb.add_orbital("pz", energy=-0.28, orbital=1, magnetic=0, spin=0)

    # ------------------------ set TB parameters----------------------
    # gamma0 = -2.78
    # gamma1 = -0.15
    # gamma2 = -0.095
    # s0 = 0.117
    # s1 = 0.004
    # s2 = 0.002
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

    # --------------------------- Hamiltonian -------------------------

    h = Hamiltonian(xyz=coords, nn_distance=[1.5, 2.5, 3.1], comp_overlap=True, sort_func=sorting).initialize()
    h.set_periodic_bc(period)

    hL, hD, hR, sL, sD, sR = h.get_hamiltonians()
    hl_list, hd_list, hr_list, sl_list, sd_list, sr_list, subblocks = \
        h.get_hamiltonians_block_tridiagonal(optimized=True)

    k_points = np.linspace(0.0, np.pi / period[0][1], 20)
    band_structure = torch.zeros((len(k_points), h.h_matrix.shape[0]))

    for jj, item in enumerate(k_points):
        band_structure[jj, :], _ = h.diagonalize_periodic_bc([0.0, item, 0.0])

    # visualize
    ax = plt.axes()
    ax.set_title('Graphene nanoribbon, armchair 11')
    ax.set_ylabel('Energy (eV)')
    ax.set_xlabel(r'Wave vector ($\frac{\pi}{a}$)')
    ax.plot(k_points, np.sort(band_structure.detach().numpy()), 'k')
    ax.xaxis.grid()
    plt.show()

    ax1 = plot_atoms(atoms, show_unit_cell=2, rotation='90x,0y,270z')
    ax1.axis('off')
    ax1.set_xlabel("graphene armchair")
    plt.show()


    ul, ur = 0, 0.5
    el, er = -5, 5
    n = 200
    zs = 0
    zd = period[0][1]
    dtype = torch.float32

    V_ext = SCF_with_hTB(
        hamiltonian=h,
        n_img=200,
        err=1e-5,
        maxIter=500,
        zs=zs,
        zd=zd,
        d_trans=1,
        Emin=-20,
        ul=ul,
        ur=ur,
        method='PDIIS'
    )
    #

    transmission, seebeck, seebeckFD, TTASE = TT_with_hTB(hamiltonian=h, V_ext=V_ext, n=n, el=el, er=er, ul=ul, ur=ur, ifASE=True, ifFD=True, ifseebeck=True, fd_step=torch.finfo(dtype).eps, dtype=dtype)



    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    x = torch.linspace(start=el, end=er, steps=n)
    ax1.plot(x, transmission.detach().numpy())
    ax1.set_title("transmission & seebeck under bias voltage of {0}~{1}".format(ul, ur))
    ax1.set_xlabel("E/ev")
    ax1.set_ylabel("T(E)")
    ax1.set_xlim((el, er))

    ax2 = ax1.twinx()

    ax2.plot(x, seebeck.detach().numpy(), 'r', linestyle='--')
    ax2.set_ylabel("S(E): dT/dE")
    ax2.set_ylim((-50, 50))
    plt.show()

    plt.plot(x, seebeck.detach(), 'b')
    plt.plot(x, seebeckFD.detach(), 'r', linestyle='--')
    plt.legend(["seebeck(AD)","seebeck(FD)"])
    plt.xlabel("E/ev")
    plt.ylabel("S(E):dT/dE")
    plt.title("seebeck with AD & FD under bias voltage of {0}~{1}".format(ul, ur))
    plt.xlim((el,er))
    plt.ylim((-50,50))
    plt.show()

    plt.plot(x, TTASE)
    plt.plot(x, transmission.detach().numpy(), linestyle='--')
    plt.xlabel("E/ev")
    plt.ylabel("T(E)")
    plt.title("transmission of graphene nanoribbons")
    plt.legend(["ASE","abNEGF"])
    plt.xlim((el, er))
    plt.show()








if __name__ == '__main__':
    graphene_nanoribbons_armchair()