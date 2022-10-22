import ase
import torch
from transport import Transport_tb
from TB.hamiltonian import Hamiltonian
from torch.autograd.functional import hessian
from Constant import *
from ase.lattice.cubic import FaceCenteredCubic
from ase.visualize.plot import plot_atoms
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import numpy as np
import os
from TB.orbitals import Orbitals
from TB.hamiltonian_initializer import set_tb_params, set_tb_params_bond_length

kBT = k * T / eV

def init_syst():
    atoms = FaceCenteredCubic(directions=[[1, -1, 0], [1, 1, -2], [1, 1, 1]],
                              size=(2, 1, 4), symbol='Au', latticeconstant=4.0)

    plot_atoms(atoms, show_unit_cell=2, rotation='90x,0y,270z')
    plt.tight_layout()
    plt.show()

    period = np.array([list(atoms.get_cell()[2])])
    period[:, [1, 2]] = period[:, [2, 1]]
    coord = atoms.get_positions()

    coord[:, [1, 2]] = coord[:, [2, 1]]
    coords = []
    coords.append(str(len(coord)))
    coords.append('Au')

    for j, item in enumerate(coord):
        coords.append('Au' + str(j + 1) + ' ' + str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]))

    coords = '\n'.join(coords)
    print(coords)

    s_orb = Orbitals('Au')
    s_orb.add_orbital("s", energy=0.7811379, spin=0)
    # s_orb.add_orbital("px", energy=7.529375, orbital=1, magnetic=-1, spin=0)
    # s_orb.add_orbital("py", energy=7.529375, orbital=1, magnetic=1, spin=0)
    # s_orb.add_orbital("pz", energy=7.529375, orbital=1, magnetic=0, spin=0)

    set_tb_params(
        s_orb,
        PARAMS_AU_AU1={
            'ss_sigma':-0.8957304,
            'sp_sigma':1.315182,
            'pp_sigma':1.648105,
            'pp_pi':-0.5064959
        })


    c_c1 = 2.8284
    c_c2 = 2.45951214
    c_c3 = 2.84
    c_c1_pow = 2.2
    c_c2_pow = 3.84
    c_c3_pow = 3.84

    set_tb_params_bond_length(s_orb, BL_AU_AU1={'bl': c_c1, 'ss_sigma': c_c1_pow})

    return coords, period, atoms

def plot_band(atoms=None):

    from ase.visualize.plot import plot_atoms

    if atoms is None:
        atoms = FaceCenteredCubic(directions=[[1, -1, 0], [1, 1, -2], [1, 1, 1]],
                                  size=(2, 5, 1), symbol='Au', latticeconstant=4.0)
        atoms.wrap()

    period = np.array([list(atoms.get_cell()[2])])
    period[:, [1, 2]] = period[:, [2, 1]]
    coord = atoms.get_positions()

    coord[:, [1, 2]] = coord[:, [2, 1]]
    coords = []
    coords.append(str(len(coord)))
    coords.append('Au')


    for j, item in enumerate(coord):
        coords.append('Au' + str(j + 1) + ' ' + str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]))

    coords = '\n'.join(coords)
    print(coords)

    # --------------------------- Basis set --------------------------

    s_orb = Orbitals('Au')
    s_orb.add_orbital("s", energy=0.7811379, spin=0)
    # s_orb.add_orbital("px", energy=7.529375, orbital=1, magnetic=-1, spin=0)
    # s_orb.add_orbital("py", energy=7.529375, orbital=1, magnetic=1, spin=0)
    # s_orb.add_orbital("pz", energy=7.529375, orbital=1, magnetic=0, spin=0)

    set_tb_params(
        s_orb,
        PARAMS_AU_AU1={
            'ss_sigma': -0.8957304,
            'sp_sigma': 1.315182,
            'pp_sigma': 1.648105,
            'pp_pi': -0.5064959
        })

    c_c1 = 2.8284
    c_c2 = 2.45951214
    c_c3 = 2.84
    c_c1_pow = 2.2
    c_c2_pow = 3.84
    c_c3_pow = 3.84

    set_tb_params_bond_length(s_orb, BL_AU_AU1={'bl': c_c1, 'ss_sigma': c_c1_pow})

    # --------------------------- Hamiltonian -------------------------

    h = Hamiltonian(xyz=coords, xyz_new=coords, nn_distance=[3.], comp_overlap=False)
    h.initialize()
    h.set_periodic_bc(period)

    k_points = np.linspace(0.0, np.pi/period[0][1], 20)
    band_structure = torch.zeros((len(k_points), h.h_matrix.shape[0]))

    for jj, item in enumerate(k_points):
        band_structure[jj, :], _ = h.diagonalize_periodic_bc([0.0, item, 0.0])

    # visualize
    ax = plt.axes()
    ax.set_title('Band structure of Au nanoribbon')
    ax.set_ylabel('Energy (eV)')
    ax.set_xlabel(r'Wave vector ($\frac{\pi}{a}$)')
    ax.plot(k_points, np.sort(band_structure.detach().numpy()), 'k')
    ax.xaxis.grid()
    plt.show()

    ax1 = plot_atoms(atoms, show_unit_cell=2, rotation='10x,50y,30z')
    ax1.axis('off')
    plt.show()

def get_heission(ul, ur, transport_tb, ee, period):
    flag = 0
    if os.path.exists("../examples/experimental_data/au_hess.pth"):
        f = torch.load("../examples/experimental_data/au_hess.pth")
        hess = None
        for i in f:
            if abs(ur - i) < 1e-6:
                hess = f.get(i, None)
        try:
            flag = 1
            assert hess is not None
            assert len(hess) == ee.shape[0]
        except AssertionError:
            flag = 0

    if flag == 0:
        x = torch.stack([transport_tb.h.atom_list[k] for k in transport_tb.h.atom_list.keys()])
        hess = []

        def func(x, e):
            for i, k in enumerate(transport_tb.h.atom_list.keys()):
                transport_tb.h.atom_list[k] = x[i]

            transport_tb.refresh()
            TT = transport_tb.negf.calGreen(
                ee=e,
                ul=ul,
                ur=ur,
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
                ifSCF=False,
                n_int_neq=100,
                cutoff=False,
                sgfMethod='Lopez-Schro'
            )['TT'][0]



            return TT

        for ie in tqdm(ee, desc="Computing Hessian: "):
            fn = lambda x: func(x, e=ie)
            hess.append(hessian(fn, x))


    if os.path.exists("../examples/experimental_data/au_hess.pth"):
        f = torch.load("../examples/experimental_data/au_hess.pth")
        f.update({ur: hess})
    else:
        f = {ur:hess}
    torch.save(f, "../examples/experimental_data/au_hess.pth")

    return hess

def getTT(ul, ur, transport_tb, ee, period):
    if isinstance(ul, (float, int)):
        ul = torch.scalar_tensor(ul)
    if isinstance(ur, (float, int)):
        ur = torch.scalar_tensor(ur)
    return transport_tb.negf.calGreen(
        ee=ee,
        ul=ul,
        ur=ur,
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
        ifSCF=False,
        n_int_neq=100,
        cutoff=True,
        sgfMethod='Lopez-Schro'
    )['TT']

def getTT_approx(hess, dispo, TT_eq):
    hess = torch.stack(hess)
    # print(0.5 * (torch.einsum('ij,nijlm->nlm', dispo.type_as(hess), hess)*dispo).sum(dim=(1,2)))
    temp_TT = 0.5 * (torch.einsum('ij,nijlm->nlm', dispo.type_as(hess), hess)*dispo).sum(dim=(1,2)) + TT_eq

    return temp_TT

def get_current(transport_tb, ul, ur, n_int=100, delta=1., eps=0, **Options):
    if isinstance(ul, (float, int)):
        ul = torch.scalar_tensor(ul)
    if isinstance(ur, (float, int)):
        ur = torch.scalar_tensor(ur)


    xu = max(ul, ur) + delta
    xl = min(ul, ur) - delta

    xlg, wlg = np.polynomial.legendre.leggauss(n_int)
    ndim = len(xu.shape)
    xlg = torch.tensor(xlg, dtype=xu.dtype, device=xu.device)[(...,) + (None,) * ndim]  # (n, *nx)
    wlg = torch.tensor(wlg, dtype=xu.dtype, device=xu.device)[(...,) + (None,) * ndim]  # (n, *nx)
    wlg *= 0.5 * (xu - xl)
    xs = xlg * (0.5 * (xu - xl)) + (0.5 * (xu + xl))
    TT = transport_tb.negf.calGreen(ee=xs, ul=ul, ur=ur, **Options)['TT']

    hess = get_heission(ul, ur, transport_tb, xs, period=Options.get("period"))
    dispo = transport_tb.fluctuate(eps=eps)
    TT_approx = getTT_approx(hess=hess, dispo=dispo, TT_eq=TT)


    TT_real = transport_tb.negf.calGreen(ee=xs, ul=ul, ur=ur, **Options)['TT']
    DOS_real = transport_tb.negf.calGreen(ee=xs, ul=ul, ur=ur, **Options)["DOS"]

    plt.plot(DOS_real.detach())
    plt.show()


    for i, t in enumerate(TT):
        TT[i] = (fermi_dirac(xs[i] - xu + 1) - fermi_dirac(xs[i] - xl - 1)) * TT[i]
        TT_approx[i] = (fermi_dirac(xs[i] - xu + 1) - fermi_dirac(xs[i] - xl - 1)) * TT_approx[i]
        TT_real[i] = (fermi_dirac(xs[i] - xu + 1) - fermi_dirac(xs[i] - xl - 1)) * TT_real[i]
    Current = (TT * wlg).sum() / pi
    Current_approx = (TT_approx * wlg).sum() / pi
    Current_real = (TT_real * wlg).sum() / pi

    return Current, Current_approx, Current_real

def fermi_dirac(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(x / kBT))

def visualize(ee, TT_eq, TT_approx, TT, index):
    fig, ax = plt.subplots(1, 1, sharex=True)
    # ax[0].plot(ee, TT.detach().numpy(), c=sns.color_palette("Blues")[4])
    ax.plot(ee, TT.detach().numpy(), c="red", label="negf")
    ax.plot(ee, TT_eq.detach(), c="black", label='Eq')
    ax.plot(ee, TT_approx.detach().numpy(), c='blue', label="ad-negf")
    ax.legend()
    ax.set_ylabel("T(E)")
    ax.set_title("transmission of Au")
    ax.set_xlim((-3, 3))
    # ax.set_ylim((-0.1, 5))

    plt.savefig("./experimental_data/traj/dispo_au_img_"+str(index)+".png", dpi=100)
    plt.show()

    # ax[1].plot(ee, dos.detach().numpy(), c=sns.color_palette("Blues")[4])
    # ax[1].set_xlabel("E/ev")
    # ax[1].set_ylabel("DOS")
    # ax[1].set_title("DOS of AGNR(7)")
    # ax[1].set_xlim((-3, 3))

if __name__ == '__main__':
    # atoms = FaceCenteredCubic(directions=[[1, -1, 0], [1, 1, -2], [1, 1, 1]],
    #                           size=(5, 5, 5), symbol='Cu')
    #
    # ax1 = plot_atoms(atoms, show_unit_cell=2, rotation='90x,0y,270z')
    # plt.tight_layout()
    # plt.show()
    #
    # period = np.array([list(atoms.get_cell()[2])])
    #
    # period[:, [1, 2]] = period[:, [2, 1]]
    # coord = atoms.get_positions()
    #
    # coord[:, [1, 2]] = coord[:, [2, 1]]
    # coords = []
    # coords.append(str(len(coord)))
    # coords.append('Au')
    #
    # print(period)
    #
    #
    #
    # for j, item in enumerate(coord):
    #     coords.append('Au' + str(j + 1) + ' ' + str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]))
    #
    # coords = '\n'.join(coords)
    # print(coords)
    #
    # s_orb = Orbitals('Au')
    # s_orb.add_orbital("s", energy=0.7811379, spin=0)
    # s_orb.add_orbital("px", energy=7.529375, orbital=1, magnetic=-1, spin=0)
    # s_orb.add_orbital("py", energy=7.529375, orbital=1, magnetic=1, spin=0)
    # s_orb.add_orbital("pz", energy=7.529375, orbital=1, magnetic=0, spin=0)
    #
    # set_tb_params(
    #     s_orb,
    #     PARAMS_AU_AU={
    #         'ss_sigma':-0.8957304,
    #         'sp_sigma':1.315182,
    #         'pp_sigma':1.648105,
    #         'pp_pi':-0.5064959
    #     })

    # set_tb_params_bond_length(s_orb, BL_C_C1={'bl': c_c1, 'pp_pi': c_c1_pow})
    #
    def radial_dependence_func(bond_length, ne_bond_length, param):

        return (bond_length / ne_bond_length) ** param
        # return torch.exp(-param*(ne_bond_length/bond_length-1))

    def sorting(coords, **kwargs):
        return np.argsort(coords[:, 1], kind='mergesort')


    def plot_dispo_atoms(atoms, dispo, index):
        new_atoms = atoms.copy()
        new_atoms.set_positions(newpositions=new_atoms.get_positions() + dispo[:, [0, 2, 1]])
        # plot_band(atoms=new_atoms)
        # new_atoms.write("./experimental_data/traj/dispo_au_"+str(index)+".vasp")
        plot_atoms(new_atoms, show_unit_cell=2, rotation='90x,0y,270z')
        plt.tight_layout()
        plt.show()
    #
    # transport_tb = Transport_tb(comp_overlap=False, radial_dep=None, xyz=coords, xyz_new=coords, nn_distance=2.6, sort_func=sorting, period=period)
    # transport_tb.refresh()
    # ee = torch.linspace(-3, 3, 50)
    # out = transport_tb.negf.calGreen(
    #     ee=ee,
    #     ul=0,
    #     ur=0,
    #     atom_coord=transport_tb.h.get_site_coordinates()[transport_tb.h._offsets],
    #     d_trains=1,
    #     left_pos=period[0][0],
    #     right_pos=period[0][1],
    #     offset=transport_tb.h._offsets,
    #     calDOS=True,
    #     calTT=True,
    #     calSeebeck=False,
    #     etaLead=1e-5,
    #     etaDevice=0.,
    #     ifSCF=True,
    #     n_int_neq=100,
    #     cutoff=False,
    #     sgfMethod='Lopez-Schro'
    # )
    # TT = out["TT"]
    # DOS = out["DOS"]
    # plt.plot(ee.detach(), TT.detach())
    # plt.show()
    #
    # plt.plot(ee.detach(), DOS.detach())
    # plt.show()

    # plot_band()

    eps = 0.02
    avg_times = 100
    ul = 0
    ur = 0

    ee = torch.linspace(-3, 3, 50)
    coords, period, atoms = init_syst()

    plot_atoms(atoms, show_unit_cell=2, rotation='90x,0y,270z')
    plt.tight_layout()
    plt.show()

    transport_tb = Transport_tb(
        radial_dep=radial_dependence_func, xyz=coords, xyz_new=coords,
        period=period, nn_distance=[2.9], comp_overlap=False,
        sort_func=sorting
    )

    # TT_eq = getTT(ul=ul, ur=ur, transport_tb=transport_tb, ee=ee, period=period)


    # plt.plot(ee.detach(), TT_eq.detach())
    # plt.title("Transmission TB (2,5)")
    # plt.xlabel("Energy (eV)")
    # plt.ylabel("Transmission (E)")
    # plt.legend(["Au (2x5)"])
    # # plt.ylim([11,31])
    # plt.show()

    # # compute hessian matrix dict
    hess = get_heission(ul=ul, ur=ur, transport_tb=transport_tb, ee=ee, period=period)

    TT_eq = getTT(ul=ul, ur=ur, transport_tb=transport_tb, ee=ee, period=period)



    TT_dispo_list = []
    TT_approx_list = []
    dispo_list = []

    hDD = torch.eye(5) + 0j

    if not os.path.exists("./examples/experimental_data/phonon_graphene_data.pth"):
        for tt in range(30):
            dispo = transport_tb.fluctuate(eps)

            TT_dispo_list.append(getTT(ul, ur, transport_tb, ee, period))
            # print(torch.linalg.eigh(transport_tb.hD.detach())[0])
            # print(torch.linalg.eigh(hDD.real.detach())[0])
            # plt.matshow(transport_tb.hD.real.detach().numpy())
            # plt.show()
            # plt.matshow(hDD.real.detach().numpy())
            # plt.show()
            hDD = transport_tb.hD
            TT_approx_list.append(getTT_approx(hess, dispo, TT_eq))
            plot_dispo_atoms(atoms, dispo=dispo.numpy(), index=tt)
            visualize(ee, TT_eq=TT_eq, TT_approx=TT_approx_list[tt], TT=TT_dispo_list[tt], index=tt)

            dispo_list.append(dispo)
        torch.save({'TT_dispo': TT_dispo_list, 'TT_approx': TT_approx_list, 'dispo': dispo_list, 'TT_eq': TT_eq},
                   "./examples/experimental_data/phonon_graphene_data.pth")

    else:
        f = torch.load("./examples/experimental_data/phonon_graphene_data.pth")
        TT_dispo_list, TT_approx_list, TT_eq, dispo_list = f['TT_dispo'], f['TT_approx'], f['TT_eq'], f['dispo']
        for tt in range(30):
            plot_dispo_atoms(atoms, dispo=dispo_list[tt].numpy())
            visualize(ee, TT_eq=TT_eq, TT_approx=TT_approx_list[tt], TT=TT_dispo_list[tt])

    TT = torch.stack(TT_dispo_list).mean(dim=0)
    TT_approx = torch.stack(TT_approx_list).mean(dim=0)

    visualize(ee, TT_eq = TT_eq, TT_approx=TT_approx, TT=TT)