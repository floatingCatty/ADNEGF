import numpy as np
import matplotlib.pyplot as plt
from TB.orbitals import Orbitals
from torch.autograd.functional import hessian
from transport import Transport_tb
from tqdm import tqdm
from Constant import *
import torch
import os
import sys
from TB.hamiltonian_initializer import set_tb_params, set_tb_params_bond_length
from ase.build.ribbon import graphene_nanoribbon
from ase.visualize.plot import plot_atoms

kBT = k * T / eV

def init_syst():
    atoms = graphene_nanoribbon(1.5, 1, type='armchair', saturated=False)



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

    return coords, period, atoms


def get_heission(ul, ur, transport_tb, ee, period):
    flag = 0
    if os.path.exists("./examples/experimental_data/hess.pth"):
        f = torch.load("./examples/experimental_data/hess.pth")
        hess = None
        for i in f:
            if (ur - i).abs() < 1e-6:
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

            return transport_tb.negf.calGreen(
                ee=e,
                ul=ul,
                ur=ur,
                atom_coord=transport_tb.h.get_site_coordinates()[transport_tb.h._offsets],
                d_trains=1,
                left_pos=period[0][0],
                right_pos=period[0][1],
                offset=transport_tb.h._offsets,
                calDOS=True,
                calTT=True,
                calSeebeck=False,
                etaLead=1e-5,
                etaDevice=0.,
                ifSCF=False,
                n_int_neq=100,
                cutoff=False,
                sgfMethod='Lopez-Schro'
            )['TT']

        for ie in tqdm(ee, desc="Computing Hessian: "):
            fn = lambda x: func(x, e=ie)
            hess.append(hessian(fn, x))
            print(hess[-1])

    if os.path.exists("./examples/experimental_data/hess.pth"):
        f = torch.load("./examples/experimental_data/hess.pth")
        f.update({ur: hess})
    else:
        f = {ur:hess}
    torch.save(f, "./examples/experimental_data/hess.pth")

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
    print(dispo.size())
    print(hess.size())
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

def visualize(ee, TT_eq, TT_approx, TT):
    fig, ax = plt.subplots(1, 1, sharex=True)
    # ax[0].plot(ee, TT.detach().numpy(), c=sns.color_palette("Blues")[4])
    ax.plot(ee, TT.detach().numpy(), c="red", label="negf")
    ax.plot(ee, TT_eq.detach(), c="black", label='Eq')
    ax.plot(ee, TT_approx.detach().numpy(), c='blue', label="ad-negf")
    ax.legend()
    ax.set_ylabel("T(E)")
    ax.set_title("transmission of AGNR(7)")
    ax.set_xlim((-3, 3))
    ax.set_ylim((-0.1, 5))

    plt.show()

    # ax[1].plot(ee, dos.detach().numpy(), c=sns.color_palette("Blues")[4])
    # ax[1].set_xlabel("E/ev")
    # ax[1].set_ylabel("DOS")
    # ax[1].set_title("DOS of AGNR(7)")
    # ax[1].set_xlim((-3, 3))
    plt.show()




def radial_dependence_func(bond_length, ne_bond_length, param):
    return torch.exp(-param*(ne_bond_length/bond_length-1))

def sorting(coords, **kwargs):
    return np.argsort(coords[:, 1], kind='mergesort')
