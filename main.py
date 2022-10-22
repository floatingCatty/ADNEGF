import ase
from examples.phonon_graphene import *
from transport import Transport_tb
import torch
import os
import sys

def plot_dispo_atoms(atoms, dispo):
    new_atoms = atoms.copy()
    new_atoms.set_positions(newpositions=new_atoms.get_positions()+dispo[:,[0,2,1]])
    plot_atoms(new_atoms, show_unit_cell=2, rotation='90x,0y,270z')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    eps = 0.02
    avg_times = 100
    ul = 0
    ur = 1

    ee = torch.linspace(-3, 3, 50)
    coords, period, atoms = init_syst()

    plot_atoms(atoms, show_unit_cell=2, rotation='90x,0y,270z')
    plt.tight_layout()
    plt.show()

    transport_tb = Transport_tb(
        radial_dep=radial_dependence_func, xyz=coords, xyz_new=coords,
        period=period, nn_distance=[1.65, 2.65, 3.35], comp_overlap=True,
        sort_func=sorting
    )

    # compute hessian matrix dict
    hess = get_heission(ul=ul, ur=ur, transport_tb=transport_tb, ee=ee, period=period)

    # TT_eq = getTT(ul=ul, ur=ur, transport_tb=transport_tb, ee=ee, period=period)
    # TT_dispo_list = []
    # TT_approx_list = []
    # dispo_list = []
    #
    # if not os.path.exists("./examples/experimental_data/phonon_graphene_data.pth"):
    #     for tt in range(30):
    #         dispo = transport_tb.fluctuate(eps)
    #         TT_dispo_list.append(getTT(ul, ur, transport_tb, ee, period))
    #         TT_approx_list.append(getTT_approx(hess, dispo, TT_eq))
    #         plot_dispo_atoms(atoms, dispo=dispo.numpy())
    #         visualize(ee, TT_eq=TT_eq, TT_approx=TT_approx_list[tt], TT=TT_dispo_list[tt])
    #
    #         dispo_list.append(dispo)
    #     torch.save({'TT_dispo': TT_dispo_list, 'TT_approx': TT_approx_list, 'dispo': dispo_list, 'TT_eq': TT_eq},
    #                "./examples/experimental_data/phonon_graphene_data.pth")
    #
    # else:
    #     f = torch.load("./examples/experimental_data/phonon_graphene_data.pth")
    #     TT_dispo_list, TT_approx_list, TT_eq, dispo_list = f['TT_dispo'], f['TT_approx'], f['TT_eq'], f['dispo']
    #     for tt in range(30):
    #         plot_dispo_atoms(atoms, dispo=dispo_list[tt].numpy())
    #         visualize(ee, TT_eq=TT_eq, TT_approx=TT_approx_list[tt], TT=TT_dispo_list[tt])
    #
    # TT = torch.stack(TT_dispo_list).mean(dim=0)
    # TT_approx = torch.stack(TT_approx_list).mean(dim=0)
    #
    # visualize(ee, TT_eq = TT_eq, TT_approx=TT_approx, TT=TT)

    # ur_list = []
    # I_list = []
    # I_approx_list = []
    # I_real_list = []
    # for ur in tqdm(range(5), desc="Computing Current: "):
    #     I, I_approx, I_real = get_current(transport_tb=transport_tb, ul=0, ur=ur*0.15, n_int=50, eps=eps,
    #         atom_coord=transport_tb.h.get_site_coordinates()[transport_tb.h._offsets],
    #         d_trains=1,
    #         left_pos=period[0][0],
    #         right_pos=period[0][1],
    #         period=period,
    #         offset=transport_tb.h._offsets,
    #         calDOS=True,
    #         calTT=True,
    #         calSeebeck=False,
    #         etaLead=1e-5,
    #         etaDevice=0.,
    #         ifSCF=False,
    #         n_int_neq=100,
    #         cutoff=True,
    #         sgfMethod='Lopez-Schro')
    #     ur_list.append(ur*0.15)
    #     I_list.append(I)
    #     I_approx_list.append(I_approx)
    #     I_real_list.append(I_real)
    #
    # I_list = torch.stack(I_list).detach()
    # I_approx_list = torch.stack(I_approx_list).detach()
    # I_real_list = torch.stack(I_real_list).detach()
    #
    # plt.plot(ur_list, I_list, label="eq")
    # plt.plot(ur_list, I_approx_list, label="approx_dispo")
    # plt.plot(ur_list, I_real_list, label="real_dispo")
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()

