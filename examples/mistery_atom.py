from calc.NEGF import NEGF
from skopt.space import Real
from bayes_opt import BayesianOptimization
from matplotlib.legend_handler import HandlerTuple
import time
from bayes_opt import UtilityFunction
from torch.optim import Adam, LBFGS, RMSprop
from bayes_opt.logger import JSONLogger
import numpy as np
from calc.utils import geneticalgorithm as ga
from bayes_opt.event import Events
import torch
from skopt import BayesSearchCV
from ase.build.ribbon import graphene_nanoribbon
from ase.visualize.plot import plot_atoms
import seaborn as sns
from TB import *
import matplotlib.pyplot as plt


def sorting(coords, **kwargs):
    return np.argsort(coords[:, 1], kind='mergesort')

def create_graphene_nanoribbon(tuneList, w=3.5, l=5):
    atoms = graphene_nanoribbon(w, l, type='armchair', saturated=True)

    period = np.array([list(atoms.get_cell()[2])])
    period[:, [1, 2]] = period[:, [2, 1]]
    coord = atoms.get_positions()

    coord[:, [1, 2]] = coord[:, [2, 1]]
    coords = []
    coords.append(str(len(coord)))
    coords.append('Nanoribbon')

    for j, item in enumerate(coord):
        if item[0] > 6 and item[0] < 7:
            coords.append('M' + str(j + 1) + ' ' + str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]))
        # else:
        # if j in tuneList:
        #     coords.append('M' + str(j + 1) + ' ' + str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]))
        else:
            coords.append('C' + str(j + 1) + ' ' + str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]))

    coords = '\n'.join(coords)

    return period, coords

def rebuild_hamiltonian(CM, coords, period):
    s_orb = Orbitals('M')
    s_orb.add_orbital("pz", energy=CM[12], orbital=1, magnetic=0, spin=0)
    s_orb = Orbitals('C')
    s_orb.add_orbital("pz", energy=-0.28, orbital=1, magnetic=0, spin=0)

    gamma0 = -2.97
    gamma1 = -0.073
    gamma2 = -0.33
    s0 = 0.073
    s1 = 0.018
    s2 = 0.026

    set_tb_params(s_orb, PARAMS_M_M1={'pp_pi': CM[0]},
                  PARAMS_M_M2={'pp_pi': CM[1]},
                  PARAMS_M_M3={'pp_pi': CM[2]},
                  OV_M_M1={'pp_pi': CM[9]},
                  OV_M_M2={'pp_pi': CM[10]},
                  OV_M_M3={'pp_pi': CM[11]})

    set_tb_params(s_orb, PARAMS_C_M1={'pp_pi': CM[3]},
                  PARAMS_C_M2={'pp_pi': CM[4]},
                  PARAMS_C_M3={'pp_pi': CM[5]},
                  OV_C_M1={'pp_pi': CM[6]},
                  OV_C_M2={'pp_pi': CM[7]},
                  OV_C_M3={'pp_pi': CM[8]})

    set_tb_params(s_orb, PARAMS_C_C1={'pp_pi': gamma0},
                  PARAMS_C_C2={'pp_pi': gamma1},
                  PARAMS_C_C3={'pp_pi': gamma2},
                  OV_C_C1={'pp_pi': s0},
                  OV_C_C2={'pp_pi': s1},
                  OV_C_C3={'pp_pi': s2})

    h = Hamiltonian(xyz=coords, xyz_new=coords, nn_distance=[1.5, 2.5, 3.1], comp_overlap=True, sort_func=sorting).initialize()
    h.set_periodic_bc(period)

    hL, hD, hR, sL, sD, sR = h.get_hamiltonians()
    hl_list, hd_list, hr_list, sl_list, sd_list, sr_list, subblocks = \
        h.get_hamiltonians_block_tridiagonal(optimized=True)

    def pack(**options):
        return options

    hmt_ovp = pack(hd=hd_list,hu=hr_list,hl=hl_list,sd=sd_list,su=sr_list,sl=sl_list,lhd=hD,lhu=hL.conj().T,lsd=sD,lsu=sL.conj().T,rhd=hD,rhu=hR,rsd=sD,rsu=sR)
    # plt.matshow(hD.detach().real, vmin=-5, vmax=0.5)
    # plt.colorbar()
    negf = NEGF(hmt_ovp)

    return negf, h

def opt_fn(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13):
    CM = torch.tensor([a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13], dtype=torch.float64)
    ee = torch.linspace(-1, 1, 50)
    period, coord = create_graphene_nanoribbon(tuneList=[])
    negf, h = rebuild_hamiltonian(CM, coord, period)
    TT = negf.calGreen(ee=ee, ul=0, ur=0, atom_coord=h.get_site_coordinates()[h._offsets],
            d_trains=1,
            left_pos=period[0][0],
            right_pos=period[0][1],
            offset=h._offsets,
            etaLead=1e-5,
            etaDevice=0.,
            ifSCF=True,
            n_int_neq=100,
            cutoff=True, calTT=True, sgfMethod='Lopez-Sancho')['TT']

    return -TT.mean()

def opt_fn_GB(CM):
    ee = torch.linspace(-1, 1, 50)
    period, coord = create_graphene_nanoribbon(tuneList=[])
    negf, h = rebuild_hamiltonian(CM, coord, period)
    TT = negf.calGreen(ee=ee, ul=0, ur=0, atom_coord=h.get_site_coordinates()[h._offsets],
            d_trains=1,
            left_pos=period[0][0],
            right_pos=period[0][1],
            offset=h._offsets,
            etaLead=1e-5,
            etaDevice=0.,
            ifSCF=True,
            n_int_neq=100,
            cutoff=True, calTT=True, sgfMethod='Lopez-Sancho')['TT']

    return TT.mean()

def opt_fn_GO(CM):
    CM = torch.tensor(CM).view(-1)
    ee = torch.linspace(-1, 1, 50)
    period, coord = create_graphene_nanoribbon(tuneList=[])
    negf, h = rebuild_hamiltonian(CM, coord, period)
    TT = negf.calGreen(ee=ee, ul=0, ur=0, atom_coord=h.get_site_coordinates()[h._offsets],
            d_trains=1,
            left_pos=period[0][0],
            right_pos=period[0][1],
            offset=h._offsets,
            etaLead=1e-5,
            etaDevice=0.,
            ifSCF=True,
            n_int_neq=100,
            cutoff=True, calTT=True, sgfMethod='Lopez-Sancho')['TT']

    return TT.mean()

def compute_TT(CM):
    ee = torch.linspace(-3, 3, 200)
    period, coord = create_graphene_nanoribbon(tuneList=[])
    negf, h = rebuild_hamiltonian(CM, coord, period)
    # hd = negf.hmt_ovp['rhd']
    TT = negf.calGreen(ee=ee, ul=0, ur=0, atom_coord=h.get_site_coordinates()[h._offsets],
            d_trains=1,
            left_pos=period[0][0],
            right_pos=period[0][1],
            offset=h._offsets,
            etaLead=1e-5,
            etaDevice=0.,
            ifSCF=True,
            n_int_neq=100,
            cutoff=True, calTT=True, sgfMethod='Lopez-Sancho')['TT']

    return TT
    # return TT,hd

def BO(k, delta=0.5):
    gamma0 = -2.97
    gamma1 = -0.073
    gamma2 = -0.33
    s0 = 0.073
    s1 = 0.018
    s2 = 0.026

    CM = torch.tensor([gamma0, gamma1, gamma2, gamma0, gamma1, gamma2, s0, s1, s2, s0, s1, s2, -0.28],
                      dtype=torch.float64)

    # Baysian search

    param = {
        'a1':(gamma0-delta, gamma0+delta),
        'a2':(gamma1-delta, gamma1+delta),
        'a3': (gamma2 - delta, gamma2 + delta),
        'a4': (gamma0 - delta, gamma0 + delta),
        'a5': (gamma1 - delta, gamma1 + delta),
        'a6': (gamma2 - delta, gamma2 + delta),
        'a7': (s0 - delta, s0 + delta),
        'a8': (s1 - delta, s1 + delta),
        'a9': (s2 - delta, s2 + delta),
        'a10': (s0 - delta, s0 + delta),
        'a11': (s1 - delta, s1 + delta),
        'a12': (s2 - delta, s2 + delta),
        'a13': (-0.28 - delta, -0.28 + delta),
    }
    optimizer = BayesianOptimization(f=None, pbounds=param, random_state=(k+2), verbose=2)
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
    loss = []
    point = []
    timelist = []

    start = time.time()
    for _ in range(100):
        next_point = optimizer.suggest(utility)
        target = opt_fn(**next_point)
        optimizer.register(params=next_point, target=target)
        loss.append(target)
        point.append(next_point)
        timelist.append(time.time()-start)
        print(target, next_point, timelist[-1])
    print(optimizer.max)

    torch.save(obj=(loss, point, timelist), f="./dop_BO"+str(k)+".pth")

    return True



def Genetic(k, delta=0.3):
    gamma0 = -2.97
    gamma1 = -0.073
    gamma2 = -0.33
    s0 = 0.073
    s1 = 0.018
    s2 = 0.026

    bound = np.array([
        [gamma0 - delta, gamma0 + delta],
        [gamma1 - delta, gamma1 + delta],
        [gamma2 - delta, gamma2 + delta],
        [gamma0 - delta, gamma0 + delta],
        [gamma1 - delta, gamma1 + delta],
        [gamma2 - delta, gamma2 + delta],
        [s0 - delta, s0 + delta],
        [s1 - delta, s1 + delta],
        [s2 - delta, s2 + delta],
        [s0 - delta, s0 + delta],
        [s1 - delta, s1 + delta],
        [s2 - delta, s2 + delta],
        [-0.28 - delta, -0.28 + delta]
    ])
    algorithm_param = {'max_num_iteration': None,
                       'population_size': 20,
                       'mutation_probability': 0.1,
                       'elit_ratio': 0.01,
                       'crossover_probability': 0.5,
                       'parents_portion': 0.3,
                       'crossover_type': 'uniform',
                       'max_iteration_without_improv': None}
    model = ga(id=k,function=opt_fn_GO, dimension=13, variable_type='real', variable_boundaries=bound,algorithm_parameters=algorithm_param)
    model.funtimeout = 100000
    model.run()
    report = model.report
    output_dict = model.output_dict

    torch.save(obj=(report, output_dict), f="./dop_GO"+str(k)+".pth")

    return True

def GB(k, delta, method='LBFGS'):
    gamma0 = -2.97
    gamma1 = -0.073
    gamma2 = -0.33
    s0 = 0.073
    s1 = 0.018
    s2 = 0.026

    CM = torch.tensor([gamma0, gamma1, gamma2, gamma0, gamma1, gamma2, s0, s1, s2, s0, s1, s2, -0.28],
                      dtype=torch.float64)
    org_CM = CM.detach()
    CM.requires_grad_()
    if method == 'LBFGS':
        optimizer = LBFGS([CM], max_iter=10, lr=(k+1)*0.02)
        loss = []
        def closure():
            optimizer.zero_grad()
            TT = opt_fn_GB(CM)
            TT.backward()
            print(TT)
            loss.append(TT.detach())
            return TT

        optimizer.step(closure)
    start = time.time()
    timelist = []
    if method == 'RMSprop':
        optimizer = RMSprop([CM], lr=(k+1)*0.005)
        loss = []
        for i in range(100):
            optimizer.zero_grad()
            TT = opt_fn_GB(CM)
            TT.backward()
            print(TT)
            optimizer.step()
            for i in range(len(CM)):
                if (CM[i] - org_CM[i]).abs() > delta:
                    if CM[i] < org_CM[i]: CM[i] = org_CM[i] - delta
                    else: CM[i] = org_CM[i] + delta
            timelist.append(time.time()-start)

            loss.append(TT.detach())
    loss = torch.stack(loss)
    plt.plot(loss)
    plt.show()

    TT = compute_TT(CM)
    plt.plot(TT.detach())
    plt.show()

    torch.save(obj={"CM":CM.detach(), "loss":loss, "time":timelist}, f="./dop_GB_"+method+str(k)+".pth")

    return CM, loss


def plot(name):
    if name == 'loss':

        fig, (ax) = plt.subplots(2, 1, figsize=(6.8, 5.8))

        loss, point, time = torch.load("./dop_BO" + str(0) + ".pth")
        time = torch.tensor(time, dtype=torch.float64)
        loss, point = torch.load("./dop_BO" + str(4) + ".pth")
        loss = torch.tensor(loss)
        ax[0].plot(time, -loss, c='tab:green', ls='-.', lw=2)

        f = torch.load("./dop_GB_RMSprop" + str(0) + ".pth")
        time = torch.tensor(f['time'])
        f = torch.load("./dop_GB_RMSprop" + str(4) + ".pth")
        ax[0].plot(time,f['loss'], c='tab:blue', lw=2)

        f = torch.load("../dop_GA1.pth")
        time = []
        loss = []
        for (report, best_fn, best_var, t) in f:
            time.append(t)
            loss.append(best_fn)
        ax[0].plot(time, loss,'--', c='tab:red', lw=2)

        ax[0].set_xlabel("time (second)", fontsize=14)
        ax[0].set_xlim((0,1200))


        loss, point = torch.load("./dop_BO" + str(4) + ".pth")
        loss = torch.tensor(loss)
        ax[1].plot(-loss, c='tab:green', ls='-.', lw=2)

        f = torch.load("./dop_GB_RMSprop" + str(4) + ".pth")
        ax[1].plot(f['loss'], c='tab:blue', lw=2)

        f = torch.load("../dop_GA1.pth")
        loss = []
        for (report, best_fn, best_var, t) in f:
            loss.append(best_fn)
        ax[1].plot(loss, '--', c='tab:red', lw=2, ls='--')

        plt.legend(['Bayesian Optimization', "AD-NEGF based", "Genetic Algorithm"], fontsize=10)
        ax[1].set_xlabel("step", fontsize=14)
        ax[0].set_ylabel('loss', fontsize=14)
        ax[1].set_ylabel('loss', fontsize=14)
        plt.xlim((0, 100))

        plt.tight_layout()

        plt.subplots_adjust(hspace=0.38)

        # plt.xlim((0, 600))
        plt.savefig("./experimental_data/Dop_loss.pdf", dpi=600)

        plt.show()
    if name == 'TT':
        point_BO = torch.load("./dop_BO" + str(4) + ".pth")
        f = torch.load("./dop_GB_RMSprop" + str(4) + ".pth")
        point_GB = f['CM']
        f = torch.load("../dop_GA1.pth")
        point_GA = []
        for (report, best_fn, best_var, t) in f:
            point_GA.append(best_var)
        point_GA = point_GA[-1]

        point_BO = torch.tensor(point_BO[0])
        point_GA = torch.tensor(point_GA)

        # TT_BO, hd_BO = compute_TT(point_BO)
        # TT_GB, hd_GB = compute_TT(point_GB)
        # TT_GA, hd_GA = compute_TT(point_GA)

        TT_BO = compute_TT(point_BO)
        TT_GB = compute_TT(point_GB)
        TT_GA = compute_TT(point_GA)
        gamma0 = -2.97
        gamma1 = -0.073
        gamma2 = -0.33
        s0 = 0.073
        s1 = 0.018
        s2 = 0.026

        CM = torch.tensor([gamma0, gamma1, gamma2, gamma0, gamma1, gamma2, s0, s1, s2, s0, s1, s2, -0.28],
                          dtype=torch.float64)
        TT_ORG = compute_TT(CM)
        # TT_ORG, hd_ORG = compute_TT(CM)
        # plt.matshow((hd_BO-hd_ORG).detach().real, vmin=-0.3, vmax=0.3)
        # plt.colorbar()
        # plt.matshow((hd_GB - hd_ORG).detach().real, vmin=-0.3, vmax=0.3)
        # plt.colorbar()
        # plt.matshow((hd_GA - hd_ORG).detach().real, vmin=-0.3, vmax=0.3)
        # plt.colorbar()
        ee = torch.linspace(-3, 3, 200)
        fig, (ax) = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(5.6,4.8))

        ax[0].plot(ee, TT_ORG.detach(), c='black', label='ORG')
        ax[0].axline((-1, 0),(-1,1), ls='--',c='black', lw=1.5)
        ax[0].axline((1, 0), (1, 1),ls='--', c='black', lw=1.5)
        ax[0].set_title("Before Doping")
        ax[1].plot(ee, TT_GB.detach(), c='tab:blue', label='GB')
        ax[1].axline((-1, 0), (-1, 1), ls='--', c='black', lw=1.5)
        ax[1].axline((1, 0), (1, 1), ls='--', c='black', lw=1.5)
        ax[1].set_title("Doping Based on AD-NEGF")
        ax[2].plot(ee, TT_BO.detach(), c='tab:green', label='BO')
        ax[2].axline((-1, 0),(-1,1),ls='--',c='black', lw=1.5)
        ax[2].axline((1, 0), (1, 1),ls='--', c='black', lw=1.5)
        ax[2].set_title("Doping Based on Bayesian Optimization")
        ax[3].plot(ee, TT_GA.detach(), c='tab:red', label='GA')
        ax[3].axline((-1, 0),(-1,1),ls='--',c='black', lw=1.5)
        ax[3].axline((1, 0), (1, 1), ls='--', c='black', lw=1.5)
        ax[3].set_title("Doping Based on Genetic Algorithm")
        fig.add_subplot(111, frameon=False)
        # fig.legend(loc='upper center', fontsize=12, ncol=4)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.tight_layout()
        plt.xlabel("E (eV)", fontsize=14)
        plt.ylabel("T(E)", fontsize=14)


        # plt.tight_layout()
        plt.savefig("./experimental_data/Dop_TT.pdf", dpi=600)
        plt.show()








if __name__ == '__main__':
    # f = torch.load("./M.pth")
    # print(f)
    # plt.plot(f[0])
    # plt.plot(f[1])
    # plt.show()
    # gamma0 = -2.97
    # gamma1 = -0.073
    # gamma2 = -0.33
    # s0 = 0.073
    # s1 = 0.018
    # s2 = 0.026
    #
    # CM = torch.tensor([gamma0, gamma1, gamma2, gamma0, gamma1, gamma2, s0, s1, s2, s0, s1, s2, -0.28], dtype=torch.float64)
    # # CM = torch.tensor([-2.7305, -0.0730, -0.3714, -2.9932,  1.4914, -1.0927,  0.0720,  0.0825,
    # #     -0.3587,  0.2019,  0.0180,  0.1431, -0.6131], dtype=torch.float64)
    # # # CM.requires_grad_()
    # #
    # period, coord = create_graphene_nanoribbon(tuneList=[])
    # negf, h = rebuild_hamiltonian(CM, coord, period)
    #
    # k_points = np.linspace(0.0, np.pi / period[0][1], 20)
    # band_structure = torch.zeros((len(k_points), h.h_matrix.shape[0]))
    #
    # for jj, item in enumerate(k_points):
    #     band_structure[jj, :], _ = h.diagonalize_periodic_bc([0.0, item, 0.0])
    #
    # # visualize
    # ax = plt.axes()
    # ax.set_title('Graphene nanoribbon, armchair 11')
    # ax.set_ylabel('Energy (eV)')
    # ax.set_xlabel(r'Wave vector ($\frac{\pi}{a}$)')
    # ax.plot(k_points, np.sort(band_structure.detach().numpy()), 'k')
    # ax.xaxis.grid()
    # plt.show()
    #
    # TT = compute_TT(CM)
    # plt.plot(TT)
    #
    # C = []
    # C_M = []
    # for i in range(10):
    #     I = negf.calCurrent_NUM(ul=0, ur=i*0.3, atom_coord=h.get_site_coordinates()[h._offsets],
    #         d_trains=1,
    #         left_pos=period[0][0],
    #         right_pos=period[0][1],
    #         offset=h._offsets,
    #         etaLead=1e-5,
    #         etaDevice=0.,
    #         ifSCF=True,
    #         n_int_neq=100,
    #         cutoff=True)
    #     C.append(I)
    #
    # CM = torch.tensor([-2.7305, -0.0730, -0.3714, -2.9932,  1.4914, -1.0927,  0.0720,  0.0825,
    #     -0.3587,  0.2019,  0.0180,  0.1431, -0.6131], dtype=torch.float64)
    #
    # negf, h = rebuild_hamiltonian(CM, coord, period)
    #
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
    #     C_M.append(I)
    #
    # C = torch.stack(C)
    # C_M = torch.stack(C_M)
    # torch.save((C,C_M), "./M.pth")
    #
    # plt.plot(C)
    # plt.plot(C_M)
    # plt.show()

    # for i in range(1):
    #     BO(i, delta=0.3)
    # for i in range(1):
    #     GB(k=i, delta=0.3, method='RMSprop')

    # f = torch.load("./dop_GB_RMSprop12.pth")
    # plt.plot(f['loss'])
    # plt.show()
    # TT = compute_TT(f['CM'])
    # plt.plot(TT)
    # plt.show()
    #
    # for i in range(1,2):
    #     Genetic(k=i, delta=0.3)
    plot("TT")