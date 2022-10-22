from calc.transport import attachPotential, sigmaLR2Gamma, finite_difference, calTT, calCurrent
from tqdm import tqdm
from calc.NEGF import NEGF
import ase
from Constant import *
from TB.hamiltonian_initializer import set_tb_params, set_tb_params_bond_length
import time
from calc.surface_green import selfEnergy
from calc.RGF import recursive_gf
from calc.SCF import SCF_with_hTB
from TB import *
import seaborn as sns
import torch
import matplotlib.pyplot as plt


def set_up(l, w):
    from ase.build.ribbon import graphene_nanoribbon
    from ase.visualize.plot import plot_atoms

    atoms = graphene_nanoribbon(4, 4, type='armchair', saturated=False)

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

    set_tb_params(s_orb, PARAMS_C_C1={'pp_pi': gamma0},
                  PARAMS_C_C2={'pp_pi': gamma1},
                  PARAMS_C_C3={'pp_pi': gamma2},
                  OV_C_C1={'pp_pi': s0},
                  OV_C_C2={'pp_pi': s1},
                  OV_C_C3={'pp_pi': s2})

    # --------------------------- Hamiltonian -------------------------

    h = Hamiltonian(xyz=coords, nn_distance=[1.5, 2.5, 3.1], comp_overlap=True, sort_func=sorting).initialize()
    h.set_periodic_bc(period)

    return h

def sorting(coords, **kwargs):
    return np.argsort(coords[:, 1], kind='mergesort')

def TT_DOS_with_ASE(hamiltonian, ul, ur, zs, zd, el, er, n, loadData=True):
    hL, hD, hR, sL, sD, sR = hamiltonian.get_hamiltonians()
    hl_list, hd_list, hr_list, sl_list, sd_list, sr_list, subblocks = \
        hamiltonian.get_hamiltonians_block_tridiagonal(optimized=True)
    ee_list = torch.linspace(start=el, end=er, steps=n)
    transmission = []
    DOS = []

    V_ext = SCF_with_hTB(hamiltonian, zs=zs, zd=zd, ul=ul, ur=ur)
    hd_ = attachPotential(hamiltonian._offsets, hd_list, V_ext, hamiltonian.basis_size)

    def fn(ee):
        dos = 0
        seL, _ = selfEnergy(ee=ee, hd=hD, hu=hL.conj().T, sd=sD, su=sL.conj().T, left=True, voltage=ul, method='Lopez-Sancho')
        seR, _ = selfEnergy(ee=ee, hd=hD, hu=hR, sd=sD, su=sR, left=False, voltage=ur, method='Lopez-Sancho')
        g_trans, grd, _, _, _ = recursive_gf(ee, hl=hl_list, hd=hd_, hu=hr_list, sd=sd_list, su=sr_list,
                                           sl=sl_list, left_se=seL, right_se=seR, seP=None, s_in=None, s_out=None)
        s01, s02 = hd_list[0].shape
        seL = seL[:s01, :s02]
        s11, s12 = hd_list[-1].shape
        seR = seR[-s11:, -s12:]
        gammaL, gammaR = sigmaLR2Gamma(seL), sigmaLR2Gamma(seR)
        TT = calTT(gammaL, gammaR, g_trans)
        for jj in range(len(hd_list)):
            temp = grd[jj] @ sd_list[jj]
            dos -= torch.trace(temp.imag) / pi
        return TT, dos

    if loadData:
        f = torch.load("./experimental_data/TT_DOS_with_ASE.pth")
        transmission, DOS, ASETT, ASEDOS = f['transmission'], f['DOS'], f['ASETT'], f['ASEDOS']
    else:
        for ee in tqdm(ee_list, desc="Transmission"):
            TT, dos = fn(ee)
            transmission.append(TT)
            DOS.append(dos)
        n_L = hd_list[0].shape[0]
        n_R = hd_list[-1].shape[0]
        with torch.no_grad():
            ASE = ase.transport.TransportCalculator(energies=ee_list.numpy(), h=hD.numpy(), s=sD.numpy(),
            h1=hD.numpy(), s1=sD.numpy(), h2=hD.numpy(), s2=sD.numpy(), dos=True)
            ASETT = ASE.get_transmission()
            ASEDOS = ASE.get_dos()
        transmission = torch.tensor(transmission)
        DOS = torch.tensor(DOS)

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7.5, 5.8))


    ax[0].plot(ee_list, ASETT, 'b-', c='tab:red')
    ax[0].plot(ee_list[::6], ASETT[::6], 'b+', c='tab:red')
    ax[0].plot(-1, -3, 'b+', label='ASE', c='tab:red')
    ax[0].plot(ee_list, transmission.detach().numpy(), c='black', label='AD-NEGF')
    ax[0].set_ylabel("T(E)", fontsize=16)
    ax[0].set_title("Transmission of AGNR(7)", fontsize=16)
    # ax[0].legend(loc=1, fontsize=14)
    ax[0].set_xlim((el, er))
    ax[0].set_ylim((-0.2,4.2))


    ax[1].plot(ee_list, ASEDOS, 'b-', c='tab:red')
    ax[1].plot(ee_list[::6], ASEDOS[::6], 'b+', c='tab:red')
    ax[1].plot(6, 60, 'b+', label='ASE', c='tab:red')
    ax[1].plot(ee_list, DOS.numpy(), c='black', label='AD-NEGF')
    ax[1].set_xlabel("E (ev)", fontsize=16)
    ax[1].set_ylabel("DOS", fontsize=16)
    ax[1].set_title("DOS of AGNR(7)", fontsize=16)
    plt.legend(loc=1, fontsize=14)
    ax[1].set_xlim((el, er))
    ax[1].set_ylim((-5,54))

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.savefig("./experimental_data/TT_DOS_with_ASE.pdf", dpi=600)
    plt.show()

    # if not loadData:
    #     torch.save(obj={"transmission":transmission, "DOS":DOS, "ASEDOS":ASEDOS, "ASETT":ASETT},
    #                f='./experimental_data/TT_DOS_with_ASE.pth')

    return True

def Seebeck(hamiltonian, zs, zd, el, er, ul, ur, n, fd_step=1e-6, dtype=torch.float64, loadData=True):
    hL, hD, hR, sL, sD, sR = hamiltonian.get_hamiltonians()
    hl_list, hd_list, hr_list, sl_list, sd_list, sr_list, subblocks = \
        hamiltonian.get_hamiltonians_block_tridiagonal(optimized=True)

    ee_list = torch.linspace(start=el, end=er, steps=n, dtype=dtype)
    seebeck = []
    seebeckFD = []
    transmission = []
    def fn(ee):
        seL, _ = selfEnergy(ee=ee, hd=hD, hu=hL.conj().T, sd=sD, su=sL.conj().T, left=True, voltage=ul, dtype=dtype, method='Lopez-Sancho')
        seR, _ = selfEnergy(ee=ee, hd=hD, hu=hR, sd=sD, su=sR, left=False, voltage=ur, dtype=dtype, method='Lopez-Sancho')
        g_trans, _, _, _, _ = recursive_gf(ee, hl=hl_list, hd=hd_, hu=hr_list, sd=sd_list, su=sr_list,
                                           sl=sl_list, left_se=seL, right_se=seR, seP=None, s_in=None, s_out=None)
        s01, s02 = hd_[0].shape
        seL = seL[:s01, :s02]
        s11, s12 = hd_[-1].shape
        seR = seR[-s11:, -s12:]
        gammaL, gammaR = sigmaLR2Gamma(seL), sigmaLR2Gamma(seR)
        TT = calTT(gammaL, gammaR, g_trans)
        return TT

    if loadData:
        f = torch.load("./experimental_data/Seebeck.pth")
        transmission, seebeck, seebeckFD, start, endAD, endFD = f['transmission'], f['seebeck'], f['seebeckFD'], f['start'], \
                                                                f['endAD'], f['endFD']
    else:
        ee_list.requires_grad_()
        V_ext = SCF_with_hTB(
            hamiltonian=hamiltonian,
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
        print(V_ext)
        hd_ = attachPotential(hamiltonian._offsets, hd_list, V_ext, hamiltonian.basis_size)

        start = time.time()
        for ee in tqdm(ee_list, desc="Transmission"):
            TT = fn(ee)
            transmission.append(TT)
            seebeck.append(- torch.autograd.grad(TT, ee)[0] / (TT+1e-6))
        endAD = time.time()

        for ee in tqdm(ee_list, desc="FD Seebeck"):
            seebeckFD.append(- finite_difference(fn, ee, h=fd_step, dtype=dtype) / (fn(ee)+1e-6))
        endFD = time.time()

        transmission, seebeck, seebeckFD = torch.stack(transmission).detach(), torch.stack(seebeck).detach(), torch.stack(seebeckFD).detach()

    print("Time for AD: {0}, for FD {1}.".format(endAD - start, endFD - endAD))
    ee_list = ee_list.detach()

    vr_list, current, dIdV = torch.load(f="./experimental_data/IV_data.pth")
    print(dIdV)


    fig = plt.figure()
    fig, ax = plt.subplots(2, 1, figsize=(8.0, 6.6))

    ax[0].plot(ee_list, seebeckFD.detach(), 'b-', c='tab:red')
    ax[0].plot(ee_list[::6], seebeckFD.detach()[::6], 'b+', c='tab:red')
    ax[0].plot(-1, -60, 'b+', label='Seebeck Coefficient by FD', c='tab:red')
    ax[0].plot(ee_list, seebeck, c='black', label='Seebeck Coefficient by AD')
    # ax[0].set_title("transmission & seebeck under bias voltage of {0}~{1}".format(ul, ur),fontsize=14)
    ax[0].set_xlabel("E (ev)",fontsize=16)
    ax[0].set_ylabel("S(E)",fontsize=16)
    ax[0].set_xlim((el, er))
    ax[0].set_ylim((-50, 50))


    ax2 = ax[0].twinx()

    ax2.plot(ee_list, transmission, c='tab:blue', label='T(E)',ls='-.')
    ax2.set_ylabel("T(E)",fontsize=16)
    ax[0].legend(fontsize=12, loc="upper right", ncol=1, bbox_to_anchor=(1.02,1.32))
    ax2.legend(['T(E)'], fontsize=12, loc='upper left', ncol=1, bbox_to_anchor=(-0.02,1.20))
    ax[1].plot(vr_list, current, c='tab:blue', ls='-.')

    # ax[1].set_title("IV & Differential Conductance", fontsize=14)
    ax[1].set_xlabel("V (eV)", fontsize=16)
    ax[1].set_ylabel("I (V)", fontsize=16)

    ax2 = ax[1].twinx()

    ax2.plot(vr_list, dIdV, c='black')
    ax2.set_ylabel("dI/dV", fontsize=16)

    ax[1].legend(['I (V)'], fontsize=12, loc='upper left')
    ax2.legend(['dI/dV'], fontsize=12, loc='upper left',bbox_to_anchor=(0.3,1.0))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.savefig("./experimental_data/Seebeck_IV.pdf", dpi=600)


    plt.show()

    if not loadData:
        torch.save(obj={"transmission":transmission, "V_ext":V_ext, "seebeck":seebeck, "seebeckFD":seebeckFD, "start":start, "endAD":endAD, "endFD":endFD},
                   f='./experimental_data/Seebeck.pth')

    return

def Seebeck_Compare(hamiltonian, zs, zd, el, er, ul, ur, n, fd_step, dtype=torch.float64, loadData=True):
    hL, hD, hR, sL, sD, sR = hamiltonian.get_hamiltonians()
    hl_list, hd_list, hr_list, sl_list, sd_list, sr_list, subblocks = \
        hamiltonian.get_hamiltonians_block_tridiagonal(optimized=True)
    ee_list = torch.linspace(start=el, end=er, steps=n, dtype=dtype)
    t = torch.tensor(0, dtype=torch.float16)
    t1 = torch.tensor(0, dtype=torch.float64)
    seebeck = []
    seebeckFD = []
    if loadData:
        f = torch.load("./experimental_data/Seebeck_Compare.pth")
        V_ext = f['V_ext']
    else:
        V_ext = SCF_with_hTB(
            hamiltonian=hamiltonian,
            n_img=200,
            err=1e-7,
            maxIter=500,
            zs=zs,
            zd=zd,
            d_trans=1,
            Emin=-20,
            ul=ul,
            ur=ur,
            method='PDIIS'
        )
    hd_ = attachPotential(hamiltonian._offsets, hd_list, V_ext.type_as(hd_list[0]), hamiltonian.basis_size)

    def fn(ee):
        seL, _ = selfEnergy(ee=ee, hd=hD, hu=hL.conj().T, sd=sD, su=sL.conj().T, left=True, voltage=ul, dtype=dtype)
        seR, _ = selfEnergy(ee=ee, hd=hD, hu=hR, sd=sD, su=sR, left=False, voltage=ur, dtype=dtype)
        g_trans, _, _, _, _ = recursive_gf(ee, hl=hl_list, hd=hd_, hu=hr_list, sd=sd_list, su=sr_list,
                                           sl=sl_list, left_se=seL, right_se=seR, seP=None, s_in=None, s_out=None)
        s01, s02 = hd_[0].shape
        seL = seL[:s01, :s02]
        s11, s12 = hd_[-1].shape
        seR = seR[-s11:, -s12:]
        gammaL, gammaR = sigmaLR2Gamma(seL), sigmaLR2Gamma(seR)
        TT = calTT(gammaL, gammaR, g_trans)
        return TT.type_as(t).type_as(t1)

    start = time.time()
    ee_list.requires_grad_()
    for ee in tqdm(ee_list, desc="Transmission"):
        TT = fn(ee)
        seebeck.append(- torch.autograd.grad(TT, ee)[0] / (TT+1e-6))
    endAD = time.time()

    for i in fd_step:
        sbfd = []
        for ee in tqdm(ee_list, desc="FD Seebeck"):
            sbfd.append( - finite_difference(fn, ee, h=i, dtype=torch.float16) / (fn(ee)+1e-6))
        seebeckFD.append(torch.tensor(sbfd))
    endFD = time.time()

    seebeck, seebeckFD = torch.stack(seebeck).detach(), torch.stack(seebeckFD).detach()
    print("Time for AD: {0}, for FD {1}.".format(endAD - start, (endFD - endAD)/len(fd_step)))
    ee_list = ee_list.detach()

    fig, (ax) = plt.subplots(len(fd_step) + 1, 1, sharex=True, sharey=True, figsize=(5.8, 6.4))
    # fig.set_title("transmission & seebeck under bias voltage of {0}~{1}".format(ul, ur))
    # fig.set_xlabel("E/ev")
    # fig.set_ylabel("S(E): dT/dE")
    ax[0].plot(ee_list, seebeck, c="black")
    ax[0].set_title("AD")
    ax[0].set_xlim((el, er))
    ax[0].set_ylim((-20, 20))
    ax[1].plot(ee_list, seebeckFD.detach()[0], c=sns.color_palette("Blues")[5])
    ax[1].set_title("FD (step-size=1e-2)")
    ax[1].set_ylim((-20, 20))
    ax[2].plot(ee_list, seebeckFD.detach()[1], c=sns.color_palette("Blues")[5])
    ax[2].set_title("FD (step-size=1e-3)")
    ax[2].set_ylim((-20, 20))
    ax[3].plot(ee_list, seebeckFD.detach()[2], c=sns.color_palette("Blues")[5])
    ax[3].set_title("FD (step-size=1e-4)")
    ax[3].set_ylim((-20, 20))
    ax[4].plot(ee_list, seebeckFD.detach()[3], c=sns.color_palette("Blues")[5])
    ax[4].set_title("FD (step-size=1e-5)")
    ax[4].set_ylim((-20, 20))
    plt.xlabel("E (ev)", fontsize=14)
    # fig.legend(['AD', 'FD(h=1e-3)', 'FD(h=1e-4)', 'FD(h=1e-5)', 'FD(h=1e-6)'], fontsize=12, loc="upper center", ncol=3)
    plt.tight_layout()
    plt.savefig("./experimental_data/Seebeck_Compare.pdf", dpi=600)
    plt.show()

    if not loadData:
        torch.save(obj={"V_ext":V_ext},
                   f='./experimental_data/Seebeck_Compare.pth')

    return True

def IV(hamiltonian, zs, zd, V_max, n, ifdIdV=True, loadData=True):
    hL, hD, hR, sL, sD, sR = hamiltonian.get_hamiltonians()
    hl_list, hd_list, hr_list, sl_list, sd_list, sr_list, subblocks = \
        hamiltonian.get_hamiltonians_block_tridiagonal(optimized=True)
    hr_list = hr_list[:-1]
    sr_list = sr_list[:-1]
    hl_list = hl_list[:-1]
    sl_list = sl_list[:-1]
    vl = torch.tensor(0.)
    vr_list = torch.linspace(start=0, end=V_max, steps=n)
    if loadData:
        V_list = torch.load("./experimental_data/IV.pth")['V_ext']
    else:
        V_list = [torch.zeros(hD.shape[0])]
    current = [torch.tensor(0.)]
    dIdV = [torch.tensor(0.)]
    i = 1
    for vr in tqdm(vr_list[1:]):
        if ifdIdV:
            vr.requires_grad_()
            vl.requires_grad_()
        if loadData:
            V_ext = V_list[i]
        else:
            V_ext = SCF_with_hTB(
                hamiltonian=hamiltonian,
                n_img=200,
                err=1e-7,
                maxIter=100,
                zs=zs,
                zd=zd,
                d_trans=1,
                Emin=-20,
                ul=vl,
                ur=vr,
                method='PDIIS'
            )
            V_list.append(V_ext)

        if ifdIdV:
            with torch.enable_grad():
                hd_ = attachPotential(hamiltonian._offsets, hd_list, V_ext, hamiltonian.basis_size)

                I = calCurrent(ul=vl, ur=vr, hd=hd_, hu=hr_list, hl=hl_list, sd=sd_list, su=sr_list,
                               sl=sl_list,
                               lhd=hD, lhu=hL.conj().T, lsd=sD, lsu=sL.conj().T,
                               rhd=hD, rhu=hR, rsd=sD, rsu=sR,)
        else:
            hd_ = attachPotential(hamiltonian._offsets, hd_list, V_ext, hamiltonian.basis_size)

            I = calCurrent(ul=vl, ur=vr, hd=hd_, hu=hr_list, hl=hl_list, sd=sd_list, su=sr_list,
                           sl=sl_list,
                           lhd=hD, lhu=hL.conj().T, lsd=sD, lsu=sL.conj().T,
                           rhd=hD, rhu=hR, rsd=sD, rsu=sR,)
        current.append(I)
        if ifdIdV:
            grad = torch.autograd.grad(I, [vr,vl])
            dIdV.append(grad[0]-grad[1])
        i += 1
    current, dIdV = torch.stack(current).detach(), torch.stack(dIdV).detach()
    vr_list = vr_list.detach()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(vr_list, current, c=sns.color_palette("Blues")[0], linestyle='--')

    ax1.set_title("IV & Differential Conductance", fontsize=14)
    ax1.set_xlabel("V(eV)", fontsize=14)
    ax1.set_ylabel("I(V)", fontsize=14)

    if ifdIdV:
        ax2 = ax1.twinx()

        ax2.plot(vr_list, dIdV, c=sns.color_palette("Blues")[4])
        ax2.set_ylabel("dI/dV", fontsize=14)
        fig.legend(['I(V)', 'dI/dV'], fontsize=14)

        torch.save(obj=(vr_list, current, dIdV), f="./experimental_data/IV_data.pth")

    plt.savefig("./experimental_data/IV.pdf", dpi=600)
    plt.show()


    if not loadData:
        torch.save(obj={"V_ext": V_list},
                   f='./experimental_data/IV.pth')

    if ifdIdV:
        return current, dIdV
    else:
        return current


def parameter_discovery(hamiltonian, el, er, n, loadData=True):
    hL, hD, hR, sL, sD, sR = hamiltonian.get_hamiltonians()
    hl_list, hd_list, hr_list, sl_list, sd_list, sr_list, subblocks = \
        hamiltonian.get_hamiltonians_block_tridiagonal(optimized=True)
    ee_list = torch.linspace(start=el, end=er, steps=n)
    transmission = []
    DOS = []

    def fn(ee):
        dos = 0
        seL, _ = selfEnergy(ee=ee, hd=hD, hu=hL.conj().T, sd=sD, su=sL.conj().T, left=True, voltage=0.)
        seR, _ = selfEnergy(ee=ee, hd=hD, hu=hR, sd=sD, su=sR, left=False, voltage=0.)
        g_trans, grd, _, _, _ = recursive_gf(ee, hl=hl_list, hd=hd_list, hu=hr_list, sd=sd_list, su=sr_list,
                                           sl=sl_list, left_se=seL, right_se=seR, seP=None, s_in=None, s_out=None)
        s01, s02 = hd_list[0].shape
        seL = seL[:s01, :s02]
        s11, s12 = hd_list[-1].shape
        seR = seR[-s11:, -s12:]
        gammaL, gammaR = sigmaLR2Gamma(seL), sigmaLR2Gamma(seR)
        TT = calTT(gammaL, gammaR, g_trans)
        for jj in range(len(hd_list)):
            dos -= 2*torch.trace(grd[jj].imag) / (pi*len(hd_list))
        return TT, dos

    if loadData:
        f = torch.load("./experimental_data/TT_DOS_with_ASE.pth")
        transmission, DOS, ASETT, ASEDOS = f['transmission'], f['DOS'], f['ASETT'], f['ASEDOS']
    else:
        for ee in tqdm(ee_list, desc="Transmission"):
            TT, dos = fn(ee)
            transmission.append(TT)
            DOS.append(dos)
        n_L = hd_list[0].shape[0]
        n_R = hd_list[-1].shape[0]
        with torch.no_grad():
            ASE = ase.transport.TransportCalculator(energies=ee_list.numpy(), h=hD.numpy(), s=sD.numpy(),
            h1=hD.numpy(), s1=sD.numpy(), h2=hD.numpy(), s2=sD.numpy(), dos=True)
            ASETT = ASE.get_transmission()
            ASEDOS = ASE.get_dos()
        transmission = torch.tensor(transmission)
        DOS = torch.tensor(DOS)

    plt.plot(ee_list, ASETT, c=sns.color_palette("Blues")[3], linestyle='--')
    plt.plot(ee_list, transmission.detach().numpy(), c=sns.color_palette("Blues")[5])
    plt.xlabel("E(ev)")
    plt.ylabel("T(E)")
    plt.title("transmission of graphene nanoribbons")
    plt.legend(["ASE", "adNEGF"],loc=1)
    plt.xlim((el, er))
    plt.savefig("./experimental_data/TT_DOS_with_ASE_1.pdf", dpi=600)
    plt.show()

    plt.plot(ee_list, ASEDOS, c=sns.color_palette("Blues")[3], linestyle='--')
    plt.plot(ee_list, DOS.numpy(), c=sns.color_palette("Blues")[5])
    plt.xlabel("E(ev)")
    plt.ylabel("DOS")
    plt.title("DOS of graphene nanoribbons")
    plt.legend(["ASE", "adNEGF"], loc=1)
    plt.xlim((el, er))
    plt.savefig("./experimental_data/TT_DOS_with_ASE_2.pdf", dpi=600)
    plt.show()

    if not loadData:
        torch.save(obj={"transmission":transmission, "DOS":DOS, "ASEDOS":ASEDOS, "ASETT":ASETT},
                   f='./experimental_data/TT_DOS_with_ASE.pth')

    return True

if __name__ == '__main__':
    from ase.build.ribbon import graphene_nanoribbon
    from ase.visualize.plot import plot_atoms
    from ase.visualize import view
    from ase.io import write

    atoms = graphene_nanoribbon(3.5, 4, type='armchair', saturated=False)

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

    # --------------------------- Hamiltonian -------------------------

    h = Hamiltonian(dtype=torch.complex128,xyz=coords, xyz_new=coords, nn_distance=[1.5, 2.5, 3.1], comp_overlap=True, sort_func=sorting).initialize()
    h.set_periodic_bc(period)

    # k_points = np.linspace(0.0, np.pi / period[0][1], 20)
    # band_structure = torch.zeros((len(k_points), h.h_matrix.shape[0]))
    #
    # for jj, item in enumerate(k_points):
    #     band_structure[jj, :], _ = h.diagonalize_periodic_bc([0.0, item, 0.0])
    #
    # ax = plt.axes()
    # # ax.set_title('Band structure of carbon nanotube, ({0}, {1}) \n 1st nearest neighbour approximation'.format(n, m))
    # ax.set_ylabel('Energy (eV)')
    # ax.set_xlabel(r'Wave vector ($\frac{\pi}{a}$)')
    # ax.plot(k_points, np.sort(band_structure.detach().numpy()), 'k')
    # ax.xaxis.grid()
    # plt.show()



    # ax1 = plot_atoms(atoms, show_unit_cell=2, rotation='90x,0y,270z')
    # ax1.axis('off')
    # plt.tight_layout()
    # plt.show()



    # TT_DOS_with_ASE(h, el=-3, er=3, ul=0, ur=0, zs=period[0][0], zd=period[0][1], n=200, loadData=True)
    Seebeck(h, zs=period[0][0], zd=period[0][1], el=-3, er=3, ul=0, ur=0.5, n=400, fd_step=1e-6, dtype=torch.float64, loadData=True)
    # Seebeck_Compare(h, zs=period[0][0], zd=period[0][1], el=-3, er=3, ul=0, ur=0.5, n=800, fd_step=[1e-2,1e-3,1e-4,1e-5], loadData=True, dtype=torch.complex128)

    # current = IV(h, zs=period[0][0], zd=period[0][1], V_max=3, n=100, ifdIdV=True, loadData=True)
