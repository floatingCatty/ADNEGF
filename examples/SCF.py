"""
This example script computes band structure of the crystalline bismuth.
It uses the third-nearest neighbor approximation with a step-wise distance distance
associating distances with sets of TB parameters.
"""
import numpy as np
import matplotlib.pyplot as plt
from TB.hamiltonian import Hamiltonian
from TB.orbitals import Orbitals
from calc.surface_green import selfEnergy
import torch
from TB.hamiltonian_initializer import set_tb_params
from TB.aux_functions import get_k_coords
from tqdm import tqdm
from torch.optim import Adam, SGD, RMSprop, LBFGS
from calc.transport import *
import matplotlib.pyplot
from calc.SCF import *
from tqdm import tqdm

def get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors, coordinate_of_high_symmetry_point):
    cartesian_coordinate_in_reciprocal_space = np.squeeze(np.asarray(np.matrix(coordinate_of_high_symmetry_point) *
                                                                     reciprocal_lattice_vectors)).tolist()
    return cartesian_coordinate_in_reciprocal_space
a = 4.5332
c = 11.7967
g = 1.3861
gamma = 0.240385652727133


# Primitive cell

primitive_cell_a = a * np.array([[(-1.0 / 2.0), (-np.sqrt(3.0) / 6.0), 0.0],
                                 [(1.0 / 2.0), (-np.sqrt(3.0) / 6.0), 0.0],
                                 [0.0, (np.sqrt(3.0) / 3.0), 0.0]])

primitive_cell_c = c * np.array([[0.0, 0.0, (1.0 / 3.0)],
                                 [0.0, 0.0, (1.0 / 3.0)],
                                 [0.0, 0.0, (1.0 / 3.0)]])

primitive_cell = primitive_cell_a + primitive_cell_c

reciprocal_lattice_vectors_bi = g * np.matrix([[-1.0, (-np.sqrt(3.0) / 3.0), (a / c)],
                                                  [1.0, (-np.sqrt(3.0) / 3.0), (a / c)],
                                                  [0.0, (2.0 * np.sqrt(3.0) / 3.0), (a / c)]])
SPECIAL_K_POINTS_BI = {
    'LAMBDA': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [0.25, 0.25, 0.25]),
    'GAMMA': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi,  [0.00, 0.00, 0.00]),
    'T': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi,      [0.50, 0.50, 0.50]),
    'L': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi,      [0.00, 0.50, 0.00]),
    'X': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi,      [0.50, 0.50, 0.00]),
    'K': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi,      [((0.50 * gamma) + 0.25), (0.75 -
                                                                                          (0.50 * gamma)), 0.00]),
    'W': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi,      [0.50, (1.00 - gamma), gamma]),
    'U': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi,      [((0.50 * gamma) + 0.25), (1.00 -
                                                                                           gamma), ((0.50 * gamma) + 0.25)])
}

def radial_dep(coords):
    """
        Step-wise radial dependence function
    """
    norm_of_coords = np.linalg.norm(coords)
    if norm_of_coords < 3.3:
        return 1
    elif 3.7 > norm_of_coords > 3.3:
        return 2
    elif 5.0 > norm_of_coords > 3.7:
        return 3
    else:
        return 100


def main():
    # define atomic coordinates
    path_to_xyz_file = """2
                          Bi2 cell
                          Bi1       0.0    0.0    0.0
                          Bi2       0.0    0.0    5.52321494"""

    # define basis set
    bi_orb = Orbitals('Bi')
    bi_orb.add_orbital("s", energy=-10.906, orbital=0, magnetic=0, spin=0)
    bi_orb.add_orbital("px", energy=-0.486, orbital=1, magnetic=-1, spin=0)
    bi_orb.add_orbital("py", energy=-0.486, orbital=1, magnetic=1, spin=0)
    bi_orb.add_orbital("pz", energy=-0.486, orbital=1, magnetic=0, spin=0)
    bi_orb.add_orbital("s", energy=-10.906, orbital=0, magnetic=0, spin=1)
    bi_orb.add_orbital("px", energy=-0.486, orbital=1, magnetic=-1, spin=1)
    bi_orb.add_orbital("py", energy=-0.486, orbital=1, magnetic=1, spin=1)
    bi_orb.add_orbital("pz", energy=-0.486, orbital=1, magnetic=0, spin=1)


    # define TB parameters
    # 1NN - Bi-Bi
    PAR1 = {'ss_sigma': -0.608,
            'sp_sigma': 1.320,
            'pp_sigma': 1.854,
            'pp_pi': -0.600}

    # 2NN - Bi-Bi
    PAR2 = {'ss_sigma': -0.384,
            'sp_sigma': 0.433,
            'pp_sigma': 1.396,
            'pp_pi': -0.344}

    # 3NN - Bi-Bi
    PAR3 = {'ss_sigma': 0,
            'sp_sigma': 0,
            'pp_sigma': 0.156,
            'pp_pi': 0}

    set_tb_params(PARAMS_BI_BI1=PAR1, PARAMS_BI_BI2=PAR2, PARAMS_BI_BI3=PAR3)

    # compute Hamiltonian matrices
    h = Hamiltonian(xyz=path_to_xyz_file, nn_distance=[3.3, 3.7, 4.6], so_coupling=1.5, comp_overlap=False).initialize()
    h.initialize()
    HL, HD, HR = h.get_hamiltonians()
    hl, hd, hu, _ = h.get_hamiltonians_block_tridiagonal(optimized=True)
    lhd, rhd = HD, HD
    lhu, rhu = HL, HL.conj().T
    lhl, rhl = HR.conj().T, HR

    sd = [torch.ones_like(hd[i]) for i in range(len(hd))]
    su = [torch.zeros_like(hu[i]) for i in range(len(hu))]
    sl = [torch.zeros_like(hl[i]) for i in range(len(hl))]
    lsd, rsd = torch.zeros_like(lhd), torch.zeros_like(lhd)
    lsu, rsl = torch.zeros_like(lhu), torch.zeros_like(lhu)
    lsl, rsu = torch.zeros_like(lhl), torch.zeros_like(lhl)

    init_del_V = torch.zeros((2,), dtype=torch.double)

    err, maxItr = 1e-1, 1000
    offsets = h._offsets
    zs, zd = -5.52321494, 5.52321494+5.52321494
    corrd = citeCoord2Coord(h._offsets, torch.tensor(h.get_site_coordinates()))
    Emin = -27
    ul, ur = torch.tensor(0.).requires_grad_(), torch.tensor(2.).requires_grad_()
    N = h.basis_size
    n_img = 5000


    # Transmission = []
    # x = []
    # for i in tqdm(range(-200, 200)):
    #     x.append(i/100)
    #     seL, _ = selfEnergy(hd=lhd, hl=lhl, hu=lhu, sd=lsd, su=lsu, sl=lsl, ee=i/100, left=True, voltage=0)
    #     seR, _ = selfEnergy(hd=rhd, hl=rhl, hu=rhu, sd=rsd, su=rsu, sl=rsl, ee=i / 100, left=False, voltage=0)
    #     gammaL, gammaR = sigmaLR2Gamma(seL), sigmaLR2Gamma(seR)
    #     g_trans, _, _, _, _ = recursive_gf(i/100, hl=hl, hd=hd, hu=hu, sd=sd, su=su, sl=sl, left_se=seL, right_se=seR, seP=None, s_in=None, s_out=None)
    #
    #     TT = (gammaL[:1,:1] @ g_trans @ gammaR[:4,:4] @ g_trans.conj().T).real.trace()
    #     Transmission.append(TT.detach().numpy())
    #
    # plt.plot(x, Transmission)
    # plt.title("Transmission")
    # plt.show()

    # with torch.enable_grad():
    #     rho_neq = calNeqDensity(N, ul, ur,
    #                             hd=hd,hu=hu,hl=hl,sd=sd,su=su,sl=sl,lsd=lsd,lsu=lsu,lsl=lsl,rsd=rsd,rsu=rsu,rsl=rsl,lhd=lhd,lhu=lhu,lhl=lhl,rhd=rhd,rhu=rhu,rhl=rhl)
    #     print(rho_neq)
    #     print(torch.autograd.grad(rho_neq, hd[1], grad_outputs=torch.ones_like(rho_neq)))

    convergedPotential = SCFIteration(
        N, init_del_V, zs, zd, offsets, corrd, ul, ur,
        hd=hd,hu=hu,hl=hl,sd=sd,su=su,sl=sl,lsd=lsd,lsu=lsu,lsl=lsl,rsd=rsd,rsu=rsu,rsl=rsl,lhd=lhd,lhu=lhu,lhl=lhl,rhd=rhd,rhu=rhu,rhl=rhl,
        method='LBFGS'
    )

    convergedPotential.sum().backward()
    for p in bi_orb.params:
        print(p.grad)



if __name__ == '__main__':
    main()