"""
This example script computes band structure of the crystalline bismuth.
It uses the third-nearest neighbor approximation with a step-wise distance distance
associating distances with sets of TB parameters.
"""
import numpy as np
import matplotlib.pyplot as plt
from TB.hamiltonian import Hamiltonian
from TB.orbitals import Orbitals
import torch
from TB.hamiltonian_initializer import set_tb_params, set_tb_params_bond_length
from TB.aux_functions import get_k_coords
from tqdm import tqdm
from torch.optim import Adam, SGD, RMSprop, LBFGS
from torch.autograd import gradcheck
from mpl_toolkits.mplot3d import Axes3D
import sys  # 导入sys模块
sys.setrecursionlimit(3000)  # 将默认的递归深度修改为3000

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
    # # define atomic coordinates
    # path_to_xyz_file = """2
    #                       Bi2 cell
    #                       Bi1       0.0    0.0    0.0
    #                       Bi2       0.0    0.0    5.52321494"""
    # path_to_xyz_file_new = """2
    #                       Bi2 cell
    #                       Bi1       0.0    0.0    0.0
    #                       Bi2       0.0    0.0    5.52321494"""

    # # define basis set
    # bi_orb = Orbitals('Bi')
    # bi_orb.add_orbital("s", energy=-10.906, orbital=0, magnetic=0, spin=0)
    # bi_orb.add_orbital("px", energy=-0.486, orbital=1, magnetic=-1, spin=0)
    # bi_orb.add_orbital("py", energy=-0.486, orbital=1, magnetic=1, spin=0)
    # bi_orb.add_orbital("pz", energy=-0.486, orbital=1, magnetic=0, spin=0)
    # bi_orb.add_orbital("s", energy=-10.906, orbital=0, magnetic=0, spin=1)
    # bi_orb.add_orbital("px", energy=-0.486, orbital=1, magnetic=-1, spin=1)
    # bi_orb.add_orbital("py", energy=-0.486, orbital=1, magnetic=1, spin=1)
    # bi_orb.add_orbital("pz", energy=-0.486, orbital=1, magnetic=0, spin=1)

    from ase.build.ribbon import graphene_nanoribbon
    from ase.visualize.plot import plot_atoms

    atoms = graphene_nanoribbon(0.5, 7, type='armchair', saturated=True)

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
    
    gamma0 = -2.97
    gamma1 = -0.073
    gamma2 = -0.33
    s0 = 0.073
    s1 = 0.018
    s2 = 0.026

    set_tb_params_bond_length(s_orb, BL_C_C1={'pp_pi': gamma0},
                     BL_C_C2={'pp_pi': gamma1},
                     BL_C_C3={'pp_pi': gamma2},
                     OV_C_C1={'pp_pi': s0},
                     OV_C_C2={'pp_pi': s1},
                     OV_C_C3={'pp_pi': s2})

    # # define TB parameters
    # # 1NN - Bi-Bi
    # PAR1 = {'ss_sigma': -0.608,
    #         'sp_sigma': 1.320,
    #         'pp_sigma': 1.854,
    #         'pp_pi': -0.600}

    # # 2NN - Bi-Bi
    # PAR2 = {'ss_sigma': -0.384,
    #         'sp_sigma': 0.433,
    #         'pp_sigma': 1.396,
    #         'pp_pi': -0.344}

    # # 3NN - Bi-Bi
    # PAR3 = {'ss_sigma': 0,
    #         'sp_sigma': 0,
    #         'pp_sigma': 0.156,
    #         'pp_pi': 0}

    # set_tb_params(s_orb, PARAMS_BI_BI1=PAR1, PARAMS_BI_BI2=PAR2, PARAMS_BI_BI3=PAR3)

    # # 1NN - Bi-Bi
    # bl_PAR1 = {'ss_sigma': -0.608,
    #         'sp_sigma': 1.320,
    #         'pp_sigma': 1.854,
    #         'pp_pi': -0.600}

    # # 2NN - Bi-Bi
    # bl_PAR2 = {'ss_sigma': -0.384,
    #         'sp_sigma': 0.433,
    #         'pp_sigma': 1.396,
    #         'pp_pi': -0.344}

    # # 3NN - Bi-Bi
    # bl_PAR3 = {'ss_sigma': 0,
    #         'sp_sigma': 0,
    #         'pp_sigma': 0.156,
    #         'pp_pi': 0}

    # set_tb_params_bond_length(s_orb, BL_BI_BI1=bl_PAR1, BL_BI_BI2=bl_PAR2, BL_BI_BI3=bl_PAR3)

    def radial_dependence_func(bond_length, ne_bond_length, param):
        return (bond_length/ne_bond_length)**param

    def sorting(coords, **kwargs):
        return np.argsort(coords[:, 1], kind='mergesort')

    # compute Hamiltonian matrices
    h = Hamiltonian(radial_dep=radial_dependence_func,xyz=coords, xyz_new=coords, nn_distance=[1.5, 2.5, 3.1], comp_overlap=True)
    h.initialize()
    # h2 = h.h_matrix.clone().detach()
    h.set_periodic_bc(period)
    # print("###########################################################")
    # print(h._atom_list['Bi2'].grad)
    # print(h._bond_length[0][1].grad)
    # print(h.h_matrix.grad)
    # print("###########################################################")

    # define wave vectors
    sym_points = ['K', 'GAMMA', 'T', 'W', 'L', 'LAMBDA']
    num_points = [10, 10, 10, 10, 10]
    k_points = get_k_coords(sym_points, num_points, SPECIAL_K_POINTS_BI)

    # compute band structure
    band_structure = []

    for jj, item in tqdm(enumerate(k_points)):
        [eigenvalues, _] = h.diagonalize_periodic_bc(k_points[jj])
        band_structure.append(eigenvalues)
        print(eigenvalues)
        exit()

    band_structure = torch.stack(band_structure).detach()
    
    # --------------------------------reinitialized the parameters------------------

    for param in h.params:
        # print(param)
        param.data = torch.tensor(torch.randn(3), dtype=torch.float64)
        # print(param)

    def loss():
        optim.zero_grad()
        # h = Hamiltonian(xyz=path_to_xyz_file, xyz_new=path_to_xyz_file_new, radial_dep=radial_dependence_func, nn_distance=[3.3, 3.7, 4.6], so_coupling=1.5, comp_overlap=False)
        h.initialize()
        # for key in h._atom_list_new.keys():
        #     print(h._atom_list_new[key])
        # for i in range(len(h._bond_length)):
        #     for j in range(len(h._bond_length[i])):
        #         print(h._bond_length[i][j])
        # print(h2 == h.h_matrix)
        h.set_periodic_bc(period)
        pre_band = []
        for jj, item in tqdm(enumerate(k_points)):
            [eigenvalues, _] = h.diagonalize_periodic_bc(k_points[jj])
            pre_band.append(eigenvalues)

        pre_band = torch.stack(pre_band)
        # history_band.append(pre_band.detach)

        l = (pre_band - band_structure).abs().mean()
        l.backward()
        print(l.item())
        # for param in h.params:
        #     print(param)
        for key in h._atom_list.keys():
            print(h._atom_list[key])

        return l

    optim = LBFGS(params=h.params, lr=1e-1, max_iter=50, max_eval=None,
                                  tolerance_grad=1e-05,
                                  tolerance_change=1e-09,
                                  history_size=100,
                                  line_search_fn='strong_wolfe')

    history_band = []
    optim.step(loss)
    torch.save(obj=history_band, f="./bi_optim.pth")


if __name__ == '__main__':
    main()