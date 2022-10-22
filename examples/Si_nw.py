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


def sorting(coords, **kwargs):
    return np.argsort(coords[:, 2], kind='mergesort')


def main():
    # use a predefined basis sets
    Orbitals.orbital_sets = {'Si': 'SiliconSP3D5S', 'H': 'HydrogenS'}

    # specify atomic coordinates file stored in xyz format
    path = "input_samples/SiNW2.xyz"

    # define leads indices
    hamiltonian = Hamiltonian(xyz=path,
                              nn_distance=2.4,
                              sort_func=sorting).initialize()

    # set periodic boundary conditions
    a_si = 5.50
    primitive_cell = [[0, 0, a_si]]
    hamiltonian.set_periodic_bc(primitive_cell)

    # get Hamiltonian matrices
    hl, hd, hr = hamiltonian.get_hamiltonians()
    hl_bd, h0_bd, hr_bd, subblocks = hamiltonian.get_hamiltonians_block_tridiagonal(optimized=True)

    lhd, rhd = hd, hd
    lhu, lhl = hl, hl.conj().T
    rhu, rhl = hr.conj().T, hr


    sd = [torch.ones_like(h0_bd[i]) for i in range(len(h0_bd))]
    su = [torch.zeros_like(hr_bd[i]) for i in range(len(hr_bd))]
    sl = [torch.zeros_like(hl_bd[i]) for i in range(len(hl_bd))]
    lsd, rsd = torch.zeros_like(lhd), torch.zeros_like(lhd)
    lsu, rsl = torch.zeros_like(lhu), torch.zeros_like(lhu)
    lsl, rsu = torch.zeros_like(lhl), torch.zeros_like(lhl)

    init_del_V = torch.zeros((len(hamiltonian.atom_list),), dtype=torch.double)

    err, maxItr = 1e-7, 1000
    offsets = hamiltonian._offsets
    zs, zd = 0., 5.52321494
    corrd = citeCoord2Coord(hamiltonian._offsets, torch.tensor(hamiltonian.get_site_coordinates()))
    Emin = -27
    ul, ur = torch.tensor(0.).requires_grad_(), torch.tensor(2.).requires_grad_()
    N = hamiltonian.basis_size
    n_img = 500

    convergedPotential = SCFIteration(
        N, n_img, err, maxItr, init_del_V, zs, zd, offsets, corrd, Emin, ul, ur,
        hd=h0_bd, hu=hr_bd, hl=hl_bd, sd=sd, su=su, sl=sl, lsd=lsd, lsu=lsu, lsl=lsl, rsd=rsd, rsu=rsu, rsl=rsl, lhd=lhd,
        lhu=lhu, lhl=lhl, rhd=rhd, rhu=rhu, rhl=rhl
    )


if __name__ == '__main__':
    main()