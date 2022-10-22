from calc import NEGF
import torch.random as random
from TB.hamiltonian import Hamiltonian
import torch

def pack(**options):
    return options

class Transport_tb(object):
    def __init__(self, xyz, xyz_new, nn_distance, comp_overlap, period, *args, **Options):
        super(Transport_tb, self).__init__()
        self.h = Hamiltonian(xyz=xyz, xyz_new=xyz_new, nn_distance=nn_distance, comp_overlap=comp_overlap, **Options)
        self.period = period
        self.comp_overlap = comp_overlap
        self.hDD = torch.tensor(0.)
        self.refresh(True)


    def refresh(self, init=False):
        self.h.initialize()
        self.h.set_periodic_bc(self.period)
        if self.comp_overlap:
            # if init:
            self.hL, self.hD, self.hR, self.sL, self.sD, self.sR = self.h.get_hamiltonians()
            # print(self.hDD - hD)
            hl_list, hd_list, hr_list, sl_list, sd_list, sr_list, subblocks = \
                self.h.get_hamiltonians_block_tridiagonal(optimized=True)
            hmt_ovp = pack(hd=hd_list,
                           hu=hr_list,
                           hl=hl_list,
                           sd=sd_list,
                           su=sr_list,
                           sl=sl_list,

                           lhd=self.hD,
                           lhu=self.hL.conj().T,
                           lsd=self.sD,
                           lsu=self.sL.conj().T,

                           rhd=self.hD,
                           rhu=self.hR,
                           rsd=self.sD,
                           rsu=self.sR)
        else:
            # if init:
            self.hL, self.hD, self.hR = self.h.get_hamiltonians()

            hl_list, hd_list, hr_list, subblocks = \
            self.h.get_hamiltonians_block_tridiagonal(optimized=True)

            hmt_ovp = pack(hd=hd_list,
                           hu=hr_list,
                           hl=hl_list,

                           lhd=self.hD,
                           lhu=self.hL.conj().T,

                           rhd=self.hD,
                           rhu=self.hR)
        self.negf = NEGF(hmt_ovp, dtype=torch.complex128)

    def fluctuate(self, eps, atom_type=None):
        disposition = []


        self.h._atom_list = self.h.atom_list_eq.copy()
        keys = list(self.h._atom_list.keys())

        # temp
        keys = ['Au'+str(i+1) for i in range(len(keys))]
        for k in range(len(keys)):
            disposition.append((torch.rand(3) - 0.5) * 2 * eps * 1.44)
            disposition[-1][2] = 0
            self.h._atom_list[keys[k]] = self.h._atom_list[keys[k]] + disposition[-1]

        self.refresh()

        return torch.stack(disposition)
