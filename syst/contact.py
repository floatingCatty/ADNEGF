import torch
import xitorch as xt
from typing import Optional, Tuple, Union, List, Dict
from calc.surface_green import selfEnergy
from calc.transport import fermi_dirac, sigmaLR2Gamma, gamma2SigmaLRIn

class Contact(object):
    def __init__(self, hd: torch.Tensor, hu: torch.Tensor, hl: torch.Tensor, u: float, left=True, device='cpu'):
        super(Contact, self).__init__()
        '''
        Here hd, hu, hl is a list
        '''
        self.hd = hd
        self.hu = hu
        self.hl = hl

        self.u = torch.scalar_tensor(u) # electrode potential
        self.left = left
        self.device = device

        if device != 'cpu':
            self.hd = self.hd.to(device)
            self.hu = self.hu.to(device)
            self.hl = self.hl.to(device)

        self._setup_overlap_()


    def _setup_overlap_(self):
        self.sd = torch.eye(self.hd.shape[0], device=self.device)
        self.su = torch.zeros_like(self.hu, device=self.device)
        self.sl = torch.zeros_like(self.hl, device=self.device)

    def calSE(self, eta, ee):
        '''
        :param eta: imaginary part for convergence
        :param ee: energy
        :param requires_lg: decide whether lesser and greater Sigma is required
        '''
        self.self_energy, self.surface_green = selfEnergy(self.hd, self.hu,
            self.hl, self.sd, self.su, self.sl, ee, left=self.left, etaLead=eta, voltage=self.u, device=self.device)

        return self.self_energy, self.surface_green

    def getGamma(self):
        self.gamma = sigmaLR2Gamma(self.self_energy)

        return self.gamma

    def getSigmaIn(self, ee):
        self.sigmaIn = gamma2SigmaLRIn(self.gamma, ee, self.u)

        return self.sigmaIn

    def setU(self, u):
        self.u = u


