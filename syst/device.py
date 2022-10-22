import torch
import xitorch as xt
from xitorch.integrate import quad
from typing import Optional, Tuple, Union, List, Dict
import torch.autograd as autograd
from syst import Contact
from calc.RGF import recursive_gf
from calc.transport import fermi_dirac
from calc.surface_green import selfEnergy
from torchdiffeq import odeint_adjoint, odeint
from Constant import *
from calc.transport import *


class Device(object):
    def __init__(self, hd: torch.Tensor, hu: torch.Tensor, hl: torch.Tensor,
                 dm: Optional[torch.Tensor], Da: Optional[torch.Tensor], Do: Optional[torch.Tensor], a=1., device='cuda'):
        super(Device, self).__init__()
        self.hd = hd
        self.hu = hu
        self.hl = hl
        self.device = device
        if Da is not None:
            self.Da = Da
        if Do is not None:
            self.Do = Do
        self.a = a

        assert len(hu) == len(hl) and len(hu) == len(hd)-1

        self._init_density_(dm)
        self._setup_overlap_()

        self._ee_ = 0

    def _init_density_(self, dm: Union[None, torch.Tensor]):
        if not dm:
            self.dm = dm
        else:
            self.dm = torch.zeros_like(self.hd)

    def _setup_overlap_(self):
        N = len(self.hd)
        sd = torch.eye(self.hd[0].shape[0], device=self.device, dtype=torch.complex128)
        self.sd = [sd for _ in range(N)]
        if N > 1:
            su = torch.zeros_like(self.hu[0], device=self.device, dtype=torch.complex128)
            sl = torch.zeros_like(self.hl[0], device=self.device, dtype=torch.complex128)
            self.su = [su for _ in range(N - 1)]
            self.sl = [sl for _ in range(N - 1)]
        else:
            self.su = []
            self.sl = []



    def update_H(self, hd: torch.Tensor, hu: torch.Tensor, hl: torch.Tensor):
        self.gf_ready = False
        self.hd = hd
        self.hu = hu
        self.hl = hl

    def update_dm(self, dm: torch.Tensor):
        self.dm = dm

    def update_ee(self, ee):
        self._ee_ = ee

    def getGF(self, L_lead: Contact, R_lead: Contact, eta=1e-5, reuseSigmaP=False):
        '''
        :param L_lead: class Contact
        :param R_Lead: class Contact
        :param eta: the imaginary part of energy
        :param reuseSigmaP: ture when sigmaP is calculated before
        :param mode: acoustic or ballistic
        :return:
        '''
        self.__calSigmaLR__(L_lead, R_lead, eta=eta)
        self.__calGamma__()
        self.__calSigmaLRIn__(L_lead, R_lead)
        self.__calSigmaLROut__(L_lead, R_lead)

        self.__calGF__(reuseP=reuseSigmaP, eta=eta)

        self.GFready = True

        return True

    def updateSigmaP(self, rho: Optional, vs: Optional, wo: Optional, mode='Acoustic'):
        '''
        :param mode: Acoustic or Optical
        :reuseGF: True if GF have been already updated
        :return:
        '''

        assert self.GFready == True

        if mode == 'Acoustic':
            self.__calAcousticSigma__(rho, vs)
        elif mode == 'Optical':
            self.__calOpticalSigma__(wo)

    # def selfConsistentScattering(self, L_lead: Contact, R_lead: Contact, eta, rho: Optional, vs: Optional, wo: Optional, converge_err=1e-4, max_iter=100, mode='Acoustic'):
    #     self.__calSigmaLR__(L_lead, R_lead, eta=eta)
    #     self.sigmaPIn, self.sigmaPOut, self.sigmaRP = acousticScatteringIteration.apply(self._ee_, L_lead.u, R_lead.u, self.sigmaL, self.sigmaR, self.hd,
    #                                       self.sd, self.su, self.sl, self.hu, self.hl, self.Da, rho, vs, self.a)

    def calTT(self):
        assert self.GFready

        self.TT = calTT(self.gammaL, self.gammaR, gtrans=self.g_trans)

    def calSeebeck(self):
        assert self.GFready
        self.TT = calTT(self.gammaL, self.gammaR, gtrans=self.g_trans)
        self.seebeck = - (pi**2*k**2*T) / (3*eV) * (autograd.grad(self.TT, self._ee_)[0] / self.TT)


    def calDOS(self):
        assert self.GFready

        self.DOS = (1j * (self.grd[0] - self.grd[0].conj().T)).real.trace()

    def calEqDensity(self, Emin, L_lead, R_lead, rho, vs):
        self.p_eq = calEqDensity(Emin, ul=L_lead.u, ur=R_lead.u, hd=self.hd, hu=self.hu, hl=self.hl,
                     sd=self.sd, su=self.su, sl=self.sl, lhd=L_lead.hd, lhu=L_lead.hu, lhl=L_lead.hl,
                     rhd=R_lead.hd, rhu=R_lead.hu, rhl=R_lead.hl, lsd=L_lead.sd, lsu=L_lead.su, lsl=L_lead.sl,
                     rsd=R_lead.sd, rsu=R_lead.su, rsl=R_lead.sl, relerr=1e-30)

        return True

    def calNeqDensity(self, L_lead, R_lead, rho, vs):
        N = 0
        for i in self.hd:
            N+=i.shape[0]
        self.p_neq = calNeqDensity(N=N, ul=L_lead.u, ur=R_lead.u, hd=self.hd, hu=self.hu, hl=self.hl,
                     sd=self.sd, su=self.su, sl=self.sl, lhd=L_lead.hd, lhu=L_lead.hu, lhl=L_lead.hl,
                     rhd=R_lead.hd, rhu=R_lead.hu, rhl=R_lead.hl, lsd=L_lead.sd, lsu=L_lead.su, lsl=L_lead.sl,
                     rsd=R_lead.sd, rsu=R_lead.su, rsl=R_lead.sl)

        return True



    def calCurrent(self, L_lead, R_lead):
        self.Current = calCurrent(ul=L_lead.u, ur=R_lead.u, hd=self.hd, hu=self.hu, hl=self.hl, sd=self.sd, su=self.su, sl=self.sl,
                   lhd=L_lead.hd, lhu=L_lead.hu, lhl=L_lead.hl, rhd=R_lead.hd, rhu=R_lead.hu, rhl=R_lead.hl, lsd=L_lead.sd,
                   lsu=L_lead.su, lsl=L_lead.sl, rsd=R_lead.sd, rsu=R_lead.su, rsl=R_lead.sl)
        return True


    def calNDensity(self, ee):
        pass




    # privite function
    def __calSigmaLR__(self, L_lead, R_lead, eta):
        self.sigmaL, _ = L_lead.calSE(eta, self._ee_)
        self.sigmaR, _ = R_lead.calSE(eta, self._ee_)

    def __calGamma__(self):
        self.gammaL = sigmaLR2Gamma(self.sigmaL)
        self.gammaR = sigmaLR2Gamma(self.sigmaR)

    def __calSigmaLRIn__(self, L_lead, R_lead):
        self.sigmaLIn = gamma2SigmaLRIn(self.gammaL, self._ee_, L_lead.u)
        self.sigmaRIn = gamma2SigmaLRIn(self.gammaR, self._ee_, R_lead.u)

    def __calSigmaLROut__(self, L_lead, R_lead):
        self.sigmaLOut = gamma2SigmaLROut(self.gammaL, self._ee_, L_lead.u)
        self.sigmaROut = gamma2SigmaLROut(self.gammaR, self._ee_, R_lead.u)

    # def __calAcousticSigma__(self, rho, vs):
    #     self.sigmaPIn, self.sigmaPOut, self.sigmaRP = acousticSigma(self.Da, rho, vs, 1, self.sd, self.gnd, self.gpd, self.grd)

    def __calOpticalSigma__(self, wo):
        pass

    def __getNDensityfn__(self):
        pass

    def __getPDensityfn__(self):
        pass

    def __getOpticalDn__(self, vs, rho):
        '''
        :param vs: sound velocity
        :param rho: mass density
        '''
        assert self.Da is not None
        temp = (self.Da**2) * (k*T) / (vs*rho)
        self.Dn = [temp * self.sd[i] for i in range(len(self.sd))]
        # whether the wf overlap is the overlap of hd should be checked further

        return self.Dn

    def __calGF__(self, reuseP=True, eta=1e-5):
        if reuseP:
            sigmaIn = self.sigmaPIn
            sigmaIn[0] = sigmaIn[0] + self.sigmaLIn
            sigmaIn[-1] = sigmaIn[-1] + self.sigmaRIn

            sigmaOut = self.sigmaPOut
            sigmaOut[0] = sigmaOut[0] + self.sigmaLOut
            sigmaOut[-1] = sigmaOut[-1] + self.sigmaROut
        else:
            sigmaIn = [torch.zeros_like(self.hd[i], dtype=torch.complex128) for i in range(len(self.hd))]
            sigmaIn[0] = sigmaIn[0] + self.sigmaLIn
            sigmaIn[-1] = sigmaIn[-1] + self.sigmaRIn

            sigmaOut = [torch.zeros_like(self.hd[i], dtype=torch.complex128) for i in range(len(self.hd))]
            sigmaOut[0] = sigmaOut[0] + self.sigmaLOut
            sigmaOut[-1] = sigmaOut[-1] + self.sigmaROut
            self.sigmaRP = None

        if not isinstance(sigmaIn, list) and not isinstance(sigmaOut, list):
            self.g_trans, self.grd, self.grl, self.gru, self.gr_left \
                = recursive_gf(self._ee_, self.hl, self.hd, self.hu, self.sd, self.su, self.sl, left_se=self.sigmaL,
                                                           right_se=self.sigmaR, seP=self.sigmaRP, s_in=sigmaIn, s_out=sigmaOut, eta=eta)

        elif isinstance(sigmaIn, list) and not isinstance(sigmaOut, list):
            self.g_trans, self.grd, self.grl, self.gru, self.gr_left, \
            self.gnd, self.gnl, self.gnu, self.gin_left \
                = recursive_gf(self._ee_, self.hl, self.hd, self.hu, self.sd, self.su, self.sl, left_se=self.sigmaL,
                                                           right_se=self.sigmaR, seP=self.sigmaRP, s_in=sigmaIn, s_out=sigmaOut, eta=eta)

        elif not isinstance(sigmaIn, list) and isinstance(sigmaOut, list):
            self.g_trans, self.grd, self.grl, self.gru, self.gr_left, \
            self.gpd, self.gpl, self.gpu, self.gip_left \
                = recursive_gf(self._ee_, self.hl, self.hd, self.hu, self.sd, self.su, self.sl, left_se=self.sigmaL,
                                                           right_se=self.sigmaR, seP=self.sigmaRP, s_in=sigmaIn, s_out=sigmaOut, eta=eta)

        else:
            self.g_trans, self.grd, self.grl, self.gru, self.gr_left, \
            self.gnd, self.gnl, self.gnu, self.gin_left, \
            self.gpd, self.gpl, self.gpu, self.gip_left \
                = recursive_gf(self._ee_, self.hl, self.hd, self.hu, self.sd, self.su, self.sl, left_se=self.sigmaL,
                                                           right_se=self.sigmaR, seP=self.sigmaRP, s_in=sigmaIn, s_out=sigmaOut, eta=eta)

