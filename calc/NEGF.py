from typing import List
import torch
from calc.pole_summation import pole_maker
from calc.RGF import recursive_gf
from calc.surface_green import selfEnergy
from calc.utils import quad
from calc.poisson import density2Potential, getImg
from calc.SCF import _SCF
from Constant import *
import torch.optim as optim
from tqdm import tqdm
import numpy as np

kBT = k * T / eV


def fermi_dirac(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(x / kBT))


class _fermi_dirac(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kBT):
        ctx.save_for_backward(x)
        if T == 0 and x == 0:
            return (torch.sign(x) + 1) / 2


class NEGF(object):
    def __init__(self, hmt_ovp: dict, dtype=torch.complex128, device='cpu', **options):
        '''

        :param hmt_ovp: transport system description, should be a dict form, containing:
                ['id': the identity of the computed system, str
                'hd':list of Tensors
                'hu':list of Tensors, the last item is the coupling matrix with right leads
                'hl':list of Tensors, the last item is the coupling matrix with left leads,
                'lhd':
                'lhu':
                'rhd':
                'rhu':
                overlap is labeled as 'sd','lsd'... as such
                ]
        :param dtype:
        :param device:
        :param options:
        '''
        self.hmt_ovp = hmt_ovp
        self.basis_size = 0
        self.hd_shape = []
        self.hu_shape = []
        self.hl_shape = []
        for i, hd in enumerate(hmt_ovp['hd']):
            self.basis_size += hd.shape[0]
            self.hd_shape.append(hd.shape)
            self.hu_shape.append(hmt_ovp['hu'][i].shape)
            self.hl_shape.append(hmt_ovp['hl'][i].shape)
        self.leftScatterSize = hmt_ovp['hd'][0].shape
        self.rightScatterSize = hmt_ovp['hd'][-1].shape
        self.options = options
        self.device = torch.device(device)
        self.dtype = dtype
        self.V_ext = {}
        self.green = {}
        self.SE = {}

        self.initialize()

    def initialize(self):
        '''
        1. transfer all parameters to the proper devices
        2. load saved system files
        3. initialize overlap if not containing in hmt_ovp
        '''
        # self.saved_file = torch.load(...)
        # self.V_ext = self.saved_file['V_ext']

        hs_keys = list(self.hmt_ovp.keys())
        if "sd" not in hs_keys:
            self.hmt_ovp.update({"sd": [torch.eye(i[0], dtype=self.dtype, device=self.device) for i in self.hd_shape]})
        if "su" not in hs_keys:
            self.hmt_ovp.update({"su": [torch.zeros(i, dtype=self.dtype, device=self.device) for i in self.hu_shape]})
        if "sl" not in hs_keys:
            self.hmt_ovp.update({"sl": [torch.zeros(i, dtype=self.dtype, device=self.device) for i in self.hl_shape]})
        if "lsd" not in hs_keys:
            self.hmt_ovp.update({"lsd": torch.eye(self.hmt_ovp['lhd'].shape[0], dtype=self.dtype, device=self.device)})
        if "lsu" not in hs_keys:
            self.hmt_ovp.update({"lsu": torch.zeros(self.hmt_ovp['lhu'].shape, dtype=self.dtype, device=self.device)})
        if "rsd" not in hs_keys:
            self.hmt_ovp.update({"rsd": torch.eye(self.hmt_ovp['rhd'].shape[0], dtype=self.dtype, device=self.device)})
        if "rsu" not in hs_keys:
            self.hmt_ovp.update({"rsu": torch.zeros(self.hmt_ovp['rhu'].shape, dtype=self.dtype, device=self.device)})

    def load_data(self, ul: torch.Tensor, ur: torch.Tensor):
        self.green, self.SE = self.saved_file[(ul, ur)]['green'], self.saved_file[(ul, ur)]['SE']

    def compute_selfEnergy(self, ee: List[torch.Tensor], ul: torch.Tensor, ur: torch.Tensor, etaLead : float = 1e-5, cutoff: bool =True, method: str = 'Sancho-Rubio'):
        seL_list = []
        seR_list = []

        if cutoff:
            l_coup_u = self.hmt_ovp['hl'][-1].conj().T
            r_coup_u = self.hmt_ovp['hu'][-1]
            l_ovp_u = self.hmt_ovp['sl'][-1].conj().T
            r_ovp_u = self.hmt_ovp['su'][-1]
            for e in tqdm(ee, desc="Compute Self-Energy: "):
                seL, _ = selfEnergy(hd=self.hmt_ovp['lhd'], hu=self.hmt_ovp['lhu'], sd=self.hmt_ovp['lsd'],
                                    su=self.hmt_ovp['lsu'], coup_u=l_coup_u, ovp_u=l_ovp_u, ee=e, left=True, voltage=ul,
                                    etaLead=etaLead, method=method)
                seR, _ = selfEnergy(hd=self.hmt_ovp['rhd'], hu=self.hmt_ovp['rhu'], sd=self.hmt_ovp['rsd'],
                                    su=self.hmt_ovp['rsu'], coup_u=r_coup_u, ovp_u=r_ovp_u, ee=e, left=False,
                                    voltage=ur, etaLead=etaLead, method=method)
                seL_list.append(seL)
                seR_list.append(seR)
        else:
            for e in tqdm(ee, desc="Compute Self-Energy: "):
                seL, _ = selfEnergy(hd=self.hmt_ovp['lhd'], hu=self.hmt_ovp['lhu'], sd=self.hmt_ovp['lsd'],
                                    su=self.hmt_ovp['lsu'], ee=e, left=True, voltage=ul, etaLead=etaLead,
                                    method=method)
                seR, _ = selfEnergy(hd=self.hmt_ovp['rhd'], hu=self.hmt_ovp['rhu'], sd=self.hmt_ovp['rsd'],
                                    su=self.hmt_ovp['rsu'], ee=e, left=False, voltage=ur, etaLead=etaLead,
                                    method=method)
                seL_list.append(seL)
                seR_list.append(seR)
        seL_list = torch.stack(seL_list)
        seR_list = torch.stack(seR_list)

        return seL_list, seR_list

    def SCF(self, ul: torch.Tensor, ur: torch.Tensor, atom_coord: torch.Tensor, d_trains: int, left_pos, right_pos, offset, del_V0=None,
            n_int_neq=100, poissonMethod="image charge", sgfMethod='Sancho-Rubio', n_img=200, Emin=-25, etaLead=1e-5,
            etaDevice=0, maxIter=100, conv_err=1e-7, cutoff=True, SEpreCal=True, **SCFOptions):
        # return a V_ext under bias voltage
        if isinstance(ul, (float, int)):
            ul = torch.scalar_tensor(ul, device=self.device)
        if isinstance(ur, (float, int)):
            ur = torch.scalar_tensor(ur, device=self.device)
        if ul == ur:
            V_ext = torch.zeros(len(offset), dtype=torch.float64, device=self.device)
            return V_ext
        if (ul, ur) in list(self.V_ext.keys()):
            return self.V_ext[(ul, ur)]
        if del_V0 is None:
            del_V0 = torch.zeros(len(offset), dtype=torch.float64, device=self.device)

        xl = min(ul, ur)
        xu = max(ul, ur)

        # generating energy point for pole summation and Neq Density calculation
        pole, residue = pole_maker(Emin, ChemPot=float(xl.data) - 4 * kBT, kT=kBT, reltol=1e-15)

        if SEpreCal:
            xlg, wlg = np.polynomial.legendre.leggauss(n_int_neq)
            xlg = torch.tensor(xlg, dtype=xu.dtype, device=xu.device)
            ee_Neq = xlg * (0.5 * (xu + 8 * kBT - xl)) + (0.5 * (xu + xl))

            seL_Neq, seR_Neq = self.compute_selfEnergy(ee_Neq, ul, ur, etaLead=etaLead, cutoff=cutoff,
                                                       method=sgfMethod)  # of Tensor type (N, 2, *)
            seL_pole, seR_pole = self.compute_selfEnergy(pole, ul, ur, etaLead=0., cutoff=cutoff, method=sgfMethod)
        else:
            seL_Neq, seR_Neq = None, None
            seL_pole, seR_pole = None, None

        # generating image charge position
        imgCoord = getImg(n=n_img, coord=atom_coord, d=abs(left_pos - right_pos), dim=d_trains)

        rho0 = self.cal_EqDensity(pole, residue, torch.tensor(0., device=self.device),
                                  torch.tensor(0., device=self.device),
                                  offset=offset, SE_pole_list=(seL_pole, seR_pole), etaDevice=etaDevice, **self.hmt_ovp)
        V_drop = self.calVdrop(ul=ul, ur=ur, left_pos=left_pos, right_pos=right_pos, tCoord=atom_coord[:, d_trains])
        if poissonMethod == 'image charge':
            params = [ul, ur, rho0, V_drop, atom_coord, seL_Neq, seR_Neq, seL_pole, seR_pole]
            dic = {}
            for p, v in self.hmt_ovp.items():
                if isinstance(v, torch.Tensor):
                    dic[p] = len(params)
                    params.append(v)

                elif isinstance(v, (list, tuple)):
                    dic[p] = len(params)
                    params += list(v)
                    dic[p] = (dic[p], len(params))

            fcn = lambda x, *p: self.scfFn_img(x, offset, imgCoord, self.basis_size, pole,
                                               n_img, n_int_neq, residue, dic, etaLead, etaDevice, sgfMethod, *p)
        else:
            raise ValueError

        V = _SCF.apply(fcn, del_V0, SCFOptions, maxIter, conv_err, "PDIIS", *params)
        self.V_ext.update({(float(ul), float(ur)): V})

        return V

    def scfFn_img(self, del_V, offset, imgCoord, basis_size, pole, n_img, n_int_neq, residue, dic, etaLead, etaDevice,
                  sgfMethod, *params):
        hd_ = self.attachPotential(offset, params[dic['hd'][0]:dic['hd'][1]], del_V,
                                   basis_size)
        rho_eq = self.cal_EqDensity(pole, residue, ul=params[0],
                                    ur=params[1], offset=offset, hd=hd_,
                                    hu=params[dic['hu'][0]:dic['hu'][1]], hl=params[dic['hl'][0]:dic['hl'][1]],
                                    sd=params[dic['sd'][0]:dic['sd'][1]], su=params[dic['su'][0]:dic['su'][1]],
                                    sl=params[dic['sl'][0]:dic['sl'][1]], SE_pole_list=(params[7], params[8]),
                                    sgfMethod=sgfMethod, etaDevice=etaDevice)
        rho_neq = self.cal_NeqDensity(params[0], params[1], offset=offset, hd=hd_,
                                      hu=params[dic['hu'][0]:dic['hu'][1]], hl=params[dic['hl'][0]:dic['hl'][1]],
                                      sd=params[dic['sd'][0]:dic['sd'][1]], su=params[dic['su'][0]:dic['su'][1]],
                                      sl=params[dic['sl'][0]:dic['sl'][1]], SE_list=(params[5], params[6]),
                                      n_int=n_int_neq,
                                      sgfMethod=sgfMethod, etaDevice=etaDevice, etaLead=etaLead)
        del_rho = rho_eq + rho_neq - params[2]
        # transcript into xyz coordinate
        del_V_dirichlet = density2Potential.apply(imgCoord, params[4], del_rho, n_img)
        del_V_ = del_V_dirichlet + params[3]

        return del_V_

    def p2V(self, density, atom_coord, d_trains, method='image charge', **options):
        if method == 'image charge':
            n_img = options['n_img']
            d_img = options['d_img']
            potential = density2Potential.apply(options['imgCoord'],
                                                atom_coord, density, n=n_img, d=d_img, d_trans=d_trains)
        else:
            raise ValueError

        return potential

    def cal_EqDensity(self, pole, residue, ur, ul, offset, SE_pole_list=None, etaDevice=0., sgfMethod='Sancho-Rubio',
                      **hmt_ovp):
        N_pole = len(pole)
        eq = torch.zeros(self.basis_size, dtype=torch.float64, device=self.device)

        if SE_pole_list is not None:
            (seL_list, seR_list) = SE_pole_list
            for i, seL in enumerate(seL_list):
                _, grd, _, _, _ = recursive_gf(pole[i], hl=hmt_ovp['hl'], hd=hmt_ovp['hd'], hu=hmt_ovp['hu'],
                                               sd=hmt_ovp['sd'], su=hmt_ovp['su'],
                                               sl=hmt_ovp['sl'], left_se=seL, right_se=seR_list[i], seP=None, s_in=None,
                                               s_out=None, eta=etaDevice)
                eq = eq - residue[i] * torch.cat([i.diag() for i in grd], dim=0)
        else:
            l_coup_u = self.hmt_ovp['hl'][-1].conj().T
            r_coup_u = self.hmt_ovp['hu'][-1]
            l_ovp_u = self.hmt_ovp['sl'][-1].conj().T
            r_ovp_u = self.hmt_ovp['su'][-1]
            for i in range(N_pole):
                seL, _ = selfEnergy(hd=hmt_ovp['lhd'], hu=hmt_ovp['lhu'], sd=hmt_ovp['lsd'],
                                    su=hmt_ovp['lsu'], ee=pole[i], coup_u=l_coup_u, ovp_u=l_ovp_u,
                                    method=sgfMethod, left=True, voltage=ul, etaLead=0.)
                seR, _ = selfEnergy(hd=hmt_ovp['rhd'], hu=hmt_ovp['rhu'], sd=hmt_ovp['rsd'],
                                    su=hmt_ovp['rsu'], ee=pole[i], coup_u=r_coup_u, ovp_u=r_ovp_u,
                                    method=sgfMethod, left=False, voltage=ur, etaLead=0.)

                _, grd, _, _, _ = recursive_gf(pole[i], hl=hmt_ovp['hl'], hd=hmt_ovp['hd'], hu=hmt_ovp['hu'],
                                               sd=hmt_ovp['sd'],
                                               su=hmt_ovp['su'],
                                               sl=hmt_ovp['sl'], left_se=seL, right_se=seR, seP=None, s_in=None,
                                               s_out=None, eta=etaDevice)
                eq = eq - residue[i] * torch.cat([i.diag() for i in grd], dim=0)

        self.rho_eq = torch.zeros((len(offset),), dtype=torch.complex128, device=self.device)
        for i in range(len(offset) - 1):
            self.rho_eq[i] += eq[offset[i]:offset[i + 1]].sum()
        self.rho_eq[-1] += eq[offset[-1]:].sum()

        return 2 * self.rho_eq.imag

    def cal_NeqDensity(self, ul, ur, offset, SE_list=None, n_int=100, etaLead=1e-5, etaDevice=0.,
                       sgfMethod='Sancho-Rubio', **hmt_ovp):
        # n_int use when SE_list is not None
        xl = min(ul, ur) - 4 * kBT
        xu = max(ul, ur) + 4 * kBT
        neq = torch.zeros(self.basis_size, dtype=torch.float64, device=self.device)

        dic = {}
        if SE_list is None:
            l_coup_u = self.hmt_ovp['hl'][-1].conj().T
            r_coup_u = self.hmt_ovp['hu'][-1]
            l_ovp_u = self.hmt_ovp['sl'][-1].conj().T
            r_ovp_u = self.hmt_ovp['su'][-1]
            params = [l_coup_u, l_ovp_u, r_coup_u, r_ovp_u]
            for p, v in hmt_ovp.items():
                if isinstance(v, torch.Tensor):
                    dic[p] = len(params)
                    params.append(v)

                elif isinstance(v, (list, tuple)):
                    dic[p] = len(params)
                    params += list(v)
                    dic[p] = (dic[p], len(params))

            # params = [i for i in params if isinstance(i, torch.Tensor)]
            def fn(ee, *params):
                seL, _ = selfEnergy(hd=params[dic['lhd']], hu=params[dic['lhu']],
                                    sd=params[dic['lsd']],
                                    su=params[dic['lsu']], coup_u=params[0], ovp_u=params[1], ee=ee,
                                    method=sgfMethod, left=True, voltage=ul, etaLead=etaLead)
                seR, _ = selfEnergy(hd=params[dic['rhd']], hu=params[dic['rhu']],
                                    sd=params[dic['rsd']],
                                    su=params[dic['rsu']], coup_u=params[2], ovp_u=params[3], ee=ee,
                                    method=sgfMethod, left=False, voltage=ur, etaLead=etaLead)
                _, grd, _, _, _ = recursive_gf(ee, hl=params[dic['hl'][0]:dic['hl'][1]],
                                               hd=params[dic['hd'][0]:dic['hd'][1]],
                                               hu=params[dic['hu'][0]:dic['hu'][1]],
                                               sd=params[dic['sd'][0]:dic['sd'][1]],
                                               su=params[dic['su'][0]:dic['su'][1]],
                                               sl=params[dic['sl'][0]:dic['sl'][1]], left_se=seL, right_se=seR,
                                               seP=None, s_in=None,
                                               s_out=None, eta=etaDevice)
                dp_neq = torch.cat([-2 * i.diag() for i in grd], dim=0)
                return dp_neq.imag

            neq = quad(fcn=fn, xl=xl, xu=xu, params=params, n=n_int)
        else:
            (seL_list, seR_list) = SE_list
            n = seL_list.shape[0]
            xlg, wlg = np.polynomial.legendre.leggauss(n)
            ndim = len(xu.shape)
            xlg = torch.tensor(xlg, dtype=xu.dtype, device=xu.device)[(...,) + (None,) * ndim]  # (n, *nx)
            wlg = torch.tensor(wlg, dtype=xu.dtype, device=xu.device)[(...,) + (None,) * ndim]  # (n, *nx)
            wlg *= 0.5 * (xu - xl)
            xs = xlg * (0.5 * (xu - xl)) + (0.5 * (xu + xl))  # (n, *nx)
            for i, seL in enumerate(seL_list):
                _, grd, _, _, _ = recursive_gf(xs[i], hl=hmt_ovp['hl'], hd=hmt_ovp['hd'], hu=hmt_ovp['hu'],
                                               sd=hmt_ovp['sd'],
                                               su=hmt_ovp['su'],
                                               sl=hmt_ovp['sl'], left_se=seL, right_se=seR_list[i], seP=None,
                                               s_in=None,
                                               s_out=None, eta=etaDevice)
                neq += wlg[i] * torch.cat([-2 * i.diag() for i in grd], dim=0).imag

        self.rho_neq = torch.zeros((len(offset),), dtype=torch.float64, device=self.device)
        for i in range(len(offset) - 1):
            self.rho_neq[i] += neq[offset[i]:offset[i + 1]].sum()
        self.rho_neq[-1] += neq[offset[-1]:].sum()

        return self.rho_neq / (2 * pi)

    def calVdrop(self, ul, tCoord, left_pos, right_pos, ur):
        return ul + (ur - ul) * (tCoord - left_pos) / (right_pos - left_pos)

    def attachPotential(self, offset, hd, V, basis_size):
        offset_ = list(offset) + [basis_size]
        site_V = torch.cat([V[i].repeat(offset_[i + 1] - offset_[i]) for i in range(len(offset_) - 1)], dim=0)
        start = 0
        hd_V = []
        for i in range(len(hd)):
            hd_V.append(hd[i] - torch.diag(site_V[start:start + len(hd[i])]))
            start = start + len(hd[i])

        return hd_V

    def attachDop(self, dop, hd):
        h = []
        id = 0
        for hd_b in hd:
            h.append(hd_b + dop[id:id + hd_b.shape[0]])
            id = id + hd_b.shape[0]
        return h

    def calGreen(self, ee, ul, ur, etaLead=1e-5, etaDevice=0., cutoff=True, ifSCF=False, calDOS=False, calTT=False,
                 sgfMethod='Sancho-Rubio', calSeebeck=False, **Options):
        # load parameters:
        # if Vext of such bias is already exist, then there is not need to do SCF, in that case Options can have no args
        # otherwise, it should contains the required parameters for scf computation.
        '''
        Compute Green Function and saved in self.ans, whether to use SCF calculation is decided by the user
        :param ee: energy point, can be a scalar or a tensor
        :param ul:
        :param ur:
        :param eta:
        :param ifSCF:
        :param Options: recommend atom_coord: torch.Tensor, d_trains: int, left_pos, right_pos, offset to be included
        :return:
        '''
        if isinstance(ul, (float, int)):
            ul = torch.scalar_tensor(ul, device=self.device)
        if isinstance(ur, (float, int)):
            ur = torch.scalar_tensor(ur, device=self.device)

        if isinstance(ee, list):
            ee = torch.tensor(ee, dtype=self.dtype, device=self.device)
        elif isinstance(ee, (float, int, complex)):
            ee = torch.tensor([ee])
        elif isinstance(ee, torch.Tensor):
            ee = ee.reshape(-1)
        else:
            raise TypeError

        if calSeebeck:
            ee.requires_grad_()

        # start = time.time()

        if len(ee) == 1:
            seL, seR = self.compute_selfEnergy([ee], ul, ur, etaLead=etaLead, cutoff=cutoff, method=sgfMethod)
        elif len(ee) > 1:
            seL, seR = self.compute_selfEnergy(ee, ul, ur, etaLead=etaLead, cutoff=cutoff, method=sgfMethod)
        else:
            raise TypeError

        k = (float(ul), float(ur))
        if k in list(self.V_ext.keys()):
            V = self.V_ext[k].detach()
        elif not ifSCF:
            V = None
        else:
            V = self.SCF(ul, ur, **Options).detach()

        if V is not None:
            hd_ = self.attachPotential(Options['offset'], self.hmt_ovp['hd'], V, self.basis_size)
        else:
            hd_ = self.hmt_ovp['hd']

        for i, e in tqdm(enumerate(ee), desc="Compute green functions: "):
            if e not in self.green.keys():
                ans = recursive_gf(e, hl=self.hmt_ovp['hl'], hd=hd_, hu=self.hmt_ovp['hu'],
                                   sd=self.hmt_ovp['sd'], su=self.hmt_ovp['su'],
                                   sl=self.hmt_ovp['sl'], left_se=seL[i], right_se=seR[i], seP=None, s_in=None,
                                   s_out=None, eta=etaDevice)
                self.green.update({float(e.data): ans})

        # calculating transport properties

        out = {}
        DOS = []
        TT = []
        for i, e in tqdm(enumerate(ee), desc="Compute Properties: "):
            g_trans, grd, grl, gru, gr_left = self.green[float(e.data)]
            if calDOS:
                DOS.append(self.calDOS(grd))
            if calTT:
                TT.append(self.calTT(seL[i], seR[i], g_trans))

        if calDOS:
            out.update({'DOS': torch.stack(DOS)})
        if calTT:
            out.update({"TT": torch.stack(TT)})
        if calSeebeck:
            out.update({"Seebeck": - torch.autograd.grad(out['TT'].sum(), ee)[0] / (out['TT'] + 1e-8)})

        # end = time.time()
        # print(end-start)

        return out

    def calCurrent_NUM(self, ul, ur, n_int=100, delta=1., **Options):
        '''
        This method does not ensure the necessity of strict formula of gradient, but accurate numerical graident
        :param ul:
        :param ur:
        :param n_int:
        :param Options:
        :param expand range to include full fermi window
        :return:
        '''
        if isinstance(ul, (float, int)):
            ul = torch.scalar_tensor(ul, device=self.device)
        if isinstance(ur, (float, int)):
            ur = torch.scalar_tensor(ur, device=self.device)
        xu = max(ul, ur) + delta
        xl = min(ul, ur) - delta

        xlg, wlg = np.polynomial.legendre.leggauss(n_int)
        ndim = len(xu.shape)
        xlg = torch.tensor(xlg, dtype=xu.dtype, device=xu.device)[(...,) + (None,) * ndim]  # (n, *nx)
        wlg = torch.tensor(wlg, dtype=xu.dtype, device=xu.device)[(...,) + (None,) * ndim]  # (n, *nx)
        wlg *= 0.5 * (xu - xl)
        xs = xlg * (0.5 * (xu - xl)) + (0.5 * (xu + xl))
        TT = self.calGreen(xs, ul, ur, calTT=True, **Options)['TT']
        for i, t in enumerate(TT):
            TT[i] = (fermi_dirac(xs[i] - xu + 1) - fermi_dirac(xs[i] - xl - 1)) * TT[i]
        Current = (TT * wlg).sum() / pi

        return Current

    def calDOS(self, grd):
        dos = 0
        for jj in range(len(grd)):
            temp = grd[jj] @ self.hmt_ovp['sd'][jj]
            dos -= torch.trace(temp.imag) / pi

        return dos

    def calTT(self, seL, seR, gtrans):
        tx, ty = gtrans.shape
        lx, ly = seL.shape
        rx, ry = seR.shape
        x0 = min(lx, tx)
        x1 = min(rx, ty)

        gammaL = torch.zeros(size=(tx, tx), dtype=self.dtype, device=self.device)
        gammaL[:x0, :x0] += self.sigmaLR2Gamma(seL)[:x0, :x0]
        gammaR = torch.zeros(size=(ty, ty), dtype=self.dtype, device=self.device)
        gammaR[-x1:, -x1:] += self.sigmaLR2Gamma(seR)[-x1:, -x1:]

        TT = torch.trace(gammaL @ gtrans @ gammaR @ gtrans.conj().T).real

        return TT

    def optimize(self, fn, variables, target, criteria, step, method, lr=1.):
        '''
        :param fn: process to compute the target quantity
        :param variables: some list parameters that need to be optimized
        :param target: the optimal value of fn's output
        :param criteria: a loss function, scalar type output
        :param step: how many step to do a single optimize, if want to do fitting, set to be a large number
        :param method: "LBFGS" is prefered, "Adam", ""...(black box)
        :return:
        '''

        if method == 'LBFGS':
            optimizer = optim.LBFGS(params=variables, lr=lr)

            def closure():
                optimizer.zero_grad()
                loss = criteria(fn(*variables), target)
                loss.backward()
                print(loss)
                return loss

            for _ in range(step):
                optimizer.step(closure)

        return variables

    def pointWiseTransmissionControl(self, opt_ee_range, step=2, up=True, init_dop=True, Path="../data/dop.pth"):
        if init_dop:
            dop = torch.randn(self.basis_size, dtype=torch.float64, device=self.device)
        else:
            dop = torch.load(Path, map_location=self.device)
        TT = self.calGreen(ee=opt_ee_range, ul=0, ur=0, calTT=True)['TT'].detach()
        dop.requires_grad_()

        def fn(dopping):
            self.hmt_ovp['hd'] = self.attachDop(dopping, self.hmt_ovp['hd'])
            TT = self.calGreen(ee=opt_ee_range, ul=0, ur=0, calTT=True)['TT']
            self.hmt_ovp['hd'] = self.attachDop(-dopping, self.hmt_ovp['hd'])

            return TT

        criteria = torch.nn.MSELoss()
        if up:
            self.optimize(fn=fn, variables=[dop], target=TT + torch.ones_like(TT),
                          criteria=criteria, step=step, method='LBFGS')

        dop = dop.detach()

        torch.save(dop, Path)
        return fn(dop)

    def sigmaLR2Gamma(self, se):
        return -1j * (se - se.conj())


if __name__ == '__main__':
    from ase.build.ribbon import graphene_nanoribbon
    from ase.visualize.plot import plot_atoms
    import seaborn as sns
    from TB import *
    import matplotlib.pyplot as plt

    atoms = graphene_nanoribbon(2.5, 6, type='armchair', saturated=False)
    ax1 = plot_atoms(atoms, show_unit_cell=2, rotation='90x,0y,270z')
    plt.tight_layout()
    plt.show()

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

    set_tb_params(PARAMS_C_C1={'pp_pi': gamma0},
                  PARAMS_C_C2={'pp_pi': gamma1},
                  PARAMS_C_C3={'pp_pi': gamma2},
                  OV_C_C1={'pp_pi': s0},
                  OV_C_C2={'pp_pi': s1},
                  OV_C_C3={'pp_pi': s2})


    # --------------------------- Hamiltonian -------------------------
    def sorting(coords, **kwargs):
        return np.argsort(coords[:, 1], kind='mergesort')


    h = Hamiltonian(xyz=coords, nn_distance=[1.5, 2.5, 3.1], comp_overlap=True, sort_func=sorting)
    # for i in list(h.atom_list.keys()):
    #     h.atom_list[i].requires_grad_()
    h.initialize()
    h.set_periodic_bc(torch.tensor(period))

    hL, hD, hR, sL, sD, sR = h.get_hamiltonians()
    # print(hD[:10,:10].sum())
    # torch.autograd.set_detect_anomaly(True)
    # print(torch.autograd.grad(hD.sum(), h.atom_list['C6']))
    # print("success")

    hl_list, hd_list, hr_list, sl_list, sd_list, sr_list, subblocks = \
        h.get_hamiltonians_block_tridiagonal(optimized=True)


    def pack(**options):
        return options


    hmt_ovp = pack(hd=hd_list,
                   hu=hr_list,
                   hl=hl_list,
                   sd=sd_list,
                   su=sr_list,
                   sl=sl_list,

                   lhd=hD,
                   lhu=hL.conj().T,
                   lsd=sD,
                   lsu=sL.conj().T,

                   rhd=hD,
                   rhu=hR,
                   rsd=sD,
                   rsu=sR)
    negf = NEGF(hmt_ovp, dtype=torch.complex128)

    ee = torch.linspace(-3, 3, 400)
    out = negf.calGreen(
        ee=ee,
        ul=0,
        ur=0.5,
        atom_coord=h.get_site_coordinates()[h._offsets],
        d_trains=1,
        left_pos=period[0][0],
        right_pos=period[0][1],
        offset=h._offsets,
        calDOS=True,
        calTT=True,
        calSeebeck=True,
        etaLead=1e-5,
        etaDevice=0.,
        ifSCF=True,
        n_int_neq=100,
        cutoff=True,
        sgfMethod='Lopez-Schro'
    )

    fig, ax = plt.subplots(2, 1, sharex=True)

    ax[0].plot(ee, out['TT'].detach().numpy(), c=sns.color_palette("Blues")[4])
    ax[0].set_ylabel("T(E)")
    ax[0].set_title("transmission of AGNR(7)")
    ax[0].set_xlim((-3, 3))

    ax[1].plot(ee, out['DOS'].detach().numpy(), c=sns.color_palette("Blues")[4])
    ax[1].set_xlabel("E/ev")
    ax[1].set_ylabel("DOS")
    ax[1].set_title("DOS of AGNR(7)")
    ax[1].set_xlim((-3, 3))
    plt.show()

    plt.plot(out['Seebeck'].detach())
    plt.show()


    I = []
    for i in range(10):
        current = negf.calCurrent_NUM(ul=0, ur=i*0.1, ifSCF=False, atom_coord=h.get_site_coordinates()[h._offsets],
        d_trains=1,
        left_pos=period[0][0],
        right_pos=period[0][1],
        offset=h._offsets)
        I.append(current)
        print(current)

    I = torch.stack(I)
    plt.plot(I)
    plt.show()

