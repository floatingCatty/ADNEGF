import torch
from torch.optim import LBFGS, Adam
from xitorch.linalg.solve import solve
from xitorch.grad.jachess import jac
from Constant import *


# def SCFIteration(basis_size, init_del_V, zs, zd, offsets, coord, ul, ur, n_img=1000, err=1e-13, maxIter=100, d_trans=1,  Emin=-25, method='PDIIS', **hmt_ovp):
#     '''
#     :param basis_size: total hamiltonian basis size
#     :param n_img: number of image electrons added in calculation of the potenial
#     :param init_del_V: initial value for residule part of V, defaulf: 0 tensor of atom size
#     :param err: maximal deviation allowed for SCF iteration converging
#     :param maxItr: maximal Iteration number for SCF computation if it not yet converges
#     :param zs: source coordinate of z-axis
#     :param zd: drain cooridnate of z-axis
#     :param offsets: denote the start site of each atom in TB hamiltonian
#     :param coord: denote the spatial coordinate of each atom
#     :param Emin: minimum energy as the start point for pole summation of Non-equilibrium density
#     :param ul: chemical potential(Voltage) of left contact
#     :param ur: chemical potential(Voltage) of right contact
#     :param params: hamiltonians and overlaps of leads and device
#     :return: The (converged) potential
#     '''
#     kBT = k * T / eV
#     if isinstance(ul, (float, int)):
#         ul = torch.scalar_tensor(ul)
#     if isinstance(ur, (float, int)):
#         ur = torch.scalar_tensor(ur)

#     xl = min(ur, ul)
#     xu = max(ur, ul)

#     pole, residue = pole_maker(Emin, ChemPot=float(xl.data) - 4*kBT, kT=kBT, reltol=1e-15)
#     pole = torch.tensor(pole, dtype=torch.complex128)

#     if xl == xu:
#         return init_del_V

#     # pole_0, residue_0 = pole_maker(Emin, ChemPot=float(xl.data), kT=k * T / eV, reltol=1e-15)
#     rho0 = calEqDensity(pole, residue, basis_size, torch.tensor(0.), torch.tensor(0.), **hmt_ovp)

#     del_V_drop = calVdrop(ul, coord[:, d_trans], zs, zd, ur)
#     fn = lambda x, *params: potential2potential(x, basis_size=basis_size, offsets=offsets, Emin=Emin,
#                                                 zd=zd, zs=zs, d_trans=d_trans, n_img=n_img, pole=pole,
#                                                 residue=residue, *params)
#     params = [ul, ur, rho0, del_V_drop]
#     dic = {}
#     for p, v in hmt_ovp.items():
#         if isinstance(v, torch.Tensor):
#             dic[p] = len(params)
#             params.append(v)

#         elif isinstance(v, (list, tuple)):
#             dic[p] = len(params)
#             params += list(v)
#             dic[p] = (dic[p], len(params))



#     def potential2potential(del_V, *params, **options):
#         hd_ = attachPotential(options['offsets'], params[dic['hd'][0]:dic['hd'][1]], del_V, options['basis_size'])
#         rho_eq = calEqDensity(options['pole'], options['residue'], options['basis_size'], ul=params[0], ur=params[1], hd=hd_,
#                               hu=params[dic['hu'][0]:dic['hu'][1]], hl=params[dic['hl'][0]:dic['hl'][1]],
#                               sd=params[dic['sd'][0]:dic['sd'][1]], su=params[dic['su'][0]:dic['su'][1]],
#                               sl=params[dic['sl'][0]:dic['sl'][1]], lhd=params[dic['lhd']], lhu=params[dic['lhu']],
#                             lsd=params[dic['lsd']], lsu=params[dic['lsu']],
#                               rhd=params[dic['rhd']], rhu=params[dic['rhu']],
#                               rsd=params[dic['rsd']], rsu=params[dic['rsu']])
#         rho_neq = calNeqDensity(params[0], params[1], hd=hd_,
#                                 hu=params[dic['hu'][0]:dic['hu'][1]], hl=params[dic['hl'][0]:dic['hl'][1]],
#                                 sd=params[dic['sd'][0]:dic['sd'][1]], su=params[dic['su'][0]:dic['su'][1]],
#                                 sl=params[dic['sl'][0]:dic['sl'][1]], lhd=params[dic['lhd']], lhu=params[dic['lhu']],
#                                 lsd=params[dic['lsd']], lsu=params[dic['lsu']],
#                                rhd=params[dic['rhd']], rhu=params[dic['rhu']],
#                                 rsd=params[dic['rsd']], rsu=params[dic['rsu']])

#         del_rho = rho_eq + rho_neq - params[2]
#         # transcript into xyz coordinate
#         del_rho = getxyzdensity(offset=options['offsets'], siteDensity=del_rho)
#         del_V_dirichlet = density2Potential.apply(coord, del_rho, options['n_img'],
#                                                   options['zd']-options['zs'], options['d_trans'])
#         del_V_ = del_V_dirichlet + params[3]

#         # print(del_V_)

#         return del_V_

#     return _SCF.apply(fn, init_del_V, {}, maxIter, err, method, *params)

class _SCF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fcn, x0, options, maxIter=100, err=1e-7, method='PDIIS', *params):
        # with torch.no_grad():
        #     x_ = fcn(x0, *params)

        x_ = fcn(x0, *params)

        if method == "default":
            it = 0
            old_x = x0
            while (x_-old_x).norm() > err and it < maxIter:
                it += 1
                old_x = x_
                x_ = fcn(x_, *params)

        elif method == 'GD':
            x_ = x_.detach().requires_grad_()
            temp_p = [p.detach() for p in params]
            it = 0
            loss = 1

            def new_fcn(x_):

                loss = (x_ - fcn(x_, *temp_p)).abs().sum()
                print(loss)
                return loss

            with torch.enable_grad():
                while it < maxIter and loss > err:
                    it += 1
                    loss = new_fcn(x_)
                    x_ = x_ - 1e-3 * torch.autograd.grad(loss, (x_,))[0]

        elif method == 'Adam':
            # x = torch.randn(200,1, dtype=torch.float64)
            # x = x / x.norm()
            # x_ = x_.unsqueeze(1) @ x.T
            x_ = x_.detach().requires_grad_()
            temp_p = [p.detach() for p in params]
            optim = Adam(params=[x_], lr=1e-3)
            def new_fcn(x_):
                loss = (x_ - fcn(x_, *temp_p)).norm()
                print(loss)
                return loss
            i = 0
            loss = 1
            with torch.enable_grad():
                while i < maxIter and loss > err:
                    optim.zero_grad()
                    loss = new_fcn(x_)
                    loss.backward()
                    optim.step()


        elif method == "PDIIS":
            with torch.no_grad():
                x_ = PDIIS(lambda x: fcn(x, *params), p0=x_, maxIter=maxIter, **options)

        elif method == 'LBFGS':
            x_ = x_.detach().requires_grad_()
            temp_p = [p.detach() for p in params]
            optim = LBFGS(params=[x_], lr=1e-2)

            def new_fcn():
                optim.zero_grad()
                loss = (x_ - fcn(x_, *temp_p)).norm()
                loss.backward()
                print(loss)
                return loss

            with torch.enable_grad():
                for i in range(maxIter):
                    optim.step(new_fcn)
                    print(x_)

        else:
            raise ValueError

        print("Convergence achieved !")
        x_ = x_ + 0j
        ctx.save_for_backward(x_, *params)
        ctx.fcn = fcn

        return x_

    @staticmethod
    def backward(ctx, grad_outputs):
        x_ = ctx.saved_tensors[0].detach().requires_grad_()
        params = ctx.saved_tensors[1:]

        idx = [i for i in range(len(params)) if params[i].requires_grad]


        fcn = ctx.fcn
        def new_fcn(x, *params):
            return x - fcn(x, *params)

        with torch.enable_grad():
            grad = jac(fcn=new_fcn, params=(x_, *params), idxs=[0])[0]

        # pre = solve(grad.H, -grad_outputs.reshape(-1, 1))
        pre = solve(grad.H, -grad_outputs.reshape(-1, 1).type_as(x_))
        pre = pre.reshape(grad_outputs.shape)


        with torch.enable_grad():
            params_copy = [p.detach().requires_grad_() for p in params]
            yfcn = new_fcn(x_, *params_copy)

        grad = torch.autograd.grad(yfcn, [params_copy[i] for i in idx], grad_outputs=pre,
                                   create_graph=torch.is_grad_enabled(),
                                   allow_unused=True)
        grad_out = [None for _ in range(len(params))]
        for i in range(len(idx)):
            grad_out[idx[i]] = grad[i]


        return None, None, None, None, None, None, *grad_out


def PDIIS(fn, p0, a=0.05, n=6, maxIter=100, k=3, err=1e-6, relerr=1e-3, **options):
    """The periodic pully mixing from https://doi.org/10.1016/j.cplett.2016.01.033.

    Args:
        fn (function): the iterative functions
        p0 (_type_): the initial point
        a (float, optional): the mixing beta value, or step size. Defaults to 0.05.
        n (int, optional): the size of the storage of history to compute the pesuedo hessian matrix. Defaults to 6.
        maxIter (int, optional): the maximum iteration. Defaults to 100.
        k (int, optional): the period of conducting pully mixing. The algorithm will conduct pully mixing every k iterations. Defaults to 3.
        err (_type_, optional): the absolute err tolerance. Defaults to 1e-6.
        relerr (_type_, optional): the relative err tolerance. Defaults to 1e-3.

    Returns:
        p _type_: the stable point
    """
    i = 0
    f = fn(p0) - p0
    p = p0
    R = [None for _ in range(n)]
    F = [None for _ in range(n)]
    # print("SCF iter 0 abs err {0} | rel err {1}: ".format( 
    #         f.abs().max().detach().numpy(), 
    #         (f.abs() / p.abs()).max().detach().numpy())
    #         )
    while (f.abs().max() > err or (f.abs() / p.abs()).max() > relerr) and i < maxIter:
        if not (i+1) % k:
            F_ = torch.stack([t for t in F if t != None])
            R_ = torch.stack([t for t in R if t != None])
            p_ = p + a*f - (R_.T+a*F_.T)@(F_ @ F_.T).inverse() @ F_ @ f
        else:
            p_ = p + a * f

        f_ = fn(p_) - p_
        F[i % n] = f_ - f
        R[i % n] = p_ - p

        p = p_.clone()
        f = f_.clone()
        i += 1

        # print("SCF iter {0} abs err {1} | rel err {2}: ".format(
        #     i, 
        #     f.abs().max().detach().numpy(), 
        #     (f.abs() / p.abs()).max().detach().numpy())
        #     )


    if i == maxIter:
        print("Not Converged very well at {0}.".format(i))
    else:
        print("Converged very well at {0}.".format(i))


    return p


