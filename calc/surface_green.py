import torch
import torch.linalg as tLA
from xitorch.linalg.solve import solve
import scipy.linalg as SLA
import matplotlib.pyplot as plt
from xitorch.grad.jachess import jac

class surface_green(torch.autograd.Function):

    @staticmethod
    def forward(ctx, H, h01, S, s01, ee, left=True, method='Lopez-Sancho'):
        '''
        找找改进
        1. ee can be a list, to handle a batch of samples
        '''
        if method == 'GEP':
            gs = calcg0(ee, H, S, h01, s01, left=left)
        else:
            h10 = h01.conj().T
            s10 = s01.conj().T
            alpha, beta = h10 - ee * s10, h01 - ee * s01
            eps, epss = H.clone(), H.clone()

            converged = False
            iteration = 0
            while not converged:
                iteration += 1
                oldeps, oldepss = eps.clone(), epss.clone()
                oldalpha, oldbeta = alpha.clone(), beta.clone()
                tmpa = tLA.solve(ee * S - oldeps, oldalpha)
                tmpb = tLA.solve(ee * S - oldeps, oldbeta)

                alpha, beta = torch.mm(oldalpha, tmpa), torch.mm(oldbeta, tmpb)
                eps = oldeps + torch.mm(oldalpha, tmpb) + torch.mm(oldbeta, tmpa)
                if left:
                    epss = oldepss + torch.mm(oldalpha, tmpb)
                else:
                    epss = oldepss + torch.mm(oldbeta, tmpa)
                LopezConvTest = torch.max(alpha.abs() + beta.abs())

                if LopezConvTest < 1.0e-40:
                    gs = (ee * S - epss).inverse()

                    if left:
                        test = ee * S - H - torch.mm(ee * s10 - h10, gs.mm(ee * s01 - h01))
                    else:
                        test = ee * S - H - torch.mm(ee * s01 - h01, gs.mm(ee * s10 - h10))
                    myConvTest = torch.max((test.mm(gs) - torch.eye(H.shape[0], dtype=h01.dtype)).abs())
                    if myConvTest < 1.0e-5:
                        converged = True
                        if myConvTest > 1.0e-8:
                            v = "RIGHT"
                            if left: v = "LEFT"
                            print(
                                "WARNING: Lopez-scheme not-so-well converged for " + v + " electrode at E = %.4f eV:" % ee.real.item(),
                                myConvTest.item())
                    else:
                        print("Lopez-Sancho", myConvTest,
                              "Error: gs iteration {0}".format(iteration))
                        raise ArithmeticError("Criteria not met. Please check output...")

        ctx.save_for_backward(gs, H, h01, S, s01, ee)
        ctx.left = left

        return gs

    @staticmethod
    def backward(ctx, grad_outputs):
        gs_, H_, h01_, S_, s01_, ee_ = ctx.saved_tensors
        left = ctx.left

        if left:
            def sgfn(gs, *params):
                [H, h01, S, s01, ee] = params
                return tLA.inv(ee*S-H-(ee*s01.conj().T-h01.conj().T).matmul(gs).matmul(ee*s01-h01)) - gs
        else:
            def sgfn(gs, *params):
                [H, h01, S, s01, ee] = params
                return tLA.inv(ee*S - H - (ee*s01 - h01).matmul(gs).matmul(ee*s01.conj().T - h01.conj().T)) - gs

        params = [H_, h01_, S_, s01_, ee_]
        idx = [i for i in range(len(params)) if params[i].requires_grad]
        params_copy = [p.detach().requires_grad_() for p in params]

        with torch.enable_grad():

            grad = jac(fcn=sgfn, params=(gs_, *params), idxs=[0])[0] # dfdz
            pre = solve(A=grad.H, B=-grad_outputs.reshape(-1, 1))
            pre = pre.reshape(grad_outputs.shape)

            yfcn = sgfn(gs_, *params_copy)

            grad = torch.autograd.grad(yfcn, [params_copy[i] for i in idx], grad_outputs=pre,
                                                         create_graph=torch.is_grad_enabled(),
                                                         allow_unused=True)

        # grad = torch.autograd.grad(yfcn, params_copy, grad_outputs=pre,
        #                         create_graph=torch.is_grad_enabled(),
        #                         allow_unused=True)

            grad_out = [None for _ in range(len(params))]
            for i in range(len(idx)):
                grad_out[idx[i]] = grad[i]


            '''
            2. Is the matrix index direction correct? Also, is T necessarily becomes H when comes to complex matrix?
            '''
            # return *grad, None, None
            return *grad_out, None, None


def selfEnergy(hd, hu, sd, su, ee, coup_u=None, ovp_u=None, left=True, etaLead=1e-8, Bulk=False, voltage=0.0, dtype=torch.complex128, device='cpu', method='Lopez-Sancho'):

    if not isinstance(ee, torch.Tensor):
        eeshifted = torch.scalar_tensor(ee, dtype=dtype) - voltage  # Shift of self energies due to voltage(V)
    else:
        eeshifted = ee - voltage

    if coup_u == None:
        ESH = (eeshifted * sd - hd)
        SGF = surface_green.apply(hd, hu, sd, su, eeshifted + 1j * etaLead, left, method)
        # Sig = -0.5j*10 * torch.ones_like(SGF)
        # SGF = tLA.inv(ESH - Sig)

        if Bulk:
            Sig = tLA.inv(SGF)  # SGF^1
        else:
            Sig = ESH - tLA.inv(SGF)
    else:
        a, b = coup_u.shape
        SGF = surface_green.apply(hd, hu, sd, su, eeshifted + 1j * etaLead, left, method)
        if left:
            Sig = (ee*ovp_u.conj().T-coup_u.conj().T) @ SGF[-a:,-a:] @ (ee*ovp_u-coup_u)
        else:
            Sig = (ee*ovp_u-coup_u) @ SGF[:b,:b] @ (ee*ovp_u.conj().T-coup_u.conj().T)
    return Sig, SGF  # R(nuo, nuo)


def calcg0(ee, h00, s00, h01, s01, left=True):
    # here, for a single surface green function, for a specific |k>, ee is a matrix
    # Calculate surface Green's function
    # Euro Phys J B 62, 381 (2008)
    # Inverse of : NOTE, setup for "right" lead.
    # e-h00 -h01  ...
    # -h10  e-h00(e-h11) ...

    NN, ee = h00.shape[0], ee.real + max(torch.max(ee.imag).item(), 1e-8) * 1.0j
    if left:
        h01, s01 = h01.conj().T, s01.conj().T  # dagger is hermitian conjugation

    # Solve generalized eigen-problem
    # ( e I - h00 , -I) (eps)          (h01 , 0) (eps)
    # ( h10       ,  0) (xi ) = lambda (0   , I) (xi )
    a, b = torch.zeros((2 * NN, 2 * NN), dtype=h00.dtype), torch.zeros((2 * NN, 2 * NN),
                                                                             dtype=h00.dtype)
    a[0:NN, 0:NN] = ee * s00 - h00
    a[0:NN, NN:2 * NN] = -torch.eye(NN, dtype=h00.dtype)
    a[NN:2 * NN, 0:NN] = h01.conj().T - ee * s01.conj().T
    b[0:NN, 0:NN] = h01 - ee * s01
    b[NN:2 * NN, NN:2 * NN] = torch.eye(NN, dtype=h00.dtype)


    ev, evec = SLA.eig(a=a, b=b)
    ev = torch.tensor(ev, dtype=h00.dtype)
    evec = torch.tensor(evec, dtype=h00.dtype)
    # ev = torch.complex(real=torch.tensor(ev.real), imag=torch.tensor(ev.imag))
    # evec = torch.complex(real=torch.tensor(evec.real), imag=torch.tensor(evec.imag))

    # Select lambda <0 and the eps part of the evec
    ipiv = torch.where(ev.abs() < 1.)[0]

    ev, evec = ev[ipiv], evec[:NN, ipiv].T
    # Normalize evec
    norm = torch.diag(torch.mm(evec, evec.conj().T)).sqrt()
    evec = torch.mm(torch.diag(1.0 / norm), evec)

    # E^+ Lambda_+ (E^+)^-1 --->>> g00
    EP = evec.T
    FP = EP.mm(torch.diag(ev)).mm(torch.inverse(torch.mm(EP.conj().T, EP))).mm(EP.conj().T)
    g00 = torch.inverse(ee * s00 - h00 - torch.mm(h01 - ee * s01, FP))

    if left:
        g00 = iterative_gf(ee, g00, h00, h01.conj().T, s00, s01.conj().T, iter=3, left=left)
    else:
        g00 = iterative_gf(ee, g00, h00, h01, s00, s01, iter=3, left=left)

    # Check!
    err = torch.max(torch.abs(g00 - torch.inverse(ee * s00 - h00 - \
                                                  torch.mm(h01 - ee * s01, g00).mm(
                                                      h01.conj().T - ee * s01.conj().T))))
    if err > 1.0e-8 and left:
        print("WARNING: Lopez-scheme not-so-well converged for LEFT electrode at E = {0} eV:".format(ee.real.numpy()), err.numpy())
    if err > 1.0e-8 and not left:
        print("WARNING: Lopez-scheme not-so-well converged for RIGHT electrode at E = {0} eV:".format(ee.real.numpy()), err.numpy())
    return g00


def iterative_gf(ee, gs, h00, h01, s00, s01, iter=1, left=True):
    for i in range(iter):
        if not left:
            gs = ee*s00 - h00 - (ee * s01 - h01) @ gs @ (ee * s01.conj().T - h01.conj().T)
            gs = tLA.pinv(gs)
        else:
            gs = ee*s00 - h00 - (ee * s01.conj().T - h01.conj().T) @ gs @ (ee * s01 - h01)
            gs = tLA.pinv(gs)

    return gs