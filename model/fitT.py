import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from examples.graphene import make_syst_junction
from calc import selfEnergy, GreenFunction
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import loss_landscapes
import loss_landscapes.metrics

device = 'cuda:0'

def adam(t0, eps, grad_t0_old, grad_eps_old, denome_t0_old, denome_eps_old,  beta1=0.9, beta2=0.999, lr=1e-3):
    grad_t0 = beta1 * grad_t0_old + (1-beta1) * t0.grad
    grad_eps = beta1 * grad_eps_old + (1-beta1) * eps.grad

    denom_t0 = beta2 * denome_t0_old + (1-beta2) * t0.grad**2
    denom_eps = beta2 * denome_eps_old + (1-beta2) * eps.grad**2

    grad_t0_ = grad_t0 / (1 - beta1)
    grad_eps_ = grad_eps / (1 - beta1)

    denom_t0_ = denom_t0 / (1 - beta2)
    denom_eps_ = denom_eps / (1- beta2)

    t0 = t0 - lr * grad_t0_ / (torch.sqrt(denom_t0_) + 1e-8)
    eps = eps - lr * grad_eps_ / (torch.sqrt(denom_eps_) + 1e-8)

    return t0, eps, grad_t0, grad_eps, denom_t0, denom_eps


def epoch(GF, bsz):
    indices = torch.randint(low=-1000, high=1000, size=(bsz,)).sort()[0]

    ee = indices * 0.01
    GF.calcGF(ee, kpoint=torch.tensor([0,0]), etaLead=1e-4)
    conductance = torch.einsum("bii->b", GF.TT.real[:,0,0,:,:])
    return conductance, ee, indices + 1000


if __name__ == '__main__':
    epcohSize = 2000
    lr = 0.1
    t0 = -1.
    eps = 2.
    nL = 11
    nR = 6
    lrScale = 1.
    device = 'cpu'
    bsz = 64

    plt.title("t=" + str(t0)+" eps="+str(eps))

    t0 = nn.Parameter(torch.scalar_tensor(t0, dtype=torch.double).requires_grad_())
    eps = nn.Parameter(torch.scalar_tensor(eps, dtype=torch.double).requires_grad_())

    lossfn = nn.MSELoss()

    HD, HL, HR, HL01, HR01, S, SL, SR, SL01, SR01 = make_syst_junction(nL, nR, eps, t0, device=device)
    seL = selfEnergy(H=HL, h01=HL01, S=SL, s01=SL01, nk1=1, nk2=1, device=device)
    seR = selfEnergy(H=HR, h01=HR01, S=SR, s01=SR01, nk1=1, nk2=1, device=device)
    GF = GreenFunction(H=HD, S=S, elecL=seL, elecR=seR, ul=0, ur=0, Bulk=False, device=device)

    # load data
    data = torch.load("../data/toyset.pth")["L11R6"]

    conduct = data[-1]
    # plt.plot(data[-2], conduct.numpy())

    # grad_t0_old, grad_eps_old, denome_t0_old, denome_eps_old = 0,0,0,0
    # optim = torch.optim.LBFGS((t0, eps,), lr=lr, max_iter=5)
    optim = torch.optim.LBFGS((t0, eps,), lr=lr)
    loss_list = []
    t0_list = []
    eps_list = []
    for i in tqdm(range(1,epcohSize+1)):


        def closure():
            conductance, energies, indices = epoch(GF, bsz)
            loss = lossfn(conductance, conduct[indices])
            optim.zero_grad()
            loss.backward(retain_graph=True)
            return loss


        # conductance, energies, indices = epoch(GF, bsz)
        # loss = lossfn(conductance, conduct[indices])
        # optim.zero_grad()
        # loss.backward()
        # optim.step()

        # loss_list.append(loss.item())

        optim.step(closure)

        # t0 = t0.detach()
        # t0.requires_grad = True
        # eps = eps.detach()
        # eps.requires_grad = True
        t0_list.append(t0.data.item())
        eps_list.append(eps.data.item())
        # print(t0.data, eps.data)

        print(t0.data, eps.data)


        # reload GF
        HD, HL, HR, HL01, HR01, S, SL, SR, SL01, SR01 = make_syst_junction(nL, nR, eps, t0, device=device)

        seL = selfEnergy(H=HL, h01=HL01, S=SL, s01=SL01, nk1=1, nk2=1, device=device)
        seR = selfEnergy(H=HR, h01=HR01, S=SR, s01=SR01, nk1=1, nk2=1, device=device)

        GF = GreenFunction(H=HD, S=S, elecL=seL, elecR=seR, ul=0, ur=0, Bulk=False, device=device)

        # update lr
        lr = lr * lrScale

    # fig = plt.figure()
    # ax = Axes3D(fig)
    #
    # ax.scatter(t0_list, eps_list, loss_list)
    torch.save({"t0":t0_list,"eps":eps_list, "loss":loss_list}, f="./data.pth")
    # plt.show()

    #     if i % 100 == 0:
    #
    #         plt.plot(energies, conductance.detach().numpy(), linestyle=':')
    #
    # plt.legend(("band","iter0","iter1","iter2","iter3","iter4"))
    #
    # plt.savefig("../img/comverge.png", dpi=600)

