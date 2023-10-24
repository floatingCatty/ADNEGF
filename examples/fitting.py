import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.optim import Adam
import seaborn as sns
from graphene import make_syst_junction
from calc.NEGF import NEGF
import numpy as np

loss_list = []

def getNEGFjunction(nL=11, nR=6, eps=0, t=2.7):
    HD, HL, HR, HL01, HR01, S, SL, SR, SL01, SR01 = make_syst_junction(nL=nL, nR=nR, eps=eps, t=t)


    def pack(**params):
        return params
    hmt_ovp = pack(hd=[HD],hu=[HR01],hl=[HL01.conj().T], lhd=HL, lhu=HL01, rhd=HR, rhu=HR01)
    negf = NEGF(hmt_ovp)

    return negf

def getSamples(negf, batchsize, n_batch):
    ee = torch.linspace(-5, 5, 2000)
    out = negf.calGreen(ee, ul=0, ur=0, calTT=True, cutoff=False)
    TT = out['TT'].detach()
    batch = []
    for i in tqdm(range(n_batch), desc="getting samples"):
        idx = torch.randint(low=0, high=2000, size=(batchsize,))
        batch.append((ee[idx], TT[idx]))

    return batch

def update(HD, HL, HR, HL01, HR01, batch, optimizer):
    def pack(**params):
        return params
    hmt_ovp = pack(hd=[HD],hu=[HR01],hl=[HL01.conj().T], lhd=HL, lhu=HL01, rhd=HR, rhu=HR01)
    negf = NEGF(hmt_ovp)
    (ee, TT) = batch
    TT_ = negf.calGreen(ee, ul=0., ur=0., calTT=True, cutoff=False)['TT']
    optimizer.zero_grad()
    loss = (TT - TT_).abs().mean()
    loss.backward()
    optimizer.step()
    loss_list.append(loss.item())

    return HD, HL, HR, HL01, HR01

def cal_TT(HD, HL, HR, HL01, HR01):
    def pack(**params):
        return params
    hmt_ovp = pack(hd=[HD],hu=[HR01],hl=[HL01.conj().T], lhd=HL, lhu=HL01, rhd=HR, rhu=HR01)
    negf = NEGF(hmt_ovp)
    ee = torch.linspace(-5, 5, 400)
    TT = negf.calGreen(ee, ul=0., ur=0., calTT=True, cutoff=False)['TT']

    return TT.detach()

def plot():
    f = torch.load("fitting_save.pth")
    loss, param = f
    HD_, HL_, HR_, HL01_, HR01_ = param
    HD, HL, HR, HL01, HR01, S, SL, SR, SL01, SR01 = make_syst_junction(nL=7, nR=4, eps=0, t=2.7, )
    TT_target = cal_TT(HD, HL, HR, HL01, HR01)
    HD, HL, HR, HL01, HR01, S, SL, SR, SL01, SR01 = make_syst_junction(nL=5, nR=2, eps=0, t=2.7, )
    TT_org = cal_TT(HD, HL, HR, HL01, HR01)
    np.savetxt(fname="HD.txt", X=HD.real.detach().numpy(), fmt="%5.2f")
    np.savetxt(fname="HD_.txt", X=HD_.real.detach().numpy(), fmt="%5.2f")
    plt.matshow(HD.real.detach().numpy())


    plt.matshow(HD_.real.detach())
    plt.matshow((HD_-HD).real.detach())
    plt.colorbar()
    TT_fit = cal_TT(HD_, HL_, HR_, HL01_, HR01_)

    ee = torch.linspace(-5, 5, 400)
    fig, (ax) = plt.subplots(2,1, figsize=(6.8, 5.8))
    ax[0].plot(loss, c='black')
    ax[0].set_xlabel("iter", fontsize=14)
    ax[0].set_ylabel("loss", fontsize=14)

    ax[1].plot(ee, TT_target,'-.', lw=2)
    ax[1].plot(ee, TT_org, '--', c='tab:green', lw=2)
    ax[1].plot(ee, TT_fit, c='black')
    ax[1].set_xlabel("E (eV)", fontsize=14)
    ax[1].set_ylabel("T(E)", fontsize=14)
    ax[1].set_xlim((-5,5))
    plt.legend(['7-4 (target)','5-2 (initial)','5-2 (fitted)'], fontsize=12, loc="lower left", mode="expand", ncol=3, bbox_to_anchor=(0.,1.02,1.0,0.2))
    plt.tight_layout()
    plt.savefig("./experimental_data/fitting.pdf", dpi=600)
    plt.show()


    return TT_org, TT_target, TT_fit





if __name__ == '__main__':
    plot()

    # negf = getNEGFjunction(nL=7, nR=4)
    # ee = torch.linspace(-5,5,200)
    # out = negf.calGreen(ee, ul=0, ur=0, calTT=True, calDOS=True, cutoff=False)
    #
    # batch = torch.load('fitting_Data.pth')
    #
    # if not batch:
    #     batch = getSamples(negf, batchsize=64, n_batch=1000)
    #     torch.save(obj=batch, f='fitting_Data.pth')
    #
    # n_iter = 100
    # HD, HL, HR, HL01, HR01, S, SL, SR, SL01, SR01 = make_syst_junction(nL=5, nR=2, eps=0, t=2.7, )
    # mask_HD = HD / (HD+1e-8)
    # mask_HL = HL / (HL+1e-8)
    # mask_HR = HR / (HR+1e-8)
    # mask_HL01 = HL01 / (HL01+1e-8)
    # mask_HR01 = HR01 / (HR01+1e-8)
    # HD.requires_grad_(), HL.requires_grad_(), HR.requires_grad_(), HL01.requires_grad_(), HR01.requires_grad_()
    #
    # optimizer = Adam(params=[HD], lr=1e-3)
    #
    # for _ in range(n_iter):
    #     for i, btz in tqdm(enumerate(batch), desc="training iter"):
    #         HD_ = HD * mask_HD
    #         HL_ = HL * mask_HL
    #         HR_ = HR * mask_HR
    #         HL01_ = HL01 * mask_HL01
    #         HR01_ = HR01 * mask_HR01
    #         HD_, HL_, HR_, HL01_, HR01_ = update(HD_, HL_, HR_, HL01_, HR01_, batch=btz, optimizer=optimizer)
    #
    #         if (i+1) % 100 == 0:
    #             torch.save(obj=(loss_list, (HD_, HL_, HR_, HL01_, HR01_)), f="fitting_save.pth")
