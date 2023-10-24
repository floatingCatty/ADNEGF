from examples.graphene import make_syst_junction
import torch
import numpy as np
from calc import selfEnergy, GreenFunction
from tqdm import tqdm
import matplotlib.pyplot as plt

def dataGenerator(structureList, eps, t, eta, n_sample=2000):
    data = {}
    for nL, nR in tqdm(structureList):
        HD, HL, HR, HL01, HR01, S, SL, SR, SL01, SR01 = make_syst_junction(nL, nR, eps, t)
        seL = selfEnergy(H=HL, h01=HL01, S=SL, s01=SL01, nk1=1, nk2=1)
        seR = selfEnergy(H=HR, h01=HR01, S=SR, s01=SR01, nk1=1, nk2=1)

        GF = GreenFunction(H=HD, S=S, elecL=seL, elecR=seR, ul=0, ur=0, Bulk=False)

        energy = []
        conductance = []

        for i in range(-int(n_sample/2), int(n_sample/2)):
            ee = i*0.01
            GF.calcGF(ee=ee, kpoint=torch.tensor([0, 0]), etaLead=eta)
            energy.append(ee)
            conductance.append(torch.trace(GF.TT[0, 0, :, :].real).detach())
        energy = torch.tensor(energy)
        conductance = torch.tensor(conductance)

        data.update({"L"+str(nL)+"R"+str(nR):[HL,HD,HR,HL01,HR01,S,SL,SR,SL01,SR01,energy, conductance]})

    torch.save(data, "./toyset1e4.pth")

    return True


if __name__ == '__main__':
    structList = [(11,6),(5,2),(13,6),(15,8),(31,16)]
    dataGenerator(structList, eps=0., t=2.7, eta=1e-4, n_sample=10000)