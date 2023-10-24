import torch
from calc import selfEnergy, GreenFunction
import torch.nn as nn

class SigProxy(nn.Module):
    def __init__(self, H, t01, S, s01, n_iter, left=True):
        super(SigProxy, self).__init__()
        self.H = H
        self.t01 = t01
        self.S = S
        self.s01 = s01
        dim = t01.shape[-1]
        self.linear1 = nn.Linear(in_features=dim, out_features=dim, dtype=torch.complex128)
        self.linear2 = nn.Linear(in_features=dim, out_features=dim, dtype=torch.complex128)
        self.n_iter = n_iter
        self.left = left

    def forward(self, ee):
        g = ee * self.S - self.H
        if self.left:
            r = ee * self.s01 - self.t01
            l = ee * self.s01.transpose(-2,-1) - self.t01.transpose(-2,-1)
        else:
            l = ee * self.s01 - self.t01
            r = ee * self.s01.transpose(-2,-1) - self.t01.transpose(-2,-1)

        for _ in range(self.n_iter):
            r = torch.tanh(self.linear1(r))
            l = torch.tanh(self.linear1(l))
            t = self.linear2(torch.tanh(r.matmul(g).matmul(l)))
            g = g + t

        return g

class preNet(nn.Module):
    def __init__(self, H, HL, HR, hL01, hR01, S, SL, SR, sL01, sR01, n_iter, nk1=1, nk2=1, ul=0, ur=0, etaLeads=1e-5, Bulk=False):
        super(preNet, self).__init__()
        self.seR = selfEnergy(H=HR, h01=hR01, S=SR, s01=sR01, nk1=nk1, nk2=nk2)
        self.seL = selfEnergy(H=HL, h01=hL01, S=SL, s01=sL01, nk1=nk1, nk2=nk2)
        self.proxySigL = SigProxy(HL, hL01, SL, sL01, n_iter, left=True)
        self.proxySigR = SigProxy(HR, hR01, SR, sR01, n_iter, left=False)
        self.greenfn = GreenFunction(H, S, elecL=self.seL, elecR=self.seR, ul=ul, ur=ur, Bulk=Bulk)
        self.etaLeads = etaLeads

    def forward(self, ee, kpoint=torch.tensor([0,0])):
        SigL = self.proxySigL(ee)
        SigR = self.proxySigR(ee)
        self.greenfn.calcGF(ee, kpoint, etaLead=self.etaLeads, proxy=True, pair=[SigL, SigR])
        pre_TT = self.greenfn.TT[0,0,:,:].real

        return pre_TT

if __name__ == '__main__':
    from examples.graphene import make_syst_junction
    from torch.optim import Adam, SGD
    import matplotlib.pyplot as plt

    t = 2.7
    eps = 0.
    nL = 11
    nR = 6
    n_iter = 2
    etaLeads = 1e-5

    epoch = 100
    lr = 1e-3

    t = torch.scalar_tensor(t, dtype=torch.complex128)
    eps = torch.scalar_tensor(eps, dtype=torch.complex128)
    HD, HL, HR, HL01, HR01, S, SL, SR, SL01, SR01 = make_syst_junction(nL, nR, eps, t)
    seR = selfEnergy(H=HR, h01=HR01, S=SR, s01=SR01, nk1=1, nk2=1)
    seL = selfEnergy(H=HL, h01=HL01, S=SL, s01=SL01, nk1=1, nk2=1)
    greenfn = GreenFunction(HD, S, elecL=seL, elecR=seR, ul=0, ur=0, Bulk=False)

    model = preNet(
        H=HD,
        HL=HL,
        HR=HR,
        hL01=HL01,
        hR01=HR01,
        S=S,
        SL=SL,
        SR=SR,
        sL01=SL01,
        sR01=SR01,
        n_iter=n_iter,
        nk1=1,
        nk2=1,
        ul=0,
        ur=0,
        etaLeads=etaLeads,
        Bulk=False
    )

    optimizer = Adam(model.parameters(), lr)
    loss_fn = nn.MSELoss()
    energies = [i*0.02 for i in range(-100, 100)]

    target = []
    for i in range(-100, 100):
        ee = i * 0.02
        greenfn.calcGF(ee, kpoint=torch.tensor([0,0]), etaLead=etaLeads)
        target.append(greenfn.TT[0,0,:,:].real)

    y = [target[i].trace().item() for i in range(200)]



    for e in range(epoch):
        prediction = []
        loss = 0
        for i in range(-100, 100):
            ee = i * 0.02
            pre = model(ee)
            prediction.append(torch.trace(pre).item())
            loss += loss_fn(pre, target[i+100])

        if e % 10 == 0:
            plt.plot(energies, prediction)
            plt.plot(energies, y)
            plt.legend(('pre', 'tgt'))
            plt.title("iter {}".format(e))
            plt.show()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


