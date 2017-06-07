from torch.nn.modules.module import Module
from functions.stnm import STNMFunction

class STNM(Module):
    def __init__(self):
        super(STNM, self).__init__()
        self.f = STNMFunction()
    def forward(self, canvas, fgimg, fggrid, fgmask):
        return self.f(canvas, fgimg, fggrid, fgmask)
