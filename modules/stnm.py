# from torch.nn.modules.module import Module
# from functions.stnm import STNMFunction
#
# class STNM(Module):
#     def __init__(self):
#         super(STNM, self).__init__()
#         self.f = STNMFunction()
#     def forward(self, canvas, fgimg, fggrid, fgmask):
#         return self.f(canvas, fgimg, fggrid, fgmask)

from torch.nn.modules.module import Module
import torch.nn.functional as F


class STNM(Module):
    def __init__(self):
        super(STNM, self).__init__()

    def forward(self, canvas, fgimg, fggrid, fgmask):
        mask = F.grid_sample(fgmask, fggrid)
        fg = F.grid_sample(fgimg, fggrid)
        out = mask * fg + (1 - mask) * canvas
        return out
