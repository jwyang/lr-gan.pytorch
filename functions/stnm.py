# functions/add.py
import torch
from torch.autograd import Function
from _ext import stnm


class STNMFunction(Function):
    def forward(self, canvas, fgimg, fggrid, fgmask):
        self.canvas = canvas
        self.fgimg = fgimg
        self.fggrid = fggrid
        self.fgmask = fgmask
        output = torch.zeros(canvas.size()[0], canvas.size()[1], canvas.size()[2], canvas.size()[3])
        if not canvas.is_cuda:
            print("only support cuda now!")
            # stnm.BilinearSamplerBHWD_updateOutput(input1, input2, output)
        else:
            output = output.cuda()
            stnm.BilinearSamplerBHWD_updateOutput_cuda(canvas, fgimg, fggrid, fgmask, output)
        return output

    def backward(self, grad_output):
        grad_canvas = torch.zeros(self.canvas.size())
        grad_fgimg = torch.zeros(self.fgimg.size())
        grad_fggrid = torch.zeros(self.fggrid.size())
        grad_fgmask = torch.zeros(self.fgmask.size())
        if not grad_output.is_cuda:
            print("only support cuda now!")
            # stnm.BilinearSamplerBHWD_updateGradInput(self.input1, self.input2, grad_input1, grad_input2, grad_output)
        else:
            grad_output = grad_output.contiguous()
            grad_canvas = grad_canvas.cuda().contiguous()
            grad_fgimg = grad_fgimg.cuda().contiguous()
            grad_fggrid = grad_fggrid.cuda().contiguous()
            grad_fgmask = grad_fgmask.cuda().contiguous()
            stnm.BilinearSamplerBHWD_updateGradInput_cuda(self.canvas, self.fgimg, self.fggrid, self.fgmask, grad_canvas, grad_fgimg, grad_fggrid, grad_fgmask, grad_output)
        return grad_canvas, grad_fgimg, grad_fggrid, grad_fgmask
