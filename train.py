from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from modules.stnm import STNM
from modules.gridgen import AffineGridGen, CylinderGridGen, CylinderGridGenV2, DenseAffine3DGridGen, DenseAffine3DGridGen_rotate

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--ntimestep', type=int, default=2, help='number of recursive steps')
parser.add_argument('--maxobjscale', type=float, default=1.2, help='maximal object size relative to image')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--outimgf', default='images', help='folder to output images checkpoints')
parser.add_argument('--outmodelf', default='models', help='folder to model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nsize = int(opt.imageSize)
ntimestep = int(opt.ntimestep)
nc = 3

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _netG(nn.Module):
    def __init__(self, ngpu, nsize):
        super(_netG, self).__init__()
        self.ngpu = ngpu

        # define recurrent net for processing input random noise
        self.lstmcell = nn.LSTMCell(nz, nz)

        # define background generator G_bg
        self.Gbg = self.buildNetGbg(nsize)

        # define foreground generator G_fg
        #### define the common part
        self.Gfgc, depth_in = self.buildNetGfg(nsize)
        #### define the layer for generating fg image
        self.Gfgi = nn.Sequential(
            nn.ConvTranspose2d(depth_in, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        #### define the layer for generating fg mask
        self.Gfgm = nn.Sequential(
            nn.ConvTranspose2d(depth_in, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

        # define grid generator G_grid
        #### define linear layer to convert high-dim vector to 6-dim
        self.Gtransform = nn.Linear(nz, 6)
        self.Gtransform.weight.data.zero_()
        self.Gtransform.bias.data.zero_()
        self.Gtransform.bias.data[0] = opt.maxobjscale
        self.Gtransform.bias.data[4] = opt.maxobjscale

        self.Ggrid = AffineGridGen(nsize, nsize, aux_loss = False)

        # define compositor
        self.Compositor = STNM()

    def buildNetGbg(self, nsize): # take vector as input, and outout bgimg
        net = nn.Sequential()
        size_map = 1
        name = str(size_map)
        net.add_module('convt' + name, nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False))
        net.add_module('bn' + name, nn.BatchNorm2d(ngf * 8))
        net.add_module('relu' + name, nn.ReLU(True))
        size_map = 4
        depth_in = 8 * ngf
        depth_out = 4 * ngf
        while size_map < nsize / 2:
            name = str(size_map)
            net.add_module('convt' + name, nn.ConvTranspose2d(depth_in, depth_out, 4, 2, 1, bias=False))
            net.add_module('bn' + name, nn.BatchNorm2d(depth_out))
            net.add_module('relu' + name, nn.ReLU(True))
            depth_in = depth_out
            depth_out = max(depth_in / 2, 64)
            size_map = size_map * 2
        name = str(size_map)
        net.add_module('convt' + name, nn.ConvTranspose2d(depth_in, nc, 4, 2, 1, bias=False))
        net.add_module('tanh' + name, nn.Tanh())
        return net

    def buildNetGfg(self, nsize): # take vector as input, and output fgimg and fgmask
        net = nn.Sequential()
        size_map = 1
        name = str(size_map)
        net.add_module('convt' + name, nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False))
        net.add_module('bn' + name, nn.BatchNorm2d(ngf * 8))
        net.add_module('relu' + name, nn.ReLU(True))
        size_map = 4
        depth_in = 8 * ngf
        depth_out = 4 * ngf
        while size_map < nsize / 2:
            name = str(size_map)
            net.add_module('convt' + name, nn.ConvTranspose2d(depth_in, depth_out, 4, 2, 1, bias=False))
            net.add_module('bn' + name, nn.BatchNorm2d(depth_out))
            net.add_module('relu' + name, nn.ReLU(True))
            depth_in = depth_out
            depth_out = max(depth_in / 2, 64)
            size_map = size_map * 2
        return net, depth_in

    def forward(self, input):
        batchSize = input.size()[1]
        hx = Variable(torch.zeros(batchSize, nz).cuda())
        cx = Variable(torch.zeros(batchSize, nz).cuda())
        outputsT = []
        fgimgsT = []
        fgmaskT = []
        for i in range(ntimestep):
            hx, cx = self.lstmcell(input[i], (hx, cx))
            hx_view = hx.view(batchSize, nz, 1, 1)
            if i == 0:
                bg = self.Gbg(hx_view)
                outputsT.append(bg)
            else:
                fgc = self.Gfgc(hx_view)
                fgi = self.Gfgi(fgc)
                fgm = self.Gfgm(fgc)
                fgt = self.Gtransform(hx) # Nx6

                # fgt.data.select(1, 0).clamp(1.2, 4)
                # fgt.data.select(1, 1).clamp(-0.2, 0.2)
                # fgt.data.select(1, 2).clamp(-1, 1)
                # fgt.data.select(1, 3).clamp(-0.2, 0.2)
                # fgt.data.select(1, 4).clamp(1.2, 4)
                # fgt.data.select(1, 5).clamp(-1, 1)

                fgt_view = fgt.view(batchSize, 2, 3) # Nx2N3
                fgg = self.Ggrid(fgt_view)
                bg4c = bg.permute(0, 2, 3, 1) # torch.transpose(torch.transpose(bg, 1, 2), 2, 3) #
                fgi4c = fgi.permute(0, 2, 3, 1) # torch.transpose(torch.transpose(fgi, 1, 2), 2, 3) #
                fgm4c = fgm.permute(0, 2, 3, 1) # torch.transpose(torch.transpose(fgm, 1, 2), 2, 3) #
                temp = self.Compositor(bg4c, fgi4c, fgg, fgm4c)
                comb = temp.permute(0, 3, 1, 2) # torch.transpose(torch.transpose(temp, 2, 3), 1, 2) #
                outputsT.append(comb)
                fgimgsT.append(fgi)
                fgmaskT.append(fgm)
        return outputsT[ntimestep - 1], outputsT, fgimgsT, fgmaskT

netG = _netG(ngpu, nsize)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

class _netD(nn.Module):
    def __init__(self, ngpu, nsize):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = self.buildNet(nsize)

    def buildNet(self, nsize):
        net = nn.Sequential()
        depth_in = nc
        depth_out = ndf
        size_map = nsize
        while size_map > 4:
            name = str(size_map)
            net.add_module('conv' + name, nn.Conv2d(depth_in, depth_out, 4, 2, 1, bias=False))
            if size_map < nsize:
                net.add_module('bn' + name, nn.BatchNorm2d(depth_out))
            net.add_module('lrelu' + name, nn.LeakyReLU(0.2, inplace=True))
            depth_in = depth_out
            depth_out = 2 * depth_in
            size_map = size_map / 2
        name = str(size_map)
        net.add_module('conv' + name, nn.Conv2d(depth_in, 1, 4, 1, 0, bias=False))
        net.add_module('sigmoid' + name, nn.Sigmoid())
        return net

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)


netD = _netD(ngpu, nsize)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(ntimestep, opt.batchSize, nz)
fixed_noise = torch.FloatTensor(ntimestep, opt.batchSize, nz).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# noise_all = []
# fixed_noise_all = []
# for i in range(ntimestep):
#     noise_temp = torch.FloatTensor(opt.batchSize, nz, 1, 1)
#     fixed_noise_temp = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
#     noise_temp, fixed_noise_temp = noise_temp.cuda(), fixed_noise_temp.cuda()
#     noise_all.append(Variable(noise_temp))
#     fixed_noise_all.append(Variable(fixed_noise_temp))

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)

        output = netD(inputv)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.resize_(ntimestep, batch_size, nz).normal_(0, 1)
        noisev = Variable(noise)
        fake, fakeseq, fgimgseq, fgmaskseq = netG(noisev)
        labelv = Variable(label.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()
        # print(torch.sum(netG.Gtransform.weight.data))

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outimgf) # normalize=True
            fake, fakeseq, fgimgseq, fgmaskseq = netG(fixed_noise)
            for i in range(ntimestep):
                vutils.save_image(fakeseq[i].data,
                        '%s/fake_samples_t_%01d.png' % (opt.outimgf, i)) # normalize=True
            if ntimestep > 1:
                vutils.save_image(fgimgseq[0].data,
                        '%s/fake_samples_t_%01d_fgimg.png' % (opt.outimgf, 1)) # normalize=True
                vutils.save_image(fgmaskseq[0].data,
                        '%s/fake_samples_t_%01d_fgmask.png' % (opt.outimgf, 1)) # normalize=True
        # exit()

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outmodelf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outmodelf, epoch))
