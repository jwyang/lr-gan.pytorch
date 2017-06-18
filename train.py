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
parser.add_argument('--epoch_s', type=int, default=0, help='start epoch for training, used for pretrained momdel')
parser.add_argument('--session', type=int, default=1, help='training session')
parser.add_argument('--evaluate', type=bool, default=False, help='training session')
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
    checkfreq = 100
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
    checkfreq = 100
    nc = 3
elif opt.dataset == 'cub200':
    trans = transforms.Compose([
        transforms.Scale(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = dset.ImageFolder('datasets/cub200/images', transform = trans)
    checkfreq = 40
    nc = 3
elif opt.dataset == 'mnist-one':
    trans = transforms.Compose([
        transforms.Scale(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = dset.ImageFolder('datasets/mnist-one/images', transform = trans)
    checkfreq = 100
    nc = 1
elif opt.dataset == 'mnist-two':
    trans = transforms.Compose([
        transforms.Scale(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = dset.ImageFolder('datasets/mnist-two/images', transform = trans)
    checkfreq = 100
    nc = 1
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nsize = int(opt.imageSize)
ntimestep = int(opt.ntimestep)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _netG(nn.Module):
    def __init__(self, ngpu, nsize):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.nsize_out = 2
        # define recurrent net for processing input random noise
        self.lstmcell = nn.LSTMCell(nz, nz)

        # define background generator G_bg
        self.Gbgc, self.depth_in_bg = self.buildNetGbg(nsize)
        self.Gbgi = nn.Sequential(
            nn.ConvTranspose2d(self.depth_in_bg, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        # define foreground generator G_fg
        #### define the common part
        self.Gfgc, self.depth_in = self.buildNetGfg(nsize)
        #### define the layer for generating fg image
        self.Gfgi = nn.Sequential(
            nn.ConvTranspose2d(self.depth_in, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        #### define the layer for generating fg mask
        self.Gfgm = nn.Sequential(
            nn.ConvTranspose2d(self.depth_in, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

        # define grid generator G_grid
        #### define linear layer to convert high-dim vector to 6-dim
        self.Gtransform = nn.Linear(nz, 6)
        self.Gtransform.weight.data.zero_()
        self.Gtransform.bias.data.zero_()
        self.Gtransform.bias.data[0] = opt.maxobjscale
        self.Gtransform.bias.data[4] = opt.maxobjscale

        # define grid generator
        self.Ggrid = AffineGridGen(nsize, nsize, aux_loss = False)

        # define compsitor
        self.Compositors = []
        for t in range(ntimestep - 1):
            self.Compositors.append(STNM())

        # define encoder when ntimestep > 2
        self.encoderconv = self.buildEncoderConv(self.depth_in, nsize / 2, self.nsize_out)
        self.encoderfc = self.buildEncoderFC(self.depth_in, self.nsize_out, nz)
        self.nlnet = nn.Sequential(
            nn.Linear(nz + nz, nz),
            nn.BatchNorm1d(nz),
            nn.Tanh()
        )

    def buildNetGbg(self, nsize): # take vector as input, and outout bgimg
        net = nn.Sequential()
        size_map = 1
        name = str(size_map)
        net.add_module('convt' + name, nn.ConvTranspose2d(nz, ngf * 4, 4, 4, 0, bias=False))
        net.add_module('bn' + name, nn.BatchNorm2d(ngf * 4))
        net.add_module('relu' + name, nn.ReLU(True))
        size_map = 4
        depth_in = 4 * ngf
        depth_out = 2 * ngf
        while size_map < nsize / 2:
            name = str(size_map)
            net.add_module('convt' + name, nn.ConvTranspose2d(depth_in, depth_out, 4, 2, 1, bias=False))
            net.add_module('bn' + name, nn.BatchNorm2d(depth_out))
            net.add_module('relu' + name, nn.ReLU(True))
            depth_in = depth_out
            depth_out = max(depth_in / 2, 64)
            size_map = size_map * 2
        return net, depth_in

    def buildNetGfg(self, nsize): # take vector as input, and output fgimg and fgmask
        net = nn.Sequential()
        size_map = 1
        name = str(size_map)
        net.add_module('convt' + name, nn.ConvTranspose2d(nz, ngf * 8, 4, 4, 0, bias=False))
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

    def buildEncoderConv(self, depth_in, nsize_in, nsize_out):
        net = nn.Sequential()
        nsize_i = nsize_in
        while nsize_i > nsize_out:
            name = str(nsize_i)
            net.add_module('avgpool' + name, nn.AvgPool2d(4, 2, 1))
            net.add_module('bn' + name, nn.BatchNorm2d(depth_in))
            net.add_module('lrelu' + name, nn.LeakyReLU(0.2, inplace=True))
            nsize_i = nsize_i / 2
        return net

    def buildEncoderFC(self, depth_in, nsize_in, out_dim):
        net = nn.Sequential(
            nn.Linear(depth_in * nsize_in * nsize_in, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Tanh()
        )
        return net

    def clampT(self, Tin):
        x_s = Tin.select(1, 0)
        x_r = Tin.select(1, 1)
        x_t = Tin.select(1, 2)

        y_r = Tin.select(1, 3)
        y_s = Tin.select(1, 4)
        y_t = Tin.select(1, 5)

        x_s_clamp = torch.unsqueeze(x_s.clamp(opt.maxobjscale, 2 * opt.maxobjscale), 1)
        x_r_clmap = torch.unsqueeze(x_r.clamp(-0.3, 0.3), 1)
        x_t_clmap = torch.unsqueeze(x_t.clamp(-1, 1), 1)

        y_r_clamp = torch.unsqueeze(y_r.clamp(-0.3, 0.3), 1)
        y_s_clamp = torch.unsqueeze(y_s.clamp(opt.maxobjscale, 2 * opt.maxobjscale), 1)
        y_t_clamp = torch.unsqueeze(y_t.clamp(-1, 1), 1)

        Tout = torch.cat([x_s_clamp, x_r_clmap, x_t_clmap, y_r_clamp, y_s_clamp, y_t_clamp], 1)
        return Tout

    def forward(self, input):
        batchSize = input.size()[1]
        hx = Variable(torch.zeros(batchSize, nz).cuda())
        cx = Variable(torch.zeros(batchSize, nz).cuda())
        outputsT = []
        fgimgsT = []
        fgmaskT = []
        for i in range(ntimestep):
            hx, cx = self.lstmcell(input[i], (hx, cx))
            hx_view = hx.contiguous().view(batchSize, nz, 1, 1)
            if i == 0:
                # We send input vector to background generator directly
                # to make it equivalent to DCGAN when ntimestep = 1.
                input_view = input[i].view(batchSize, nz, 1, 1)
                bgc = self.Gbgc(input_view)
                canvas = self.Gbgi(bgc)
                outputsT.append(canvas)
            else:
                if ntimestep > 2 and i == 1:
                    encConv = self.encoderconv(bgc)
                    encConv_view = encConv.view(batchSize, self.depth_in * self.nsize_out * self.nsize_out)
                    encFC = self.encoderfc(encConv_view)
                    concat = torch.cat([hx, encFC], 1)
                    comb = self.nlnet(concat)
                    input4g = comb
                    input4g_view = input4g.contiguous().view(batchSize, nz, 1, 1)
                elif ntimestep > 2 and i > 1:
                    encConv = self.encoderconv(fgc)
                    encConv_view = encConv.view(batchSize, self.depth_in * self.nsize_out * self.nsize_out)
                    encFC = self.encoderfc(encConv_view)
                    concat = torch.cat([hx, encFC], 1)
                    comb = self.nlnet(concat)
                    input4g = comb
                    input4g_view = input4g.contiguous().view(batchSize, nz, 1, 1)
                else:
                    input4g = hx
                    input4g_view = hx_view

                fgc = self.Gfgc(input4g_view)
                fgi = self.Gfgi(fgc)
                fgm = self.Gfgm(fgc)

                fgt = self.Gtransform(input4g) # Nx6
                fgt_clamp = self.clampT(fgt)
                fgt_view = fgt_clamp.contiguous().view(batchSize, 2, 3) # Nx2N3
                fgg = self.Ggrid(fgt_view)
                canvas4c = canvas.permute(0, 2, 3, 1).contiguous() # torch.transpose(torch.transpose(bg, 1, 2), 2, 3) #
                fgi4c = fgi.permute(0, 2, 3, 1).contiguous() # torch.transpose(torch.transpose(fgi, 1, 2), 2, 3) #
                fgm4c = fgm.permute(0, 2, 3, 1).contiguous() # torch.transpose(torch.transpose(fgm, 1, 2), 2, 3) #
                temp = self.Compositors[i - 1](canvas4c, fgi4c, fgg, fgm4c)
                canvas = temp.permute(0, 3, 1, 2).contiguous() # torch.transpose(torch.transpose(temp, 2, 3), 1, 2) #
                outputsT.append(canvas)
                fgimgsT.append(fgi)
                fgmaskT.append(fgm)
        return outputsT[ntimestep - 1], outputsT, fgimgsT, fgmaskT

netG = _netG(ngpu, nsize)
# For smaller network, initialize BN as usual
# For larger network, initialize BN with zero mean
# The later case works for both, so we commet the initializzaton
# netG.apply(weights_init)
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
# For smaller network, initialize BN as usual
# For larger network, initialize BN with zero mean
# The later case works for both, so we commet the initializzaton
# netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
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

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.epoch_s, opt.niter):
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
        if opt.dataset == 'mnist-one' or opt.dataset == 'mnist-two':
            real_cpu = torch.mean(real_cpu, 1)
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

        print('[%d][%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (opt.session, epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % checkfreq == 0:
            vutils.save_image(real_cpu,
                    '%s/%s_real_samples.png' % (opt.outimgf, opt.dataset)) # normalize=True
            fake, fakeseq, fgimgseq, fgmaskseq = netG(fixed_noise)
            for t in range(ntimestep):
                vutils.save_image(fakeseq[t].data,
                        '%s/%s_fake_samples_s_%01d_t_%01d.png' % (opt.outimgf, opt.dataset, opt.session, t)) # normalize=True
                if t > 0:
                    vutils.save_image(fgimgseq[t - 1].data,
                            '%s/%s_fake_samples_s_%01d_t_%01d_fgimg.png' % (opt.outimgf, opt.dataset, opt.session, t)) # normalize=True
                    vutils.save_image(fgmaskseq[t - 1].data.sub_(0.5).div_(0.5),
                            '%s/%s_fake_samples_s_%01d_t_%01d_fgmask.png' % (opt.outimgf, opt.dataset, opt.session, t)) # normalize=True
            if opt.evaluate:
                exit()

    # do checkpointing
    if epoch % 5 == 0:
        torch.save(netG.state_dict(), '%s/%s_netG_s_%d_epoch_%d.pth' % (opt.outmodelf, opt.dataset, opt.session, epoch))
        torch.save(netD.state_dict(), '%s/%s_netD_s_%d_epoch_%d.pth' % (opt.outmodelf, opt.dataset, opt.session, epoch))
