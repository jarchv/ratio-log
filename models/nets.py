import torch
import torch.nn as nn

from .vgg import VGGEncBN, VGGDecBN
from .vgg import VGGEnc, VGGDec

class VGGED_BN(nn.Module):        
    def __init__(self, conf):
        super(VGGED_BN, self).__init__()
        self.vgg16_encoder = VGGEncBN(conf=conf)
        self.vgg16_decoder = VGGDecBN(conf=conf)

    def forward(self, net_inp):  
        layers  = self.vgg16_encoder.forward(net_inp)
        net_out = self.vgg16_decoder.forward(layers)
        return layers['z'], net_out

class VGGED(nn.Module):
    def __init__(self, conf):
        super(VGGED, self).__init__()
        self.vgg16_encoder = VGGEnc(conf=conf)
        self.vgg16_decoder = VGGDec(conf=conf)

    def forward(self, net_inp):
        layers  = self.vgg16_encoder.forward(net_inp)
        net_out = self.vgg16_decoder.forward(layers)
        return layers['z'], net_out

class DisNet(nn.Module):
    def __init__(
        self, 
        init_ch    = 8, 
        ksize      = 4, 
        down_leves = 5):
        super(DisNet, self).__init__()
        
        inp_ch = 3
        out_ch = init_ch
        seq    = []

        # input: [224x224]
        seq += self.conv_block(inp_ch, out_ch, 3, 1, 1, 1)
        for _ in range(down_leves):
            inp_ch = out_ch
            out_ch = min(out_ch*2, init_ch * 8)
            seq += self.conv_block(inp_ch, out_ch, ksize, 2, 1, 1)

        inp_ch = out_ch
        out_ch = min(out_ch*2, init_ch * 8)
        seq.append(
            nn.Conv2d(
                in_channels  = inp_ch,
                out_channels = 1,
                kernel_size  = 3,
                stride       = 1,
                padding      = 1,
                dilation     = 1)
        )
        ds_size = 224 // 2 ** down_leves
        
        self.conv_arch = nn.Sequential(*seq)
        self.lin_layer = nn.Linear(ds_size ** 2, 1)

    def conv_block(self, inp_ch, out_ch, ksize, stride, pad, dil):
        subseq = [nn.Conv2d(in_channels  = inp_ch,
                            out_channels = out_ch,
                            kernel_size  = ksize, 
                            stride       = stride, 
                            padding      = pad,
                            dilation     = dil,
                            bias         = False)]

        subseq.append(nn.BatchNorm2d(out_ch))
        subseq.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        return  subseq

    def forward(self, inputs):
        out = self.conv_arch(inputs)
        out = torch.reshape(out, (out.shape[0], -1))
        out = self.lin_layer(out)
        
        return out

class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, pred, mode):
        if mode == 'real':
            target = self.real_label
        elif mode == 'fake':
            target = self.fake_label

        target = target.expand_as(pred)
        out    = self.loss(pred, target)
        
        return out


class UNetChen(nn.Module):
    """
        U-Net architecture based on "Learning to See in the Dark" (Chen et al., 2018)
        arXiv: https://arxiv.org/abs/1805.01934
    """

    def __init__(self, img_ch=3, init_ch=32):
        super(UNetChen, self).__init__()
        self.init_ch = init_ch

        self.conv_block1 = self.conv_block(img_ch, init_ch)
        
        self.maxpool1    = nn.MaxPool2d(2,2)
        self.conv_block2 = self.conv_block(init_ch, init_ch*2)
            
        self.maxpool2    = nn.MaxPool2d(2,2)
        self.conv_block3 = self.conv_block(init_ch*2, init_ch*4)
            
        self.maxpool3    = nn.MaxPool2d(2,2)
        self.conv_block4 = self.conv_block(init_ch*4, init_ch*8)
            
        self.maxpool4    = nn.MaxPool2d(2,2)
        self.conv_block5 = self.conv_block(init_ch*8, init_ch*16)
        self.deconv4       = nn.ConvTranspose2d(init_ch*16, init_ch*8, 2, stride=2)
            
        self.upconv_block4 = self.conv_block(init_ch*16, init_ch*8)
        self.deconv3       = nn.ConvTranspose2d(init_ch*8, init_ch*4, 2, stride=2)

        self.upconv_block3 = self.conv_block(init_ch*8, init_ch*4)
        self.deconv2       = nn.ConvTranspose2d(init_ch*4, init_ch*2, 2, stride=2)
        
        self.upconv_block2 = self.conv_block(init_ch*4, init_ch*2)              
        self.deconv1       = nn.ConvTranspose2d(init_ch*2, init_ch, 2, stride=2)

        self.upconv_block1 = self.conv_block(init_ch*2, init_ch)      
        self.out_conv      = nn.Conv2d(init_ch, img_ch, 1, padding=0)

    def conv_block(self, in_ch, out_ch):
        return  nn.Sequential(
                    nn.Conv2d(in_ch , out_ch, 3, padding=1),
                    #nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    #nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )

    def forward(self, inputs):

        # ImgSize: [224x224]
        convBlock1 = self.conv_block1(inputs)
        # ImgSize: [224x224]

        pool1      = self.maxpool1(convBlock1)
        convBlock2 = self.conv_block2(pool1)
        # ImgSize: [112x112]

        
        pool2      = self.maxpool2(convBlock2)
        convBlock3 = self.conv_block3(pool2)
        # ImgSize: [56x56]

        pool3      = self.maxpool3(convBlock3)
        convBlock4 = self.conv_block4(pool3)
        # ImgSize: [28x28]

        pool4      = self.maxpool4(convBlock4)
        convBlock5 = self.conv_block5(pool4)
            # ImgSize: [14x14]

        upBlock4 = self.deconv4(convBlock5, output_size=convBlock4.size())
        concat4  = torch.cat([upBlock4, convBlock4],dim=1)
        # ImgSize: [28x28]

        upconvBlock4 = self.upconv_block4(concat4)
        upBlock3 = self.deconv3(upconvBlock4, output_size=convBlock3.size())
        concat3  = torch.cat([upBlock3, convBlock3],dim=1)
        # ImgSize: [56x56]

        upconvBlock3 = self.upconv_block3(concat3)
        upBlock2 = self.deconv2(upconvBlock3, output_size=convBlock2.size())
        concat2  = torch.cat([upBlock2, convBlock2],dim=1)
        # ImgSize: [112x112]

        upconvBlock2 = self.upconv_block2(concat2)
        upBlock1 = self.deconv1(upconvBlock2, output_size=convBlock1.size())
        concat1  = torch.cat([upBlock1, convBlock1],dim=1)
        # ImgSize: [224x224]

        upconvBlock1 = self.upconv_block1(concat1)
        out_conv = self.out_conv(upconvBlock1)
        #out      = self.out_act(out_conv)
        out      = out_conv
        # ImgSize: [224x224]

        return out