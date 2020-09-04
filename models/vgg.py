import torch
import torch.nn as nn

from torchvision import models

class VGGEncBN(nn.Module):
    def __init__(self, conf):
        super(VGGEncBN, self).__init__()
        self.depth = conf.depth
        self.pretr = conf.pretr

        vgg16_model = models.vgg16_bn(pretrained=self.pretr)
        feat_list   = list(vgg16_model.features)

        #self.conv_pre = nn.Conv2d(3,3,1,1,0,bias=True)
        if self.depth > 0:
            # [224x224]

            self.conv1_1 = feat_list[0]
            self.bnor1_1 = feat_list[1]
            self.relu1_1 = feat_list[2] 
            self.conv1_2 = feat_list[3]
            self.bnor1_2 = feat_list[4]
            self.relu1_2 = feat_list[5]  

        if self.depth > 1:
            # [112x112]

            self.maxp2_1 = feat_list[6] 
            self.conv2_1 = feat_list[7]
            self.bnor2_1 = feat_list[8]
            self.relu2_1 = feat_list[9] 
            self.conv2_2 = feat_list[10]
            self.bnor2_2 = feat_list[11]
            self.relu2_2 = feat_list[12]                

        if self.depth > 2:
            # [56x56]

            self.maxp3_1 = feat_list[13] 
            self.conv3_1 = feat_list[14]
            self.bnor3_1 = feat_list[15]
            self.relu3_1 = feat_list[16] 
            self.conv3_2 = feat_list[17]
            self.bnor3_2 = feat_list[18]
            self.relu3_2 = feat_list[19] 
            self.conv3_3 = feat_list[20]
            self.bnor3_3 = feat_list[21]
            self.relu3_3 = feat_list[22] 

        if self.depth > 3:
            # [28x28]
        
            self.maxp4_1 = feat_list[23] 
            self.conv4_1 = feat_list[24]
            self.bnor4_1 = feat_list[25]
            self.relu4_1 = feat_list[26] 
            self.conv4_2 = feat_list[27]
            self.bnor4_2 = feat_list[28]
            self.relu4_2 = feat_list[29] 
            self.conv4_3 = feat_list[30]
            self.bnor4_3 = feat_list[31]
            self.relu4_3 = feat_list[32] 

        if self.depth > 4:
            # [14x14]

            self.maxp5_1 = feat_list[33] 
            self.conv5_1 = feat_list[34]
            self.bnor5_1 = feat_list[35]
            self.relu5_1 = feat_list[36] 
            self.conv5_2 = feat_list[37]
            self.bnor5_2 = feat_list[38]
            self.relu5_2 = feat_list[39] 
            self.conv5_3 = feat_list[40]
            self.bnor5_3 = feat_list[41]
            self.relu5_3 = feat_list[42]   

    def forward(self, input):

        layers = {}

        #out  = self.conv_pre(input)
        out  = self.conv1_1(input)
        out  = self.bnor1_1(out)
        out  = self.relu1_1(out)
        out  = self.conv1_2(out)
        out  = self.bnor1_2(out)
        out1 = self.relu1_2(out)

        if self.depth < 2: 
            layers['z'] = out1
            return layers

        layers['out1'] = out1
        out   = self.maxp2_1(out1)

        out   = self.conv2_1(out)
        out   = self.bnor2_1(out)
        out   = self.relu2_1(out)
        out   = self.conv2_2(out)
        out   = self.bnor2_2(out)
        out2  = self.relu2_2(out)

        if self.depth < 3: 
            layers['z'] = out2
            return layers

        layers['out2'] = out2
        out   = self.maxp3_1(out2)

        out   = self.conv3_1(out)
        out   = self.bnor3_1(out)
        out   = self.relu3_1(out)
        out   = self.conv3_2(out)
        out   = self.bnor3_2(out)
        out   = self.relu3_2(out)
        out   = self.conv3_3(out)
        out   = self.bnor3_3(out)
        out3  = self.relu3_3(out)

        if self.depth < 4:
            layers['z'] = out3
            return layers

        layers['out3'] = out3
        out   = self.maxp4_1(out3)

        out   = self.conv4_1(out)
        out   = self.bnor4_1(out)
        out   = self.relu4_1(out)
        out   = self.conv4_2(out)
        out   = self.bnor4_2(out)
        out   = self.relu4_2(out)
        out   = self.conv4_3(out)
        out   = self.bnor4_2(out)
        out4  = self.relu4_3(out)

        if self.depth < 5: 
            layers['z'] = out4
            return layers

        layers['out4'] = out4
        out = self.maxp5_1(out4)

        out = self.conv5_1(out)
        out = self.bnor5_1(out)
        out = self.relu5_1(out)
        out = self.conv5_2(out)
        out = self.bnor5_2(out)
        out = self.relu5_2(out)
        out = self.conv5_3(out)
        out = self.bnor5_3(out)
        out = self.relu5_3(out)

        layers['z'] = out

        return layers

class VGGDecBN(nn.Module):
    def __init__(self, conf):
        super(VGGDecBN, self).__init__()
        self.conf = conf
        self.depth = conf.depth
        
        ch_pre = 512         # 512 (default)
        ch_ini = conf.ch_ini # 512 (default)

        # input size: [14x14]
        if self.depth > 4:
            # Hout = (Hin−1)×stride[0] − 2×padding[0] + 
            #        dilation[0]×(kernel_size[0]−1) + 
            #        output_padding[0]+1
            #
            # output_padding = 0
            # dilation       = 0
            
            ch_inp = ch_pre
            ch_out = ch_ini          # 512
            ch_cat = 2 * ch_out # 1024

            #self.unconv4 = nn.ConvTranspose2d(
            #    in_channels    = ch_inp, 
            #    out_channels   = ch_out, 
            #    kernel_size    = 4, 
            #    stride         = 2, 
            #    padding        = 1,
            #    dilation       = 1
            #)

            self.unconv4 = self.upconv_block(ch_inp, ch_out)
            
            # torch.cat: ch_out + ch_pre
            if self.conf.ch_ini != 512:
                self.conv4_red = nn.Sequential(
                    nn.Conv2d(ch_pre,ch_out,1,1,0,bias=True),
                    nn.BatchNorm2d(ch_out),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)       
                )
            self.conv_block4 = self.conv_block(3, ch_cat, ch_out)

        ch_ini = int(ch_ini/2) # 256
        ch_pre = int(ch_pre/2) # 256
        
        # input size: [28x28]
        if self.depth > 3:
            if self.depth == 4: 
                ch_inp = ch_pre * 2
            else:
                ch_inp = ch_out

            ch_out = ch_ini          # 256
            ch_cat = 2 * ch_out # 512
            
            #self.unconv3  = nn.ConvTranspose2d(
            #    in_channels  = ch_inp, 
            #    out_channels = ch_out, 
            #    kernel_size  = 4, 
            #    stride       = 2, 
            #    padding      = 1,
            #    dilation     = 1
            #)

            self.unconv3 = self.upconv_block(ch_inp, ch_out)

            # torch.cat: ch_out + ch_pre
            if self.conf.ch_ini != 512:
                self.conv3_red = nn.Sequential(
                    nn.Conv2d(ch_pre,ch_out,1,1,0,bias=True),
                    nn.BatchNorm2d(ch_out),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)       
                )
            self.conv_block3 = self.conv_block(2, ch_cat, ch_out)

        ch_ini = int(ch_ini/2) # 128
        ch_pre = int(ch_pre/2) # 128
        
        # input size: [56x56]
        if self.depth > 2:
            if self.depth == 3: 
                ch_inp = ch_pre * 2
            else:
                ch_inp = ch_out

            ch_out = ch_ini          # 128
            ch_cat = 2 * ch_out # 256

            #self.unconv2  = nn.ConvTranspose2d(
            #    in_channels  = ch_inp, 
            #    out_channels = ch_out, 
            #    kernel_size  = 4, 
            #    stride       = 2, 
            #    padding      = 1,
            #    dilation     = 1
            #)
            
            self.unconv2 = self.upconv_block(ch_inp, ch_out)

            # torch.cat: ch_out + ch_pre
            if self.conf.ch_ini != 512:
                self.conv2_red = nn.Sequential(
                    nn.Conv2d(ch_pre,ch_out,1,1,0,bias=True),
                    nn.BatchNorm2d(ch_out),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)       
                )
            self.conv_block2 = self.conv_block(2, ch_cat, ch_out)

        ch_ini = int(ch_ini/2) # 64
        ch_pre = int(ch_pre/2) # 64
        
        # input size: [112x112]
        if self.depth > 1:
            if self.depth == 2: 
                ch_inp = ch_pre * 2
            else:
                ch_inp = ch_out

            ch_out = ch_ini          # 64
            ch_cat = 2 * ch_out # 128

            #self.unconv1  = nn.ConvTranspose2d(
            #    in_channels  = ch_inp, 
            #    out_channels = ch_out, 
            #    kernel_size  = 4, 
            #    stride       = 2, 
            #    padding      = 1,
            #    dilation     = 1
            #)

            self.unconv1 = self.upconv_block(ch_inp, ch_out)

            # torch.cat: ch_out + ch_pre
            if self.conf.ch_ini != 512:
                self.conv1_red = nn.Sequential(
                    nn.Conv2d(ch_pre,ch_out,1,1,0,bias=True),
                    nn.BatchNorm2d(ch_out),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)       
                )
            self.conv_block1 = self.conv_block(2, ch_cat, ch_out)       

        # input size: [224x224]
        if self.depth == 1: 
            ch_inp = ch_pre
        else:
            ch_inp = ch_out
    
        #self.conv2img = nn.Conv2d(ch_inp,3,3,1,1,padding_mode ='reflect', bias=True)
        self.conv2img = nn.Conv2d(ch_inp,3,1,1,0,bias=True)
        self.out_act  = nn.Sigmoid()

    def upconv_block(self, ch_inp, ch_out):
        unconv = nn.ConvTranspose2d(
            in_channels  = ch_inp, 
            out_channels = ch_out, 
            kernel_size  = 4, 
            stride       = 2, 
            padding      = 1,
            dilation     = 1
        )  

        seq = [unconv]
        seq.append(nn.BatchNorm2d(ch_out))
        seq.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        return nn.Sequential(*seq)

    def conv_block(self, block_size, in_ch, ch_out):

        seq = []       
        seq.append(nn.Conv2d(in_ch,ch_out,3,1,1,bias=False,padding_mode ='reflect'))
        seq.append(nn.BatchNorm2d(ch_out))
        #seq.append(nn.ReLU(inplace=True))
        seq.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        seq.append(nn.Conv2d(ch_out,ch_out,3,1,1,bias=False,padding_mode ='reflect'))
        seq.append(nn.BatchNorm2d(ch_out))
        #seq.append(nn.ReLU(inplace=True))
        seq.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        if block_size > 2:
            seq.append(nn.Conv2d(ch_out,ch_out,3,1,1,bias=False,padding_mode ='reflect'))
            seq.append(nn.BatchNorm2d(ch_out))
            #seq.append(nn.ReLU(inplace=True))
            seq.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))   

        return nn.Sequential(*seq)

    def forward(self, layers):
        out = layers['z']

        if self.depth > 4:
            enc_out4   = layers['out4']
            if self.conf.ch_ini != 512:
                enc_out4 = self.conv4_red(enc_out4)
            out_unconv = self.unconv4(out)
            out_concat = torch.cat((out_unconv,enc_out4), dim=1)
            out        = self.conv_block4(out_concat)

        if self.depth > 3:
            enc_out3   = layers['out3']
            if self.conf.ch_ini != 512:
                enc_out3 = self.conv3_red(enc_out3)
            out_unconv = self.unconv3(out)
            out_concat = torch.cat((out_unconv,enc_out3), dim=1)    
            out        = self.conv_block3(out_concat)   

        if self.depth > 2:
            enc_out2   = layers['out2']
            if self.conf.ch_ini != 512:
                enc_out2 = self.conv2_red(enc_out2)
            out_unconv = self.unconv2(out)
            out_concat = torch.cat((out_unconv,enc_out2), dim=1)    
            out        = self.conv_block2(out_concat)   

        if self.depth > 1:
            enc_out1   = layers['out1']
            if self.conf.ch_ini != 512:
                enc_out1 = self.conv1_red(enc_out1)
            out_unconv = self.unconv1(out)
            out_concat = torch.cat((out_unconv,enc_out1), dim=1)    
            out        = self.conv_block1(out_concat)   

        if self.depth > 0:
            out = self.conv2img(out)
            out = self.out_act(out)

        return out  

class VGGEnc(nn.Module):
    def __init__(self, conf):
        super(VGGEnc, self).__init__()

        self.depth = conf.depth
        self.pretr = conf.pretr

        vgg16_model = models.vgg16(pretrained=self.pretr)
        feat_list   = list(vgg16_model.features)
        
        #self.conv_pre = nn.Conv2d(3,3,1,1,0,bias=True)

        #for p in feat_list[0].parameters():
        #    p.requires_grad = False
        #for p in feat_list[2].parameters():
        #    p.requires_grad = False

        if self.depth > 0:
            # Size: [224x224]
            self.conv1_1 = feat_list[0]
            self.relu1_1 = feat_list[1] 
            self.conv1_2 = feat_list[2]
            self.relu1_2 = feat_list[3]  

        if self.depth > 1:
            self.maxp2_1 = feat_list[4]  
            
            # Size: [112x112]
            self.conv2_1 = feat_list[5]
            self.relu2_1 = feat_list[6] 
            self.conv2_2 = feat_list[7]
            self.relu2_2 = feat_list[8]                

        if self.depth > 2:
            self.maxp3_1 = feat_list[9] 

            # Size: [56x56]
            self.conv3_1 = feat_list[10]
            self.relu3_1 = feat_list[11] 
            self.conv3_2 = feat_list[12]
            self.relu3_2 = feat_list[13] 
            self.conv3_3 = feat_list[14]
            self.relu3_3 = feat_list[15] 

        if self.depth > 3:
            self.maxp4_1 = feat_list[16] 

            # Size: [28x28]
            self.conv4_1 = feat_list[17]
            self.relu4_1 = feat_list[18] 
            self.conv4_2 = feat_list[19]
            self.relu4_2 = feat_list[20] 
            self.conv4_3 = feat_list[21]
            self.relu4_3 = feat_list[22] 

        if self.depth > 4:
            self.maxp5_1 = feat_list[23] 
                
            # Size: [14x14]
            self.conv5_1 = feat_list[24]
            self.relu5_1 = feat_list[25] 
            self.conv5_2 = feat_list[26]
            self.relu5_2 = feat_list[27] 
            self.conv5_3 = feat_list[28]
            self.relu5_3 = feat_list[29]   

    def forward(self, input):
        layers = {}
        
        #out    = self.conv_pre(input)

        out    = self.conv1_1(input)
        out    = self.relu1_1(out)
        out    = self.conv1_2(out)
        out1   = self.relu1_2(out)

        if self.depth < 2: 
            layers['z'] = out1
            return layers
        layers['out1'] = out1

        out   = self.maxp2_1(out1)
        out   = self.conv2_1(out)
        out   = self.relu2_1(out)
        out   = self.conv2_2(out)
        out2  = self.relu2_2(out)

        if self.depth < 3: 
            layers['z'] = out2
            return layers
        layers['out2'] = out2

        out   = self.maxp3_1(out2)
        out   = self.conv3_1(out)
        out   = self.relu3_1(out)
        out   = self.conv3_2(out)
        out   = self.relu3_2(out)
        out   = self.conv3_3(out)
        out3  = self.relu3_3(out)

        if self.depth < 4: 
            layers['z'] = out3
            return layers
        layers['out3'] = out3
        
        out   = self.maxp4_1(out3)
        out   = self.conv4_1(out)
        out   = self.relu4_1(out)
        out   = self.conv4_2(out)
        out   = self.relu4_2(out)
        out   = self.conv4_3(out)
        out4  = self.relu4_3(out)

        if self.depth < 5: 
            layers['z'] = out4
            return layers
        layers['out4'] = out4
        
        out = self.maxp5_1(out4)
        out = self.conv5_1(out)
        out = self.relu5_1(out)
        out = self.conv5_2(out)
        out = self.relu5_2(out)
        out = self.conv5_3(out)
        out = self.relu5_3(out)

        layers['z'] = out

        return layers

class VGGDec(nn.Module):
    def __init__(self, conf):
        super(VGGDec, self).__init__()
        self.depth = conf.depth
        self.conf  = conf

        ch_pre = 512         # 512 (default)
        ch_ini = conf.ch_ini # 512 (default)

        # input size: [14x14]
        if self.depth > 4:
            ch_inp = ch_pre          # 512
            ch_out = ch_ini          # 512
            ch_cat = 2 * ch_out      # 1024

            self.unconv4 = nn.ConvTranspose2d(
                in_channels  = ch_inp, 
                out_channels = ch_out, 
                kernel_size  = 4, 
                stride       = 2, 
                padding      = 1,
                dilation     = 1
            )
                
            # torch.cat: ch_out + ch_pre
            if self.conf.ch_ini != 512:
                self.conv4_red = nn.Conv2d(ch_pre,ch_out,1,1,0,bias=True)
            self.conv_block4 = self.conv_block(3, ch_cat, ch_out)

        ch_ini = int(ch_ini/2) # 256
        ch_pre = int(ch_pre/2) # 256
        
        # input size: [28x28]
        if self.depth > 3:
            if self.depth == 4: 
                ch_inp = ch_pre * 2
            else:
                ch_inp = ch_out

            ch_out = ch_ini     # 256
            ch_cat = 2 * ch_out # 512

            self.unconv3 = nn.ConvTranspose2d(
                in_channels  = ch_inp, 
                out_channels = ch_out, 
                kernel_size  = 4, 
                stride       = 2, 
                padding      = 1,
                dilation     = 1
            )
                
            # torch.cat: ch_hid + ch_pre
            if self.conf.ch_ini != 512:
                self.conv3_red = nn.Conv2d(ch_pre,ch_out,1,1,0,bias=True)
            self.conv_block3 = self.conv_block(2, ch_cat, ch_out)

        ch_ini = int(ch_ini/2) # 128
        ch_pre = int(ch_pre/2) # 128

        # input size: [56x56]
        if self.depth > 2:
            if self.depth == 3: 
                ch_inp = ch_pre * 2
            else:
                ch_inp = ch_out

            ch_out = ch_ini     # 128
            ch_cat = 2 * ch_out # 256

            self.unconv2 = nn.ConvTranspose2d(
                in_channels  = ch_inp, 
                out_channels = ch_out, 
                kernel_size  = 4, 
                stride       = 2, 
                padding      = 1,
                dilation     = 1)
                
            # torch.cat: ch_hid + ch_pre
            if self.conf.ch_ini != 512:
                self.conv2_red = nn.Conv2d(ch_pre,ch_out,1,1,0,bias=True)
            self.conv_block2 = self.conv_block(2, ch_cat, ch_out)

        ch_ini = int(ch_ini/2) # 64
        ch_pre = int(ch_pre/2) # 64

        # input size: [128x128]
        if self.depth > 1:
            if self.depth == 2: 
                ch_inp = ch_pre * 2
            else:
                ch_inp = ch_out

            ch_out = ch_ini     # 64
            ch_cat = 2 * ch_out # 128

            self.unconv1 = nn.ConvTranspose2d(
                in_channels  = ch_inp, 
                out_channels = ch_out, 
                kernel_size  = 4, 
                stride       = 2, 
                padding      = 1,
                dilation     = 1)
                
            # torch.cat: ch_hid + ch_pre
            if self.conf.ch_ini != 512:
                self.conv1_red = nn.Conv2d(ch_pre,ch_out,1,1,0,bias=True)
            self.conv_block1 = self.conv_block(2, ch_cat, ch_out)  

        # input size: [224x224]
        if self.depth == 1: 
            ch_inp = ch_pre
        else:
            ch_inp = ch_out

        #self.conv2img = nn.Conv2d(ch_inp,3,3,1,1,padding_mode ='reflect',bias=True)
        self.conv2img = nn.Conv2d(ch_inp,3,1,1,0,bias=True)
        self.out_act  = nn.Sigmoid()
                                    
    def conv_block(self, block_size, in_ch, out_ch):
        seq = []       
        seq.append(nn.Conv2d(in_ch,out_ch,3,1,1,bias=True,padding_mode ='reflect'))
        seq.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        seq.append(nn.Conv2d(out_ch,out_ch,3,1,1,bias=True,padding_mode ='reflect'))
        seq.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        if block_size > 2:
            seq.append(nn.Conv2d(out_ch,out_ch,3,1,1,bias=True,padding_mode ='reflect'))
            seq.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))   

        return nn.Sequential(*seq)

    def forward(self, layers):
        out = layers['z']

        if self.depth > 4:
            enc_out4   = layers['out4']
            if self.conf.ch_ini != 512:
                enc_out4 = self.conv4_red(enc_out4)
            out_unconv = self.unconv4(out)
            out_concat = torch.cat((out_unconv,enc_out4), dim=1)
            out        = self.conv_block4(out_concat)

        if self.depth > 3:
            enc_out3   = layers['out3']
            if self.conf.ch_ini != 512:
                enc_out3 = self.conv3_red(enc_out3)
            out_unconv = self.unconv3(out)
            out_concat = torch.cat((out_unconv,enc_out3), dim=1)    
            out        = self.conv_block3(out_concat)  

        if self.depth > 2:
            enc_out2   = layers['out2']
            if self.conf.ch_ini != 512:
                enc_out2 = self.conv2_red(enc_out2)
            out_unconv = self.unconv2(out)
            out_concat = torch.cat((out_unconv,enc_out2), dim=1)    
            out        = self.conv_block2(out_concat)   

        if self.depth > 1:
            enc_out1   = layers['out1']
            if self.conf.ch_ini != 512:
                enc_out1 = self.conv1_red(enc_out1)
            out_unconv = self.unconv1(out)
            out_concat = torch.cat((out_unconv,enc_out1), dim=1)    
            out        = self.conv_block1(out_concat)   

        out = self.conv2img(out)
        #out = self.out_act(out)

        return out  
