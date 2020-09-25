import torch

torch.cuda.manual_seed_all(20)
torch.autograd.set_detect_anomaly(True)

import torch.nn as nn
import os
import numpy as np
import math

from torchvision import transforms

from .nets import *

class BaseModel:
    def __init__(self):
        self.device = 'cuda:0'
        mean = np.asarray([[0.485, 0.456, 0.406]], np.float32)
        std  = np.asarray([[0.229, 0.224, 0.225]], np.float32)
        
        self.mean_var  = torch.cuda.FloatTensor(mean).unsqueeze(2).unsqueeze(3)
        self.std_var   = torch.cuda.FloatTensor(std).unsqueeze(2).unsqueeze(3)
        self.conv_smooth = self.get_gaussian_conv()

    def show_conf(self):
        print('\nmodel      \t{}'.format(self.conf.model))
        if self.train_mode:
            print('rec_loss    \t{}'.format(self.conf.rec_loss))
            print('epochs   \t{}'.format(self.conf.epochs))
            print('batch_size\t{}'.format(self.conf.batch_size))
            print('depth    \t{}'.format(self.conf.depth))
            print('pretr    \t{}'.format(self.conf.pretr))
            print('bn       \t{}'.format(self.conf.bn))
            print('att_rec  \t{:d}'.format(self.conf.att_rec))

            if self.dis_mode:
                print('att_gan  \t{:d}'.format(self.conf.att_gan))
            print('rec_lr	  \t{:.1e}'.format(float(self.conf.rec_lr)))

            if self.dis_mode:
                print('gen_lr  \t{:.1e}'.format(float(self.conf.gen_lr)))
                print('dis_lr  \t{:.1e}'.format(float(self.conf.dis_lr)))
            if self.conf.load_epoch > 0:
                print('load_ep \t{:d}'.format(self.conf.load_epoch))
        print()
		
    def set_inp(self, inputs):
        self.real_X = torch.cuda.FloatTensor(inputs)

    def set_inp_in_DP(self, inputs, inputs_bf):
        self.real_X = torch.cuda.FloatTensor(inputs)
        self.real_X_fil = torch.cuda.FloatTensor(inputs_bf)

    def set_inp_and_tar(self, inputs, targets):
        self.real_X = torch.cuda.FloatTensor(inputs)
        self.real_Y = torch.cuda.FloatTensor(targets)
        
    def set_inp_and_tar_fil(self, inputs_fil, targets_fil):
        self.real_X_fil = torch.cuda.FloatTensor(inputs_fil)
        self.real_Y_fil = torch.cuda.FloatTensor(targets_fil)

    def set_inp_and_tar_in_DP_valid(self, inputs, targets, inputs_bf):
        self.real_X   = torch.cuda.FloatTensor(inputs)
        self.real_Y   = torch.cuda.FloatTensor(targets)
        self.real_X_fil = torch.cuda.FloatTensor(inputs_bf)     

    def get_inputs_to_net(self):
        norm = self.real_X * self.std_var + self.mean_var
        return norm.expand_as(self.real_X)
        
    def set_requires_grad(self, net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad

    def save_model(self, checkpoints_path, ep):
        print('Saving "model-{:d}"... '.format(ep), end='')

        file_model = 'model-{:d}.pth'.format(ep)
        save_path  = os.path.join(checkpoints_path, file_model)
		
        checkpoint = {}

        if self.conf.model == 'GAN':
            checkpoint['state_dict_gen'] = self.Gen.state_dict()
            checkpoint['state_dict_dis'] = self.Dis.state_dict()
            checkpoint['optimizer_gen']  = self.optimizer_gen.state_dict()
            checkpoint['optimizer_dis']  = self.optimizer_dis.state_dict()
        
        else:
            checkpoint['state_dict_gen'] = self.Gen.state_dict()   
        checkpoint['optimizer_rec'] = self.optimizer_rec.state_dict()
             
        torch.save(checkpoint, save_path)
        print("Done.")

    def load_model(self, checkpoints_path, ep):
        print('\nLoading "model-{:d}"... '.format(ep), end='')
        file_model = 'model-{:d}.pth'.format(ep)

        load_path  = os.path.join(checkpoints_path, file_model)
        checkpoint = torch.load(load_path)

        self.Gen.load_state_dict(checkpoint['state_dict_gen'])
        #self.Gen.load_state_dict(checkpoint)

        if self.train_mode:
            if self.conf.model == 'GAN':
                self.Dis.load_state_dict(checkpoint['state_dict_dis'])
                self.optimizer_gen.load_state_dict(checkpoint['optimizer_gen'])
                self.optimizer_dis.load_state_dict(checkpoint['optimizer_dis'])

            self.optimizer_rec.load_state_dict(checkpoint['optimizer_rec'])
        print("Done.")

    def pred(self, inputs, targets=[], inputs_fil=[]):

        if len(targets) != 0:
            self.set_inp_and_tar(inputs, targets)
        else:
            self.set_inp(inputs)

        if self.conf.model == 'DF':
            self.real_X_fil = torch.cuda.FloatTensor(inputs_fil)

        self.forward()
        if self.conf.model in ['RATIO','DF','RATIOLOG','RATIOLOGGAN']:
            self.compute_fake_Y()
    
    def compute_psnr(self):
        mse = np.mean(np.square(self.fake_Y.detach().cpu().numpy() - self.real_Y.detach().cpu().numpy()))
        return 10. * np.log10(1. / mse)

    def build_real_X_inrange(self):
        self.real_X_inrange = self.real_X * self.std_var + self.mean_var
    
    def build_attention_map(self, gamma=10.0):
        if self.conf.pretr == True:
            self.build_real_X_inrange()
        else:
            self.real_X_inrange = self.real_X

        self.diff = self.conv_smooth(self.real_Y) - self.conv_smooth(self.real_X_inrange)
        self.att_map = torch.abs(self.diff) 

        #self.att_map = 1. / (torch.exp( -gamma * self.diff) + 1.)

    def build_model(self):
        if self.train_mode:
            if self.conf.model == 'UNET':
                self.Gen = UNetChen()
            else:
                if self.conf.bn == 0:
                    self.Gen = VGGED(conf=self.conf)
                else:
                    self.Gen = VGGED_BN(conf=self.conf)

            self.Gen.to(self.device)

            print('Gen$\n')
            print(self.Gen)
            print('\nGen$\n')

            if   self.conf.rec_loss == 'L1': self.criterion = torch.nn.L1Loss()
            elif self.conf.rec_loss == 'L2': self.criterion = torch.nn.MSELoss()
            
            #for p in filter(lambda p: p.requires_grad, self.Gen.parameters()):
            #    print(p.shape, p.requires_grad)
            
            self.optimizer_rec = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.Gen.parameters()), 
                lr = self.conf.rec_lr, 
                betas = (self.conf.beta1, 0.999)
            )

            if self.dis_mode:
                self.Dis = DisNet()
                self.Dis.to(self.device)

                self.criterion_gan = GANLoss()
                self.criterion_gan.to(self.device)

                print('Dis$\n')
                print(self.Dis)
                print('\nDis$\n')

                self.optimizer_gen = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.Gen.parameters()), 
                    lr = self.conf.gen_lr, 
                    betas = (self.conf.beta1, 0.999))
                                              
                self.optimizer_dis = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.Dis.parameters()), 
                    lr = self.conf.dis_lr, 
                    betas = (self.conf.beta1, 0.999))

            if self.conf.load_epoch > 0:
                self.load_model(self.checkpoints_path, self.conf.load_epoch)

            self.Gen.train()

            print('Training settings:')
            self.show_conf()

        else:
            print('Eval settings:')
            if self.conf.model == 'UNET':
                self.Gen = UNetChen()
            else:
                if self.conf.bn == 0:
                    self.Gen = VGGED(conf=self.conf)
                else:
                    self.Gen = VGGED_BN(conf=self.conf)
            self.load_model(self.checkpoints_path, self.conf.load_epoch)
            self.Gen.to(self.device)

    def get_gaussian_conv(self, kernel_size=5, sigma=2, channels=3):
        v = torch.arange(kernel_size)
        g = v.repeat(kernel_size).view(kernel_size, kernel_size)
        g_t = g.t()
        gg_t = torch.stack([g, g_t], dim=-1)

        mu  = (kernel_size - 1.) / 2.0
        var = sigma ** 2.

        kernel = (1. / (2 * math.pi * var)) * \
                            torch.exp(-torch.sum((gg_t - mu) ** 2, dim=-1) / (2. * var))
        
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(channels, 1, 1, 1).to(self.device)

        conv = nn.Conv2d(
            in_channels  = channels, 
            out_channels = channels,
            kernel_size  = kernel_size, 
            groups       = channels, 
            bias         = False,
            padding      = int((kernel_size - 1)/2),
            padding_mode = 'reflect'
        )

        conv.weight.data = kernel
        conv.weight.requires_grad = False

        return conv

class FCN(BaseModel):
    def __init__(self, conf, checkpoints_path, train_mode=True):
        super(FCN, self).__init__()
        self.conf = conf
        self.checkpoints_path = checkpoints_path
        self.dis_mode = False
        self.train_mode = train_mode
        self.build_model()

    def forward(self):
        self.z, self.fake_Y = self.Gen(self.real_X)

    def compute_rec_loss(self):
        if self.conf.att_rec == True:
            self.build_attention_map()
            self.rec_loss = self.criterion(
                self.fake_Y * self.att_map, self.real_Y * self.att_map)
        else:
            self.rec_loss = self.criterion(self.fake_Y, self.real_Y)

    def optimize_parameters(self):
        self.forward()
        self.optimizer_rec.zero_grad()
        self.compute_rec_loss()
        self.rec_loss.backward()
        self.optimizer_rec.step()

class RATIO(BaseModel):
    def __init__(self, conf, checkpoints_path, train_mode=True):
        super(RATIO, self).__init__()
        self.conf = conf
        self.checkpoints_path = checkpoints_path
        self.dis_mode = False
        self.train_mode = train_mode
        self.build_model()

    def get_ratio(self):
        if self.conf.pretr == True:
            self.build_real_X_inrange()
        else:
            self.real_X_inrange = self.real_X

        self.raw_div = torch.div(self.real_Y + 1., self.real_X_inrange + 1.)
        self.real_ratio = (2.0 / 3.0 ) * self.raw_div - 1.0 / 3.0

    def compute_fake_Y(self):
        if self.conf.pretr == True:
            self.build_real_X_inrange()  
        else:
            self.real_X_inrange = self.real_X

        self.raw_div1 = 3.0 * (self.fake_ratio + self.real_X_inrange * self.fake_ratio) / 2.0
        self.raw_div2 = (self.real_X_inrange - 1.) / 2.0
        self.fake_Y = torch.clamp(self.raw_div1 + self.raw_div2, 0, 1)

    def forward(self):
        self.z, self.fake_ratio = self.Gen(self.real_X)
        
    def compute_rec_loss(self):
        if self.conf.att_rec == True:
            self.build_attention_map()
            self.rec_loss = self.criterion(
                self.fake_ratio * self.att_map, self.real_ratio * self.att_map)
        else:
            self.rec_loss = self.criterion(self.fake_ratio, self.real_ratio)

    def optimize_parameters(self):
        self.forward()
        self.compute_ratio()
        self.optimizer_rec.zero_grad()
        self.compute_rec_loss()
        self.rec_loss.backward()
        self.optimizer_rec.step()      

class RATIOLOG(BaseModel):
    def __init__(self, conf, checkpoints_path, train_mode=True):
        super(RATIOLOG, self).__init__()
        self.conf = conf
        self.checkpoints_path = checkpoints_path
        self.dis_mode = False
        self.train_mode = train_mode
        self.build_model()

    def compute_ratio(self):
        if self.conf.pretr == True:
            self.build_real_X_inrange()
        else:
            self.real_X_inrange = self.real_X

        self.raw_div = torch.div(self.real_Y + 1., self.real_X_inrange + 1.)
        self.real_ratio = torch.log2(self.raw_div) * 0.5 + 0.5

    def compute_fake_Y(self):
        if self.conf.pretr == True:
            self.build_real_X_inrange()  
        else:
            self.real_X_inrange = self.real_X

        self.exp_term = torch.pow(2., 2. * self.fake_ratio - 1.)
        self.fake_Y = torch.clamp((self.real_X_inrange + 1) * self.exp_term - 1., 0, 1)

    def forward(self):
        self.z, self.fake_ratio = self.Gen(self.real_X)

    def compute_rec_loss(self):        
        if self.conf.att_rec == True:
            self.att_map  = torch.exp(4.6 * torch.abs(self.real_ratio - 0.5)) / 10.0
            self.rec_loss = self.criterion(
                self.fake_ratio * self.att_map, self.real_ratio * self.att_map)
        else:
            self.rec_loss = self.criterion(self.fake_ratio, self.real_ratio)

    def optimize_parameters(self):
        self.forward()
        self.compute_ratio()

        self.optimizer_rec.zero_grad()
        self.compute_rec_loss()
        self.rec_loss.backward()
        self.optimizer_rec.step()  

class RATIOLOGGAN(BaseModel):
    def __init__(self, conf, checkpoints_path, train_mode=True):
        super(RATIOLOGGAN, self).__init__()
        self.conf = conf
        self.checkpoints_path = checkpoints_path
        self.dis_mode = True
        self.train_mode = train_mode
        self.build_model()
        self.mse = nn.MSELoss()

    def compute_ratio(self):
        if self.conf.pretr == True:
            self.build_real_X_inrange()
        else:
            self.real_X_inrange = self.real_X

        self.raw_div = torch.div(self.real_Y + 1., self.real_X_inrange + 1.)
        self.real_ratio = torch.log2(self.raw_div) * 0.5 + 0.5

    def compute_fake_Y(self):
        if self.conf.pretr == True:
            self.build_real_X_inrange()  
        else:
            self.real_X_inrange = self.real_X

        self.exp_term = torch.pow(2., 2. * self.fake_ratio - 1.)
        self.fake_Y = torch.clamp((self.real_X_inrange + 1) * self.exp_term - 1., 0, 1)

    def forward(self):
        self.z, self.fake_ratio = self.Gen(self.real_X)

    def compute_rec_loss(self):        
        if self.conf.att_rec == True:
            self.build_attention_map()
            self.rec_loss = self.criterion(
                self.fake_ratio * self.att_map, self.real_ratio * self.att_map)
        else:
            self.rec_loss = self.criterion(self.fake_ratio, self.real_ratio)

    def backward_gen(self):
        #self.gen_loss = self.criterion_gan(self.Dis(self.fake_Y), 'real') # -log(D(x_hat))

#        if self.conf.att_gan:
#            self.build_attention_map()
#            self.gen_loss = -self.criterion_gan(self.Dis(self.fake_ratio * self.att_map), 'fake') # log(1 - D(x_hat))
#        else:
#            self.gen_loss = -self.criterion_gan(self.Dis(self.fake_ratio), 'fake') # log(1 - D(x_hat))
        
        self.gen_loss = 0.5 * torch.mean(torch.square(self.Dis(self.fake_ratio)))
        self.gen_loss.backward(retain_graph=True)	

    def backward_dis(self):
        if self.conf.att_gan:
            self.build_attention_map()
            dis_out_fake = self.Dis(self.fake_ratio.detach() * self.att_map)
            dis_out_real = self.Dis(self.real_ratio * self.att_map)          
        else:
            dis_out_fake = self.Dis(self.fake_ratio.detach())
            dis_out_real = self.Dis(self.real_ratio)

#        self.dis_loss_fake = self.criterion_gan(dis_out_fake, 'fake') # -log(1-D(x_hat)))
#        self.dis_loss_real = self.criterion_gan(dis_out_real, 'real') # -log(D(x)))

        self.dis_loss_real = torch.mean(torch.square(dis_out_real - 1.))
        self.dis_loss_fake = torch.mean(torch.square(dis_out_fake + 1.))
        
        self.dis_loss = 0.5 * (self.dis_loss_fake + self.dis_loss_real)
        self.dis_loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.compute_ratio()

#       Discriminator: compute gradient and update weights
        self.set_requires_grad(self.Dis, True)  # enable grandients
        self.optimizer_dis.zero_grad()          # set gradients to zero
        self.backward_dis()					    # compute gradients	
        self.optimizer_dis.step()				# update weights in the dis.
        
#       Generator: compute gradient
        self.set_requires_grad(self.Dis, False) # disbale grandients in Dis
        self.optimizer_gen.zero_grad()   		# set gradients to zero
        self.backward_gen()                     # compute gradients, setting
                                                # retain_graph=True 


#       Update based on the Reconstruction Loss
        self.optimizer_rec.zero_grad()   	# set gradients to zero
        self.compute_rec_loss()             # compute gradients	
        self.rec_loss.backward()
        
        self.optimizer_rec.step()           # update weights in the rec.
        self.optimizer_gen.step()           # update weights in the gen. 

class GAN(BaseModel):
    def __init__(self, conf, checkpoints_path, train_mode=True):
        super(GAN, self).__init__()
        self.conf = conf
        self.checkpoints_path = checkpoints_path
        self.dis_mode = True
        self.train_mode = train_mode
        self.build_model()

    def forward(self):
        self.z, self.fake_Y = self.Gen(self.real_X)

    def compute_rec_loss(self):
        if self.conf.att_rec:
            self.build_attention_map()
            self.rec_loss = self.criterion(
                self.fake_Y * self.att_map, self.real_Y * self.att_map)
        else:
            self.rec_loss = self.criterion(self.fake_Y, self.real_Y)

    def backward_gen(self):
        #self.gen_loss = self.criterion_gan(self.Dis(self.fake_Y), 'real') # -log(D(x_hat))

        if self.conf.att_gan:
            self.build_attention_map()
            self.gen_loss = -self.criterion_gan(self.Dis(self.fake_Y * self.att_map), 'fake') # log(1 - D(x_hat))
        else:
            self.gen_loss = -self.criterion_gan(self.Dis(self.fake_Y), 'fake') # log(1 - D(x_hat))

        self.gen_loss.backward(retain_graph=True)	

    def backward_dis(self):
        if self.conf.att_gan:
            self.build_attention_map()
            dis_out_fake = self.Dis(self.fake_Y.detach() * self.att_map)
            dis_out_real = self.Dis(self.real_Y * self.att_map)          
        else:
            dis_out_fake = self.Dis(self.fake_Y.detach())
            dis_out_real = self.Dis(self.real_Y)

        self.dis_loss_fake = self.criterion_gan(dis_out_fake, 'fake') # -log(1-D(x_hat)))
        self.dis_loss_real = self.criterion_gan(dis_out_real, 'real') # -log(D(x)))

        self.dis_loss = 0.5 * (self.dis_loss_fake + self.dis_loss_real)
        self.dis_loss.backward()

    def optimize_parameters(self):
        self.forward()

#       Discriminator: compute gradient and update weights
        self.set_requires_grad(self.Dis, True)  # enable grandients
        self.optimizer_dis.zero_grad()          # set gradients to zero
        self.backward_dis()					    # compute gradients	
        self.optimizer_dis.step()				# update weights in the dis.
        
#       Generator: compute gradient
        self.set_requires_grad(self.Dis, False) # disbale grandients in Dis
        self.optimizer_gen.zero_grad()   		# set gradients to zero
        self.backward_gen()                     # compute gradients, setting
                                                # retain_graph=True 


#       Update based on the Reconstruction Loss
        self.optimizer_rec.zero_grad()   	# set gradients to zero
        self.compute_rec_loss()             # compute gradients	
        self.rec_loss.backward()
        
        self.optimizer_rec.step()           # update weights in the rec.
        self.optimizer_gen.step()           # update weights in the gen.    
            
class DeepFlash(BaseModel):
    def __init__(self, conf, checkpoints_path, train_mode=True):
        super(DeepFlash, self).__init__()
        self.conf = conf
        self.checkpoints_path = checkpoints_path
        self.dis_mode = False
        self.train_mode = train_mode
        
        if train_mode:
            self.Gen = VGGED_BN(conf)

            print('\nModel$\n')
            print(self.Gen)
            print('\nModel$\n')
            self.Gen.to(self.device)
            self.criterion = torch.nn.MSELoss()
            
            self.optimizer_rec = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.Gen.parameters()), 
                lr = conf.rec_lr, betas = (0.9, 0.999)) #lr = 1e-5 (Paper)

            print('Training settings:')
            print('\nmodel      \t{}'.format(self.conf.model))
            print('rec_loss    \t{}'.format(self.conf.rec_loss))
            print('rec_lr	  \t{:.2e}'.format(float(self.conf.rec_lr)))
            print('epochs   \t{}'.format(self.conf.epochs))
            print()

        else:
            self.Gen = VGGED_BN(conf)
            self.load_model(self.checkpoints_path, self.conf.load_epoch)
            self.Gen.to(self.device)
            
            print('Eval settings:')
        self.show_conf()

    def forward(self):
        self.z, self.y_i = self.Gen(self.real_X_fil)

    def compute_fake_Y(self):
        self.pred_i = self.real_X - self.y_i * 2.0 + 1.0 # pred_i
        self.fake_Y = torch.clamp(self.pred_i, 0.0, 1.0)
        
    def compute_rec_loss(self):
        self.real_X_fil_inrange = self.real_X_fil * self.std_var + self.mean_var
        self.t_i = (self.real_X_fil_inrange - self.real_Y_fil + 1.0) * 0.5 # g.t. diff

        mean_y_i = torch.mean(self.y_i.detach(), dim=(2,3), keepdim=True)
        mean_t_i = torch.mean(self.t_i, dim=(2,3), keepdim=True)

        self.rec_loss = self.criterion(self.y_i - mean_y_i, self.t_i - mean_t_i)
        
    def optimize_parameters(self):
        self.optimizer_rec.zero_grad()
        self.forward()
        self.compute_rec_loss()
        self.rec_loss.backward()
        self.optimizer_rec.step()


class UNET(BaseModel):
    def __init__(self, conf, checkpoints_path, train_mode=True):
        super(UNET, self).__init__()
        self.conf = conf
        self.checkpoints_path = checkpoints_path
        self.dis_mode = False
        self.train_mode = train_mode
        self.build_model()

    def forward(self):
        self.fake_Y_raw = self.Gen(self.real_X)
        self.fake_Y = torch.clamp(self.fake_Y_raw, 0, 1)

    def compute_rec_loss(self):
        self.rec_loss = self.criterion(self.fake_Y_raw, self.real_Y)
        
    def optimize_parameters(self):
        self.forward()
        self.optimizer_rec.zero_grad()
        self.compute_rec_loss()
        self.rec_loss.backward()
        self.optimizer_rec.step()

def set_model(conf, checkpoints_path, train_mode=True):
    if conf.model == 'FCN':
        return FCN(conf, checkpoints_path, train_mode)
    elif conf.model == 'RATIO':
        return RATIO(conf, checkpoints_path, train_mode)
    elif conf.model == 'RATIOLOG':
        return RATIOLOG(conf, checkpoints_path, train_mode)
    elif conf.model == 'RATIOLOGGAN':
        return RATIOLOGGAN(conf, checkpoints_path, train_mode)
    elif conf.model == 'GAN':
        return GAN(conf, checkpoints_path, train_mode)
    elif conf.model == 'DF':
        return DeepFlash(conf, checkpoints_path, train_mode)
    elif conf.model == 'UNET':
        return UNET(conf, checkpoints_path, train_mode)
    else:
        print('Non available model...')
