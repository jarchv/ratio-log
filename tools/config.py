import argparse
import os

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--model', default='FCN', 
                    help="""model: FCN, GAN, DeepFlash, etc. For Image Processing
                        techniques add '_improc' at the end of the model's name.""")

#       Paths
        self.parser.add_argument('--dataset_path', default='FAIDres', help='dataset path')
        self.parser.add_argument('--root_path', default='./', 
                    help='path where checkpoints and eval samples will be saves')
        self.parser.add_argument('--eval_data', type=str, default='valid', 
                    help='path used to eval the model: train/valid/test')
        self.parser.add_argument('--sample_dir', type=str, 
                    help='filename of image to evaluate using eval_sample.py')  
#       Network
        self.parser.add_argument('--depth', type=int, default=5, 
                    help='depth of the FCN')
        self.parser.add_argument('--pretr', type=int, default=1, 
                    help='VGG pretrained')
        self.parser.add_argument('--out_act', type=str, default='sigmoid', 
                    help='final activation: sigmoid, tanh')
        self.parser.add_argument('--bn', type=int, default=0,
                    help='batch normalization: 0:active, 1:deactive')
        self.parser.add_argument('--att_rec', type=int, default=0,
                    help='attention mec. in the reconstruction loss')
        self.parser.add_argument('--att_gan', type=int, default=0,
                    help='attention mec. in the adversarial loss')
        self.parser.add_argument('--ch_ini', type=int, default=512,
                    help='number of output channels for first unconv: ch_ini=ch_ini/2**(5-depth)')
                    
#       Images
        self.parser.add_argument('--load_size', type=int, default=240, 
                    help='crop step')
        self.parser.add_argument('--crop_size', type=int, default=224, 
                    help='crop step')

#       Experiment
        self.parser.add_argument('--save_epoch', type=int, default=20, help='save each # epoch')
        self.parser.add_argument('--load_epoch', type=int, default=0,help='load at epoch #')
        self.parser.add_argument('--try_num', default=1, type=int, help="try number")

#       Hyperparameters
#
#       *
#       *       
        self.parser.add_argument('--batch_size', type=int, default=16, help='batch size')
        self.parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
        self.parser.add_argument('--rec_lr', type=float, default=1e-4, help='learning rate in rec_loss.')
        self.parser.add_argument('--gen_lr', type=float, default=1e-6, help='learning rate in Gen.')
        self.parser.add_argument('--dis_lr', type=float, default=1e-6, help='learning rate in Dis.')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
        self.parser.add_argument('--rec_loss', type=str, default='L2', help='Loss: L1, L2, ...')

    def parse(self):	
        """Parse configurations"""
        conf = self.parser.parse_args()
        return conf 


def save_conf(path, conf):
    config_path = os.path.join(path, 'config.txt')
    with open(config_path, 'w') as f:
        f.write('model      \t{:s}\n'.format(conf.model))
        f.write('pretr      \t{:d}\n'.format(conf.pretr))
        f.write('depth      \t{:d}\n'.format(conf.depth))
        f.write('out_act    \t{:s}\n'.format(conf.out_act))
        f.write('att_rec    \t{:d}\n'.format(conf.att_rec))
        f.write('att_gan    \t{:d}\n'.format(conf.att_gan))
        f.write('rec_lr     \t{:.2}\n'.format(conf.rec_lr))
        f.write('gen_lr     \t{:.2}\n'.format(conf.gen_lr))
        f.write('dis_lr     \t{:.2}\n'.format(conf.dis_lr))
        f.write('rec_loss   \t{:s}\n'.format(conf.rec_loss))
        f.write('bn         \t{:d}\n'.format(conf.bn))

