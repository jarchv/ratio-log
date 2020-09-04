
from models.models import set_model
from tools.config import *
from tools.utils import *

import numpy as np
import time
import os
import pandas as pd

def train_op(model, conf):
    tv_path = os.path.join(
            conf.root_path,
            'exp',
            conf.model,
            'try-%d' % conf.try_num,
            'depth_%d' % conf.depth
            )
            
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    if not os.path.exists(tv_path):
        os.makedirs(tv_path)
    
    #save_conf(tv_path, conf)
    format_train = '\rEpoch {:3d}[{:4d}/{:4d}]$ rec_loss(train)={:.4f}'
    format_gan   = ', gen={:.3f}, dis={:.3f}[{:.3f}(f) + {:.3f}(r)]'

    format_valid = ', rec_loss(valid)={:.4f}'
    format_psnr   = ', PSNR={:.4f} in {:.1f} sec.'

    pair_data = read_images(conf)
 
    if conf.model == 'DF':
        #save_filtered_pairs(conf.root_path, pair_data.train_data, pair_data.valid_data)
        valid_pairs = get_eval_pairs(pair_data.valid_data, False) # get valid_pairs as list
        train_data_fil = open_BF_pairs(conf.root_path, pair_data.train_files) # get im_ob filtered
        valid_pairs_bf = get_eval_inp_bf(conf.root_path, pair_data.valid_files, conf.eval_data, conf.pretr) #get valid inp as list
    else:
        valid_pairs = get_eval_pairs(pair_data.valid_data, conf.pretr)

#   Crete a blank csv
    loss_dir = '%s/graph.csv' % (tv_path)
    
    if os.path.exists(loss_dir):
        df = pd.read_csv(loss_dir) 
    else:
        df = pd.DataFrame({})

    print()            
    for ep in range(conf.load_epoch + 1, conf.epochs + 1):
        start_t = time.time()
        model.Gen.train()

        train_loss = []
        gen_loss = []
        dis_loss = []
        dis_loss_fake = []
        dis_loss_real = []
        
        if conf.model == 'DF':
            batches = batch_gen_df(
                conf, 
                pair_data.train_data,
                train_data_fil, 
                conf.crop_size, 
                conf.batch_size,
                conf.pretr
                )
        else:
            batches = batch_gen(
                pair_data.train_data, 
                conf.crop_size, 
                conf.batch_size,
                conf.pretr
                )

        for pair_batch in batches:
            flash_batch = pair_batch.inp
            ambnt_batch = pair_batch.tar
            model.set_inp_and_tar(flash_batch, ambnt_batch)

            if conf.model == 'DF':
                flash_batch_fil = pair_batch.inp_bf
                ambnt_batch_fil = pair_batch.tar_bf	
                model.set_inp_and_tar_fil(
                    flash_batch_fil,
                    ambnt_batch_fil
                    )
            
            model.optimize_parameters()
            count = pair_batch.count 
            train_loss.append(model.rec_loss.cpu().detach().numpy())

            print_it = format_train.format(
                ep, 
                count, 
                pair_data.train_num, 
                train_loss[-1]
            )

            if 'GAN' in conf.model:
                gen_loss.append(model.gen_loss.cpu().detach().numpy())
                dis_loss_fake.append(model.dis_loss_fake.cpu().detach().numpy())
                dis_loss_real.append(model.dis_loss_real.cpu().detach().numpy())
                dis_loss.append(model.dis_loss.cpu().detach().numpy())
                
                print_it += format_gan.format(
                        gen_loss[-1],
                        dis_loss[-1],
                        dis_loss_fake[-1],
                        dis_loss_real[-1]
                )

            print(print_it, end='')

            #display_imgs([
            #    model.real_X_inrange.cpu().detach().numpy()[0],
            #    model.real_Y.cpu().detach().numpy()[0],
            #    model.att_map.cpu().detach().numpy()[0]
            #    ])
                    
        train_loss_mean = np.mean(train_loss)
        print_train = format_train.format(
            ep, 
            count, 
            pair_data.train_num, 
            train_loss_mean
            )

        
        if 'GAN' in conf.model:
            gen_loss_mean = np.mean(gen_loss)
            dis_loss_mean = np.mean(dis_loss)
            dis_loss_fake_mean = np.mean(dis_loss_fake)
            dis_loss_real_mean = np.mean(dis_loss_real)

            print_train += format_gan.format(
                gen_loss_mean,
                dis_loss_mean,
                dis_loss_fake_mean,
                dis_loss_real_mean
                )

        print("{:s}".format(print_train), end='')
        model.Gen.eval()

        valid_loss = []
        valid_psnr  = []

        for valid_pair in valid_pairs:
            flash_batch = valid_pair.inp
            ambnt_batch = valid_pair.tar

            if conf.model == 'DF':
                flash_batch_fil = valid_pairs_bf[valid_pair.file_]
                model.pred(
                    inputs  = flash_batch, 
                    targets = ambnt_batch,
                    inputs_fil  = flash_batch_fil
                )
            else:
                model.pred(
                    inputs  = flash_batch, 
                    targets = ambnt_batch
                )  

            psnr_it  = model.compute_psnr()
            
            if 'RATIO' in conf.model:
                model.compute_ratio()
                model.compute_rec_loss()
                loss_it = model.rec_loss.cpu().detach().numpy()
            
            elif 'DF' == conf.model:
                loss_it = psnr_it
            
            valid_loss.append(loss_it)
            valid_psnr.append(psnr_it)
                              
        valid_loss_mean = np.mean(valid_loss)   
        valid_psnr_mean  = np.mean(valid_psnr)

        end_t = time.time()        
        print_valid = format_valid.format(valid_loss_mean)

        print_valid+= format_psnr.format(valid_psnr_mean, end_t-start_t)
        print('{:s}{:s}'.format(print_train, print_valid)) 

        if 'GAN' not in conf.model:
            gen_loss_mean = None
            dis_loss_mean = None
            dis_loss_fake_mean = None
            dis_loss_real_mean = None

        df_tmp = pd.DataFrame({
            'gen_loss': gen_loss_mean,
            'dis_loss': dis_loss_mean,
            'dis_loss_fake': dis_loss_fake_mean,
            'dis_loss_real': dis_loss_real_mean,
            'train_loss': train_loss_mean, 
            'valid_loss': valid_loss_mean,
            'valid_psnr'  : valid_psnr_mean,
        }, index=[ep])

        if 'epoch' in df:
            if ep in df.index: df = df.update(df_tmp)
            else: df = df.append(df_tmp) 
        else: df = df.append(df_tmp) 
         
        df.to_csv(loss_dir)
        
#       Save model each {conf.save_epoch} epochs
        if ep % conf.save_epoch == 0: 
            model.save_model(checkpoints_path, ep)

if __name__ == '__main__':

#   Get parameters & build model
    conf  = Config().parse()
    checkpoints_path = os.path.join(
            conf.root_path,
            'exp', 
            conf.model, 
            'try-%d' % conf.try_num,
            'depth_%d' % conf.depth,
            'checkpoints'
    )
    model = set_model(conf, checkpoints_path, train_mode=True)

#   Loading model at epoch {conf.load_epoch}
#    model.load_model()
    
#   Training
    train_op(model,conf)
