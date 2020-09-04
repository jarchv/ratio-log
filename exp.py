import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import zipfile
import os
import sys

from tools.config import Config

def subset_measures(conf, show):
    mean_ep = []
    std_ep  = []
    
    df_out = dict()
    for try_ in [conf.try_num]:
        measures_dir = os.path.join(
                conf.root_path,
                'exp', 
                conf.model, 
                'try-%d' % (try_),
                'depth_%d' % conf.depth,
                'measures',
                'ep-%d.csv' % conf.load_epoch
                )
            
        df = pd.read_csv(measures_dir)
        
        measures = ['SSIM', 'PSNR', 'LOE', 'LOET']
        subsets  = [
            'Total',
            'Objects',
            'People',
            'Plants',
            'Rooms',
            'Shelves',
            'Toys'
            ]    
        
        for msr in measures:
            mean_var = df[msr].mean()
            df_out['_'.join([str(try_), 'Total', msr])] = mean_var
            for set_ in subsets[1:]:
                data_subset = df[df['file'].str.contains(set_)]
                mean_var = data_subset[msr].mean()           
                df_out['_'.join([str(try_), set_, msr])] = mean_var

    fmt = 'set: {0} - {1}[mean={2:.{3}f}, std={4:.{5}f}]'
    msr_subsets = dict()
    for set_ in subsets:
        for msr in measures:
            k = '_'.join(['%d', set_, msr])
            res = [df_out[k % try_] for try_ in [conf.try_num]]
            mean_res = np.mean(res)
            std_res  = np.std(res) 
            msr_subsets[(set_, msr, 'mean')] = mean_res
            msr_subsets[(set_, msr, 'std')] = std_res
            
            if msr == 'PSNR':
                print(fmt.format(set_, msr, mean_res, 2, std_res, 3))
            elif msr == 'SSIM':
                print(fmt.format(set_, msr, mean_res, 3, std_res, 4))
            else:
                print(fmt.format(set_, msr, mean_res, 1, std_res, 2))
           
    if not show: return None
    
    plt.rc('font', size=8) 
    fig, axs = plt.subplots(4,1,figsize=(5,6))
    fig.subplots_adjust(hspace=0.5) 
    fig.suptitle(measures_dir)
    
    c = ['blue', 'orange', 'red', 'purple']
    for i, msr in enumerate(measures):
        axs[i].bar(
            subsets, 
            [msr_subsets[(set_, msr, 'mean')] for set_ in subsets],
            color=c[i])
        axs[i].set_title(msr +': '+ 'mean' )
    
    plt.show()

def plot_train_vs_valid(conf):
    tv_path = os.path.join(
            conf.root_path,
            'exp',
            conf.model,
            'try-%d' % conf.try_num,
            'depth_%d' % conf.depth
            )

    loss_dir = '%s/graph.csv' % (tv_path)
    df = pd.read_csv(loss_dir)
    df.plot(use_index = True, 
            y = ['dis_loss_real', 'dis_loss_fake'], 
            title = '{:s}: pretr={:d} and bn={:d}'.format(conf.model, conf.pretr, conf.bn, conf.att_rec),
            xticks = range(0,conf.epochs+1, 20),
            ylim = (0.0, 1.0),
            figsize = (6,4),
            )
    plt.xlabel("Epochs")
    plt.ylabel("Recontruction Loss")
    plt.show()
    
if __name__ == '__main__':
    conf  = Config().parse()
    subset_measures(conf, show=True)
    #plot_train_vs_valid(conf)

