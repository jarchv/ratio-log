import os
import glob
import numpy as np
import pandas as pd

from tools.config import Config
from tools.measures import get_measures

def comp_op(conf):
    img_ambnt_path = 'datasets/%s/%s/*/*ambient.png' % (
            conf.dataset_path, 
            conf.eval_data
            )
            
    if 'improc' in conf.model:

        measures_path = os.path.join(
                conf.root_path,
                'exp',
                conf.model.split('_')[0]
                )

        eval_folder = os.path.join(
                conf.root_path,
                'exp',
                conf.model.split('_')[0],
                'eval'
                )

        mesures_dir = '%s/measures.csv' % measures_path
    else:      
        measures_path = os.path.join(
                conf.root_path,
                'exp',
                conf.model,
                'try-%d' % conf.try_num,
                'depth_%d' % conf.depth,
                'measures'
                )

        eval_folder = os.path.join(
                conf.root_path,
                'exp',
                conf.model,
                'try-%d' % conf.try_num,
                'depth_%d' % conf.depth,
                'eval',
                'ep-%d' % (conf.load_epoch)
                )

        mesures_dir = '%s/ep-%d.csv' % (measures_path, conf.load_epoch)

    if not os.path.exists(measures_path):
        os.makedirs(measures_path)
 


    tar_files = []
    inp_files = []
    out_files = []
    filenames = []

    ambnt_files = glob.glob(img_ambnt_path)
    ambnt_files.sort()

    for filename in ambnt_files:
        substr = filename[:-11].split('/')[-1]
        tar_files.append(filename)
        inp_files.append('%sflash.png' % filename[:-11])
        out_files.append('%s/%ssynth.png' % (eval_folder, substr))
        filenames.append(substr[:-1])

    psnr = []
    ssim = []
    loe  = []
    loet = []

    for inp_file, tar_file, out_file in zip(inp_files, tar_files, out_files):
        m1, m2, m3, m4 = get_measures(inp_file, tar_file, out_file)

        ssim.append(m1)
        psnr.append(m2)
        loe.append(m3)
        loet.append(m4)
        print('\rComputing [SSIM, PNSR, LOE, LOET] on {:s} {:3d}/{:3d}... '.format(
                conf.model,
                len(psnr),
                len(out_files)), 
                end=''
                )

    df = pd.DataFrame({
                'file': filenames, 
                'SSIM': ssim, 
                'PSNR': psnr,
                'LOE': loe,
                'LOET': loet
                })
                    
    df.to_csv(mesures_dir)
    print('Done.') 
    print('\rSSIM: {:.4f}, PSNR: {:.3f}, LOE: {:.2f}, LOET: {:.2f}'.format(
                                        np.mean(ssim), 
                                        np.mean(psnr), 
                                        np.mean(loe),
                                        np.mean(loet)
                                        ))

if __name__ == '__main__':
    conf = Config().parse()
    comp_op(conf)
