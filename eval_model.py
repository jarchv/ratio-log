import os
import numpy as np
import time

from models.models import set_model

from tools.config import Config
from tools.utils import *


def eval_op(model, conf):
    eval_path = os.path.join(
            conf.root_path,
            'exp',
            conf.model,
            'try-%d' % conf.try_num,
            'depth_%d' % conf.depth,
            'eval',
            'ep-%d'% conf.load_epoch
            )

    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

#   Open eval list of filenames
    eval_files, eval_data_ob = open_pairs(conf.dataset_path, conf.eval_data)
    eval_num  = len(eval_data_ob)

    t_start = time.time()
    if conf.model == 'DF':
        save_filtered_pairs_test(conf.root_path, eval_data_ob)
        eval_pairs = get_eval_pairs(eval_data_ob, False)
        eval_pairs_bf = get_eval_inp_bf(conf.root_path, eval_files, conf.eval_data, conf.pretr)
    else:
        eval_pairs = get_eval_pairs(eval_data_ob, pretrained=conf.pretr)

    t_end  = time.time()
    print("get_valid time: {:.3f}".format(t_end-t_start))

    t_start = time.time()
    for eval_pair in eval_pairs:
        flash_batch = eval_pair.inp
        
        if conf.model == 'DF':
            flash_batch_bf = eval_pairs_bf[eval_pair.file_]
            model.pred(inputs=flash_batch, inputs_fil=flash_batch_bf)

        else:
            model.pred(inputs=flash_batch)
            
        save_result(eval_path, model.fake_Y, eval_pair.file_)
        print('\rSaving results :{:3d}/{:3d}... '.format(
            eval_pair.count,
            eval_num), end='')
    print('Done.')
    t_end  = time.time()
    print('\rAverage time {:.3f} sec, eval_path={:s}'.format(
        (t_end - t_start) / eval_num,
        eval_path))

if __name__ == "__main__":
    # Get parameters
    conf  = Config().parse()
    
    # Build model, load, and run test
    print('Evaluating {} model '.format(conf.model))
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
    model = set_model(conf, checkpoints_path, train_mode=False)
    eval_op(model, conf)
