import os
import numpy as np
import time

from models.models import set_model

from tools.config import *
from tools.utils import *

from PIL import Image

def eval_op(model, conf):
    
    eval_path = os.path.join(
            conf.root_path,
            'exp',
            conf.model,
            'eval_samples'
            )

    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    t_start = time.time()

    sample_obj_img = Image.open(conf.sample_dir)
    sample_img = to_net(sample_obj_img, conf.pretr)

    print('Generating synthetic ambient image... ')
    model.pred(inputs=[sample_img])
    eval_dir = os.path.join(eval_path, conf.sample_dir.replace('flash', 'synth'))
    save_result(eval_path, model.fake_Y, eval_dir)
    print('Done.')

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
