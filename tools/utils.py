from __future__ import division

import glob
import numpy as np
import os
import random
import math
import matplotlib.pyplot as plt
	
from PIL import Image
from PIL import ImageOps
from collections import namedtuple


def save_result(eval_path, img_fake, file_):
    img_fake = img_fake.cpu().detach().numpy()
    img_fake = _to_view(img_fake, mode='ob')

    if len(file_.split('/')) != 0:
        file_synth = "_".join([file_.split('/')[-1], 'synth.png'])
    else:
        file_synth = "_".join([file_.split('\\')[-1], 'synth.png'])

    img_fake.save(os.path.join(eval_path, file_synth))
    img_fake.close()
	 
def display_imgs(imgs):
    tmp = np.hstack([_to_view(img) for img in imgs])
		
    plt.imshow(tmp)
    plt.axis('off')
    plt.show(block=False)
    plt.pause(2)
    plt.close()	
	
def get_files(dataset_path, subset='train'):
    set_path  = os.path.join('datasets/', dataset_path, subset)
    flash_set = sorted(glob.glob('%s/*/*flash.png' % set_path))
    ambnt_set = sorted(glob.glob('%s/*/*ambient.png' % set_path))

    file_pairs = []
		
    for flash_file, ambnt_file in zip(flash_set, ambnt_set):
        assert (ambnt_file[:-12] == flash_file[:-10])
        file_pairs.append([flash_file, ambnt_file])
    print('Files[{:s}]: {:3d} pairs of filenames.'.format(subset, len(file_pairs)))
    return file_pairs

def _pretrained_inputs(img_inp):
    """
    Based on "https://pytorch.org/docs/stable/torchvision/models.html", we
    should normalize loaded images (into a range [0, 1]) using 
    mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]     
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])  
    img_inp_norm = (img_inp - mean) / std

    return img_inp_norm

def to_net(img_ob, pretrained = False):
    img_arr = np.asarray(img_ob, np.float32) / 255.0

    if pretrained:
        img_arr = _pretrained_inputs(img_arr)
    return np.transpose(img_arr, (2, 0, 1))
	
def _to_view(img_arr, mode=None):
    if img_arr.ndim == 4:
        img_arr = np.squeeze(img_arr, axis=0)
    img_arr = np.transpose(img_arr, (1, 2, 0))
    if mode != 'ob':
        return img_arr
    return Image.fromarray(np.asarray(img_arr * 255, dtype=np.uint8))

def open_pairs(dataset_path, subset = 'train'):
    file_pairs = get_files(dataset_path, subset)
    nfiles = len(file_pairs)
    pairs_ob = []

    for flash_file, ambnt_file in file_pairs:
        img_ob_flash = Image.open(flash_file)
        img_ob_ambnt = Image.open(ambnt_file)

        pairs_ob.append([
            ambnt_file[:-12], 
            img_ob_flash.copy(), 
            img_ob_ambnt.copy()
            ])
        img_ob_flash.close()
        img_ob_ambnt.close()
        print('\rReading {:d}/{:d} pairs... '.format(
                len(pairs_ob), nfiles), end='')
    print('Done.')
    return file_pairs, pairs_ob

def read_images(conf):
    train_files, train_data = open_pairs(conf.dataset_path, 'train')
    valid_files, valid_data = open_pairs(conf.dataset_path, 'valid')
    
    PairData = namedtuple('PairData',[
        'train_files',
        'valid_files',
        'train_num',
        'valid_num',
        'train_data',
        'valid_data'
    ])

    return PairData(
        train_files = train_files,
        valid_files = valid_files,
        train_num = len(train_data),
        valid_num = len(valid_data),
        train_data = train_data,
        valid_data = valid_data
    )

def batch_gen(
    pairs_ob = None, 
    crop_size = 224, 
    batch_size = 16,
    pretrained = True
    ):

    random.shuffle(pairs_ob)
    Batch = namedtuple('Batch', ['count', 'files', 'inp','tar'])
	
    for it in range(0, len(pairs_ob), batch_size):
        pairs_ob_batch = pairs_ob[it:it+batch_size]
        flash_batch = []
        ambnt_batch = []
        file_batch = []

        for file_, img_ob_inp, img_ob_tar in pairs_ob_batch:

            flip_var = random.random()
            if flip_var < 0.5:
                img_ob_inp = img_ob_inp.transpose(Image.FLIP_LEFT_RIGHT)
                img_ob_tar = img_ob_tar.transpose(Image.FLIP_LEFT_RIGHT)
            
            #flip_var = random.random()
            #if flip_var < 0.5:
            #    img_ob_inp = img_ob_inp.transpose(Image.FLIP_TOP_BOTTOM)
            #    img_ob_tar = img_ob_tar.transpose(Image.FLIP_TOP_BOTTOM)
            
            #rot_var = random.randint(0,3)
            #for _ in range(rot_var):
            #    img_ob_inp = img_ob_inp.transpose(Image.ROTATE_90)
            #    img_ob_tar = img_ob_tar.transpose(Image.ROTATE_90)

            msize = min(img_ob_inp.size)
            lcrop = int(msize * random.uniform(0.8, 1.0))
            wrand = random.randint(0, int(img_ob_inp.size[0] - lcrop))
            hrand = random.randint(0, int(img_ob_inp.size[1] - lcrop))
			
            img_ob_inp = img_ob_inp.crop((wrand, hrand, wrand+lcrop, hrand + lcrop))
            img_ob_tar = img_ob_tar.crop((wrand, hrand, wrand+lcrop, hrand + lcrop))
            
            img_ob_inp = img_ob_inp.resize([crop_size, crop_size], Image.ANTIALIAS)
            img_ob_tar = img_ob_tar.resize([crop_size, crop_size], Image.ANTIALIAS)
			
            flash_batch.append(to_net(img_ob_inp, pretrained))
            ambnt_batch.append(to_net(img_ob_tar))
            file_batch.append(file_)

            img_ob_inp.close()
            img_ob_tar.close()
	    
        yield Batch(
            count = it + len(pairs_ob_batch), 
            files = file_batch, 
            inp = np.stack(flash_batch, axis=0),
            tar = np.stack(ambnt_batch, axis=0)
            )

def get_eval_pairs(pairs_ob = None, pretrained = True):
    Pair = namedtuple('Pair', ['count', 'file_', 'inp', 'tar'])
    pairs = []

    valid_num = len(pairs_ob)
    for it, (valid_file, img_ob_inp, img_ob_tar) in enumerate(pairs_ob):
        img_arr_inp = to_net(img_ob_inp.copy(), pretrained)
        img_arr_tar = to_net(img_ob_tar.copy())

        pair = Pair(
                count = it + 1, 
                file_ = valid_file, 
                inp = np.expand_dims(img_arr_inp, axis=0),
                tar = np.expand_dims(img_arr_tar, axis=0)
                )
        pairs.append(pair)
        img_ob_inp.close()
        img_ob_tar.close()
        print('\rGetting {:d}/{:d} valid pairs... '.format(
                it+1, valid_num), end='')
    print('Done.')
    return pairs

def open_BF_pairs(root_path, train_files):
    bf_path = os.path.join(root_path, 'train_BF', 'fil_')
    bf_dict = {}
    nfiles  = len(train_files)
    
    for it, (flash_file, ambnt_file) in enumerate(train_files):

        flash_bf_file = bf_path + flash_file.split('/')[-1]
        ambnt_bf_file = bf_path + ambnt_file.split('/')[-1]
        img_ob_inp_bf = Image.open(flash_bf_file)
        img_ob_tar_bf = Image.open(ambnt_bf_file)
        bf_dict[flash_file] = img_ob_inp_bf.copy()
        bf_dict[ambnt_file] = img_ob_tar_bf.copy()
        img_ob_inp_bf.close()
        img_ob_tar_bf.close()

        print('\rReading {:d}/{:d} filtered pairs... '.format(it+1, nfiles), end='')
    print('Done.')
    return bf_dict

def get_eval_inp_bf(root_path, eval_files, eval_data, pretrained=True):
    valid_inp_num = len(eval_files)
    valid_inp_path = os.path.join(root_path, eval_data+'_inp_BF')

    valid_bf_dict = {}

    for it, (flash_file, _) in enumerate(eval_files):
        filename = "_".join(['fil', flash_file.split('/')[-1]])
        file1 = os.path.join(valid_inp_path, filename)

        img_ob_inp_bf = Image.open(file1)
        img_arr_inp_bf = [to_net(img_ob_inp_bf.copy(), pretrained)]
        valid_bf_dict[flash_file[:-10]] = img_arr_inp_bf
        img_ob_inp_bf.close()
        print('\rGetting {:d}/{:d} valid inputs filtered... '.format(
                it+1, valid_inp_num), end='')
    print('Done.')
    return valid_bf_dict

def batch_gen_df(
    conf = None,
    pairs_ob = None,
    bf_dict = None, 
    crop_size  = 224,
    batch_size = 16,
    pretrained = True
    ):
    
    random.shuffle(pairs_ob)
    Batch = namedtuple('Batch', 
        ['count', 'files', 'inp', 'tar', 'inp_bf', 'tar_bf'])

    for it in range(0, len(pairs_ob), batch_size):
        pairs_ob_batch = pairs_ob[it:it+batch_size]
        flash_batch = []
        ambnt_batch = []
        flash_bf_batch = []
        ambnt_bf_batch = []
        file_batch = []
        for file_, img_ob_inp, img_ob_tar in pairs_ob_batch:
            img_ob_inp_bf = bf_dict[file_+'_flash.png']
            img_ob_tar_bf = bf_dict[file_+'_ambient.png']
			
            flip_var = random.random()
            if flip_var < 0.5:
                img_ob_inp = img_ob_inp.transpose(Image.FLIP_LEFT_RIGHT)
                img_ob_tar = img_ob_tar.transpose(Image.FLIP_LEFT_RIGHT)
                img_ob_inp_bf = img_ob_inp_bf.transpose(Image.FLIP_LEFT_RIGHT)
                img_ob_tar_bf = img_ob_tar_bf.transpose(Image.FLIP_LEFT_RIGHT)

            #flip_var = random.random()
            #if flip_var < 0.5:
            #    img_ob_inp = img_ob_inp.transpose(Image.FLIP_TOP_BOTTOM)
            #    img_ob_tar = img_ob_tar.transpose(Image.FLIP_TOP_BOTTOM)
            #    img_ob_inp_bf = img_ob_inp_bf.transpose(Image.FLIP_TOP_BOTTOM)
            #    img_ob_tar_bf = img_ob_tar_bf.transpose(Image.FLIP_TOP_BOTTOM)
            
            #rot_var = random.randint(0,3)
            #for _ in range(rot_var):
            #    img_ob_inp = img_ob_inp.transpose(Image.ROTATE_90)
            #    img_ob_tar = img_ob_tar.transpose(Image.ROTATE_90)
            #    img_ob_inp_bf = img_ob_inp_bf.transpose(Image.ROTATE_90)
            #    img_ob_tar_bf = img_ob_tar_bf.transpose(Image.ROTATE_90)

            msize = min(img_ob_inp_bf.size)
            lcrop = int(msize * random.uniform(0.8, 1.0))
            wrand = random.randint(0, int(img_ob_inp_bf.size[0] - lcrop))
            hrand = random.randint(0, int(img_ob_inp_bf.size[1] - lcrop))
			
            img_ob_inp = img_ob_inp.crop((wrand, hrand, wrand+lcrop, hrand+lcrop))
            img_ob_tar = img_ob_tar.crop((wrand, hrand, wrand+lcrop, hrand+lcrop))
            img_ob_inp = img_ob_inp.resize([crop_size, crop_size], Image.ANTIALIAS)
            img_ob_tar = img_ob_tar.resize([crop_size, crop_size], Image.ANTIALIAS)
			
            img_ob_inp_bf = img_ob_inp_bf.crop((wrand, hrand, wrand+lcrop, hrand+lcrop))
            img_ob_tar_bf = img_ob_tar_bf.crop((wrand, hrand, wrand+lcrop, hrand+lcrop))
            img_ob_inp_bf = img_ob_inp_bf.resize([crop_size, crop_size], Image.ANTIALIAS)
            img_ob_tar_bf = img_ob_tar_bf.resize([crop_size, crop_size], Image.ANTIALIAS)
	
            flash_batch.append(to_net(img_ob_inp))
            ambnt_batch.append(to_net(img_ob_tar))
			
            flash_bf_batch.append(to_net(img_ob_inp_bf, pretrained))
            ambnt_bf_batch.append(to_net(img_ob_tar_bf))			
			
            file_batch.append(file_)
            img_ob_inp.close()
            img_ob_tar.close()
            img_ob_inp_bf.close()
            img_ob_tar_bf.close()
					
        batch = Batch(
        	count = it + len(pairs_ob_batch), 
            files = file_batch, 
            inp = np.asarray(flash_batch, np.float32),
            tar = np.asarray(ambnt_batch, np.float32),
            inp_bf = np.asarray(flash_bf_batch, np.float32),
            tar_bf = np.asarray(ambnt_bf_batch, np.float32)
            )

        yield batch

def bilateral_filter(im, win_size, sigma_space=7, sigma_range=102):
    margin  = int(win_size/2)
    im      = ImageOps.expand(im, margin)
    img_arr = np.asarray(im, dtype=np.int32)
    im.close()

    H,W,_ = img_arr.shape
    mask_img = np.zeros((H,W))

    left_bias  = math.floor(-(win_size-1)/2)
    right_bias = math.floor( (win_size-1)/2)
    filtered_img = img_arr.astype(np.float32)

    gaussian_vals = {I: math.exp(-I**2/(2 * sigma_range**2)) for I in range(256)}
    
    gaussian_vals_func   = lambda x: gaussian_vals[x]
    gaussian_vals_matrix = np.vectorize(gaussian_vals_func, otypes=[np.float32])

    space_weights = np.zeros((win_size,win_size,3))

    for i in range(left_bias, right_bias+1):
        for j in range(left_bias, right_bias+1):
            space_weights[i-left_bias][j-left_bias] = math.exp(-(i**2+j**2)/(2*sigma_space**2))
    
    for i in range(margin, H-margin):
        for j in range(margin, W-margin):
            filter_input  = img_arr[i+left_bias:i+right_bias+1, 
                                    j+left_bias:j+right_bias+1]

            range_weights   = gaussian_vals_matrix(np.abs(filter_input-img_arr[i][j]))
            space_and_range = np.multiply(space_weights, range_weights)

            norm_space_and_range = space_and_range / np.sum(space_and_range, 
                                                            keepdims=False, axis=(0,1))
            output = np.sum(np.multiply(norm_space_and_range, filter_input), axis=(0,1))
                
            filtered_img[i][j] = output
    
    filtered_img = np.clip(filtered_img, 0, 255)
    out = filtered_img[margin:-margin, margin:-margin,:].astype(np.uint8)
    img_bf = Image.fromarray(out)

    return img_bf

def save_filtered_pairs(root_path, pairs_ob, pairs_valid_ob):
    train_num = len(pairs_ob)
    valid_num = len(pairs_valid_ob)
    train_BF_path = os.path.join(root_path, 'train_BF')
    if not os.path.exists(train_BF_path):
        os.makedirs(train_BF_path)

        for it, (file_, img_ob_inp, img_ob_tar) in enumerate(pairs_ob):
            img_ob_inp_bf = bilateral_filter(img_ob_inp.copy(), 7)
            img_ob_tar_bf = bilateral_filter(img_ob_tar.copy(), 7)

            filename1 = "_".join(['fil', file_.split('/')[-1], 'flash.png'])
            filename2 = "_".join(['fil', file_.split('/')[-1], 'ambient.png'])

            file1 = os.path.join(train_BF_path, filename1)
            file2 = os.path.join(train_BF_path, filename2)

            img_ob_inp_bf.save(file1)
            img_ob_tar_bf.save(file2)
            img_ob_inp_bf.close()
            img_ob_tar_bf.close()
            print("\rfiltering 'train' data {:3d}/{:3d} ...".format(it+1, train_num), end='')
        print("Done.")

    valid_inp_BF_path = os.path.join(root_path, 'valid_inp_BF')

    if not os.path.exists(valid_inp_BF_path):
        os.makedirs(valid_inp_BF_path)

        for it, (file_, img_ob_inp, _) in enumerate(pairs_valid_ob):
            img_ob_inp_bf = bilateral_filter(img_ob_inp.copy(), 7)
            filename1 = "_".join(['fil', file_.split('/')[-1], 'flash.png'])
            file1 = os.path.join(valid_inp_BF_path, filename1)
            img_ob_inp_bf.save(file1)
            img_ob_inp_bf.close()
            print("\rfiltering 'valid inputs' data {:3d}/{:3d} ...".format(it+1, valid_num), end='')
        print("Done.")


def save_filtered_pairs_test(root_path, pairs_ob):
    test_num = len(pairs_ob)
    test_BF_path = os.path.join(root_path, 'test_inp_BF')
    if not os.path.exists(test_BF_path):
        os.makedirs(test_BF_path)

        for it, (file_, img_ob_inp, _) in enumerate(pairs_ob):
            img_ob_inp_bf = bilateral_filter(img_ob_inp.copy(), 7)
            filename1 = "_".join(['fil', file_.split('/')[-1], 'flash.png'])
            file1 = os.path.join(test_BF_path, filename1)
            img_ob_inp_bf.save(file1)
            img_ob_inp_bf.close()
            print("\rfiltering 'test inputs' data {:3d}/{:3d} ...".format(it+1, test_num), end='')
        print("Done.")

if __name__ == "__main__":
    img_obj_list = read_train_data(path='FAIDres')
    data_dict = get_arrays_train(input_list = img_obj_list[0:20], 
                                 load_size  = 240, 
                                 out_size   = 224)

    del img_obj_list

    ambnt_imgs = np.array(data_dict['ambnt_imgs'])
    flash_imgs = np.array(data_dict['flash_imgs'])
        
    del data_dict
                
    imgf = flash_imgs[7]*0.5 + 0.5
    imga = ambnt_imgs[7]

    imgf1 = gamma_corr(imgf,0.5)
    imgf2 = gamma_corr(imgf,2.0)
    imgf1 = np.clip(np.transpose(imgf1, (1, 2, 0))*2.0-1,0.0,1.0)
    imgf2 = np.clip(np.transpose(imgf2, (1, 2, 0))*2.0-1,0.0,1.0)
    diff  = imgf2 - np.transpose(imgf, (1, 2, 0))
    img_stack = Image.fromarray(np.hstack(( np.uint8(imgf1*255.0),
                                            np.uint8(imgf2*255.0),
                                            np.uint8(diff *255.0))))
    img_stack.show()
    img_stack.close()
                

