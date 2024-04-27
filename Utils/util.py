"""This module contains helper functions """
from __future__ import print_function

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import multiprocessing
import json
import joblib
from joblib import Parallel, delayed
import torch
import numpy as np
from PIL import Image
import cv2
import scipy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
import contextlib
import tmqi

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()    

def get_image_name_from_path(img_path):
    return '.'.join(img_path.split('/')[-1].split('.')[:-1])

def match_and_filter_paths(ldr_paths, hdr_paths):
    new_paths = []

    ldr_images = [get_image_name_from_path(item) for item in ldr_paths]
    for img_path in hdr_paths:
        img_name = get_image_name_from_path(img_path)
        if img_name in ldr_images:
            new_paths.append(img_path)
    
    return new_paths

def calculate_avg_TMQI(fake_dir, hdr_dir, save_dir, downsample=1, save_suffix='default', max_images=0):
    """Calculate average TMQI scores for the generated LDR images in the directory.
        Image filenames in the HDR directory should match the generated images.

    Parameters:
        fake_dir (folder) --  directory of the generated LDR images
        hdr_dir (folder) --  directory of the HDR images
    """
    
    ldr_image_paths = get_file_paths(fake_dir, extensions=['png'])
    hdr_image_paths = get_file_paths(hdr_dir, extensions=['exr', 'png'])

    hdr_image_paths = match_and_filter_paths(ldr_image_paths, hdr_image_paths)

    L = len(ldr_image_paths)
    H = len(hdr_image_paths)
    assert L == H, 'number of images in the LDR folder ({}) should be less than or equal to HDR folder ({})'.format(
                    L, H
                )

    if max_images > 0:
        ldr_image_paths = ldr_image_paths[:max_images]
        hdr_image_paths = hdr_image_paths[:max_images]

    num_cores = 4

    '''
    idx = 0
    scores = {}
    naturalness = []
    structure = []
    print('calculating TMQI scores for generated images...')
    for ldr_path in ldr_image_paths:

        if max_images and max_images == idx:
            break

        hdr_path = hdr_image_paths[idx]
        score, s, n = calculate_TMQI(ldr_path, hdr_path, downsample)
        naturalness.append(n)
        structure.append(s)

        filename = ldr_path.split('/')[-1]
        scores[filename] = score
        
        idx += 1
        print('{}: {}'.format(idx, score), end='\r')
    '''

    total_images = len(ldr_image_paths)

    with tqdm_joblib(tqdm(desc="TMQI Calculation", total=total_images)) as progress_bar:
        scores = Parallel(n_jobs=num_cores)(delayed(calculate_TMQI)(ldr_image_paths[i], hdr_image_paths[i], downsample) 
                                                            for i in range(total_images)
                                        )
    scores, naturalness, structure = [list(t) for t in zip(*scores)]

    filenames = [get_image_name_from_path(p) for p in ldr_image_paths]
    scores = dict(zip(filenames, scores))
    
    score_np = np.array(list(scores.values())) * 100.0
    nat_np = np.array(naturalness) * 100.0
    struct_np = np.array(structure) * 100.0
    nans = (score_np == score_np) & (nat_np == nat_np) & (struct_np == struct_np)
    
    score_np = score_np[nans]
    nat_np = nat_np[nans]
    struct_np = struct_np[nans]

    scores['AA1_TMQI_Q_mean'] = score_np.mean()
    scores['AA1_TMQI_Q_std'] = score_np.std()

    scores['AA2_TMQI_N_mean'] = nat_np.mean()
    scores['AA2_TMQI_N_std'] = nat_np.std()
    
    scores['AA3_TMQI_S_mean'] = struct_np.mean()
    
    scores['AA3_TMQI_S_std'] = struct_np.std()

    print()
    print("average TMQI-Q:", scores['AA1_TMQI_Q_mean'], '+-', scores['AA1_TMQI_Q_std'])

    with open(os.path.join(save_dir, 'TMQI_scores_{}.json'.format(save_suffix)), 'w') as f:
        json.dump(scores, f, indent=5, sort_keys=True)

def calculate_TMQI(ldr_path, hdr_path, downsample=1, return_dict=False):
    """Calculate TMQI score of the given LDR image.
        File names should match. This is a check to make sure they are correctly paired. 

    Parameters:
        ldr_path (path) --  LDR image path
        hdr_path (path) --  HDR image path
    """
    if isinstance(ldr_path, str) and isinstance(hdr_path, str):
        ldr_name = get_image_name_from_path(ldr_path).replace('8bit', '16bit')
        hdr_name = get_image_name_from_path(hdr_path)
        
        assert ldr_name == hdr_name, 'filenames does not match, check if the hdr-ldr pair is correctly matched'

    try:
        ldr_image = tmqi.img_read(ldr_path) if isinstance(ldr_path, str) else ldr_path
        hdr_image = tmqi.img_read(hdr_path) if isinstance(hdr_path, str) else hdr_path

        h,w = ldr_image.shape[:2]
        _h,_w = hdr_image.shape[:2]

        if _h != h or _w != w:
            hdr_image = cv2.resize(hdr_image, (w,h))

        if downsample > 1:
            w, h = int(w/downsample), int(h/downsample)
            hdr_image = cv2.resize(hdr_image, (w,h))
            ldr_image = cv2.resize(ldr_image, (w,h))

        Q, S, N, s_local, s_maps = tmqi.TMQI()(hdr_image, ldr_image)
        if Q != Q or S != S or N != N:
            print('Warning: Encountered NaN in TMQI calculation: '.format(ldr_name))
    except:
        raise RuntimeError("could not process the image: {}".format(ldr_name)) 
    
    if return_dict:
        return {"TMQI_Q":Q, "TMQI_S":S, "TMQI_N":N}
    else:
        return Q, S, N

def tensor2im(input_image, label='real_B', min_pix=0.0, max_pix=65536.0, cityscapes=False):
    """Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))

        is_ldr = 'B' in label

        if is_ldr:
            min_pix = 0.0
            max_pix = 255.0
            im_type = np.uint8
        else:
            if cityscapes:
                min_pix = 0.0
                max_pix = 65536.0
                im_type = np.uint16
            else:
                im_type = np.float32

        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0
        image_numpy = image_numpy * (max_pix - min_pix) + min_pix
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image

    return image_numpy.astype(im_type)

def normalize_tensor_for_tboard(input_image):
    """Brings the output image, from tanh function to the range of [0,1] 
    """
    
    if isinstance(input_image, list):
        assert isinstance(input_image[0], torch.Tensor), 'image should be a torch tensor'
        assert input_image[0].dtype in [torch.float, torch.double, torch.half], 'image tensor sould be of floating point type'
    
        input_image = torch.cat(input_image)
    
    else:
        assert isinstance(input_image, torch.Tensor), 'image should be a torch tensor'
        assert input_image.dtype in [torch.float, torch.double, torch.half], 'image tensor sould be of floating point type'
    
    unsqueezed_image = input_image.cpu().float()
    unsqueezed_image += 1
    unsqueezed_image /= 2
    
    return unsqueezed_image[:, [2,1,0], :]


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)

def plot_grad_flow(
        named_parameters, 
        grad_dict,  
        plot_title, 
        save_path=None, 
        plot=False
    ):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    
    idx = 1
    for n, p in named_parameters:
        if hasattr(p, 'grad') and (p.grad is not None) and ("bias" not in n) and ('conv' in n):
            layer_key = "L{}".format(idx)
            idx += 1
            
            if layer_key not in grad_dict:
                grad_dict[layer_key] = []
            grad_dict[layer_key].append(p.grad.cpu().norm(2))

    
    if plot:
        plt.clf()
        plt.figure(figsize=(15, 5))

        layers = grad_dict.keys()
        avg_grads = [np.array(_layer).mean() for _,_layer in grad_dict.items() ]
        std_grads = [np.array(_layer).std() for _,_layer in grad_dict.items() ]
        
        bar1 = plt.bar(
            np.arange(len(avg_grads)), 
            avg_grads, 
            yerr=std_grads, 
            lw=1, 
            color="b", 
            ecolor="r",
            capsize=5.0)

        plt.xticks(range(0,len(avg_grads), 1), layers, rotation="horizontal")
        plt.xlim(left=-1, right=len(avg_grads))
        
        #top_y = 0.4
        #plt.ylim(bottom = -0.001, top=top_y) # zoom in on the lower gradient regions

        # Add counts above the bar graphs
        for i,rect in enumerate(bar1):
            height = rect.get_height()
            #text_height = min(top_y * 0.95, (height + std_grads[i]) * 1.05)
            text_height = (height + std_grads[i]) * 1.05
            plt.text(rect.get_x() + rect.get_width() / 2.0, text_height, f'{height:.2f}', ha='center', va='bottom')

        plt.xlabel("layers")
        plt.ylabel("avg. gradient norm")
        plt.title(plot_title)

        if save_path is not None:
            plt.savefig(save_path)

def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    '''
    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape
    
    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)
    '''

    cv2.imwrite(image_path, image_numpy)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def get_file_paths(folder, extensions=['png', 'jpg']):
    """get all files paths from the folder recursively

    Parameters:
        folder (str) -- a single directory path
        extensions (list) -- extensions of the files to include in the returned list
    """
    image_filenames = []
    file_roots = []
    full_paths = []
    for root, dirs, filenames in os.walk(folder):
        for filename in filenames:
             for ext in extensions:
                if filename.endswith('.' + ext):
                    input_path = os.path.abspath(root)
                    file_roots.append(input_path)
                    image_filenames.append(filename)
                    break

    sorted_indices = np.argsort(image_filenames)
    full_paths = [os.path.join(file_roots[i], image_filenames[i]) for i in sorted_indices]
    return full_paths

# calculate_avg_TMQI('/home/ee/HDRDataset/test/images/test_ldr','/home/ee/HDRDataset/test/hdr_images','/home/ee/HDRDataset/test')
calculate_avg_TMQI('/home/ee/HDRDataset/test/model','/home/ee/HDRDataset/test/hdr_images','/home/ee/HDRDataset/test')