from nerfstudio.cameras.rays import RayBundle
import torch
from torch import nn
import jax
import numpy as np
from nerfstudio.utils import writer
import os
from nerfstudio.utils.rich_utils import CONSOLE
import cv2
import matplotlib.pyplot as plt

EPS = 1e-6

# config, specify whether the eval run is pretrain
class GbConfig:
    DO_PRETRAIN = False
    IS_EVAL = False
    IS_RENDER = False

gbconfig = GbConfig()

def set_network_grad(model:nn.Module, val:bool):
    
    for param in model.parameters():
        param.requires_grad = val
    
    return model

def freeze_network(model):
    return set_network_grad(model, False)

def unfreeze_network(model):
    return set_network_grad(model, True)


def get_max_val(linear_image = None):
    default_max = 1
    if linear_image is None:
        return default_max
    
    lin_max = linear_image.max()
    if lin_max > default_max:
        return lin_max
    else:
        if lin_max/default_max < 0.7:
            return lin_max
        else:
            return default_max 

def plt_mapper(mapper, in_dim, out_dim, save_path = None, gt_fn = lambda x : x **(1/2.4), device="cuda", linear_img:torch.Tensor = None, max_val:float=None):
    with torch.no_grad():
        max_val = get_max_val(linear_img) if max_val is None else max_val
        lins = torch.linspace(0, max_val, 1000).reshape(-1,1).to(device)
        out = mapper(lins)

        prep = lambda x : x.detach().cpu().numpy()
        plt.plot(prep(lins), prep(out), label = "pred")

        if gt_fn is not None:
            plt.plot(prep(lins), gt_fn(prep(lins)), label = "gt")
        
        if linear_img is not None:
            linear = prep(linear_img).reshape(-1)            
            counts, bin_edges = np.histogram(linear, bins=50)

            # Calculate normalized weights
            weight_factor = prep(out).max() / counts.max()
            weights = np.ones_like(linear) * weight_factor

            plt.hist(linear, bins=bin_edges, weights=weights)

        plt.legend()

        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
        
        plt.clf()


def detect_edges(gray_image, min_threshold=50, max_threshold=200):

    edges = cv2.Canny(gray_image, min_threshold, max_threshold)

    return edges

def get_log_dir():

    for event_writer in writer.EVENT_WRITERS:
        if type(event_writer) is writer.TensorboardWriter:
            return event_writer.tb_writer.log_dir
        

        CONSOLE.print("[blue] writer is not tensorboardwriter, don't know logging dir, using ./tmp")
        os.makedirs("tmp", exist_ok=True)
        return "tmp"
        
def to_gray(img):
    if ((img.shape[-1] != 3) or (len(img.shape) == 2)) and (img.shape[-1] != 1):
        return img

    c2g_vec = torch.tensor([0.2989 , 0.5870 , 0.1140 ]).reshape(-1,1).to(img)
    grey_img = img@c2g_vec
    return grey_img


# taken from e-nerf
def solve_normal_equations(preds_logs, imgs_gt_log):
    N = np.prod(imgs_gt_log.shape)
    X = torch.ones((N, 2)).to(preds_logs.device)
    X[:, 1] = preds_logs.flatten()
    y = imgs_gt_log.flatten()

    # solve normal equaiton
    beta = torch.linalg.inv(X.T @ X) @ X.T @ y
    a = beta[1]
    b = beta[0]

    if torch.isnan(b):
        b = 5/255
    elif torch.isnan(-b):
        b = -5/255

    if torch.isnan(a):
        a = 5/255
    elif torch.isnan(-a):
        a = -5/255

    return a, b

def correct_img_scale(gt, pred):
    gt_log, pred_log = torch.log(gt + EPS), torch.log(pred + EPS)
    a, b = solve_normal_equations(pred_log, gt_log)
    return torch.exp(a * pred_log + b)
    

def linear_correction(gt, pred):
    ori_shape = pred.shape

    flat = lambda x : x.reshape(-1, 3)
    gt, pred = flat(gt), flat(pred)

    X = torch.cat([pred, torch.ones(len(pred), 1, device=pred.device)], dim = 1)
    params = torch.linalg.lstsq(X, gt).solution

    corr_pred = torch.clip((X@params).reshape(ori_shape), 0, 1)
    gt, pred = gt.reshape(ori_shape), pred.reshape(ori_shape)
    
    return corr_pred


def fix_datashape(data, ref_data):
    """
    for deblur, the n_batch < n_ray generated; so expand it before generating the ray bundles
    """
    if data is None:
        return data
    
    if len(data) == len(ref_data):
        return data
    
    assert len(ref_data)%len(data) == 0, "Should be a multiple!"
    factor = int(len(ref_data)/len(data))
    data = torch.tile(data, (1, factor, *[1]*len(data.shape[1:])))
    data = torch.flatten(data, 0, 1)  # flatten first 2 dim

    return data

def fix_appearance_id(data, ref_data, max_app_id = 1000000):
    factor = int(len(ref_data)/len(data))
    if factor == 1:
        return fix_datashape(data, ref_data)

    delta = (torch.arange(factor, device=ref_data.device) - factor//2)[None,:]
    data = data + delta
    data = torch.clip(data, 0, max_app_id - 1)
    return data.reshape(-1, 1)
    


def add_metadata(ray_bundle:RayBundle, batch:dict = None, cam_id = 0, max_app_id = 10000000):
    """
    ngp uses cam_idx for appearance embedding. That conflicts with correct location
    if events are used.
    This function replace the ray_bundles camera idx with the appearence_idx
    // only required to replace color batch since events cam_idxs are correct
    """
    device = ray_bundle.camera_indices.device
    ref_data = ray_bundle.camera_indices
    ray_bundle.metadata["appearance_id"] = fix_appearance_id(batch["appearance_id"][..., None].contiguous(), ref_data, max_app_id)
    ray_bundle.metadata["cam_type"] = fix_datashape(torch.full((len(ray_bundle),), cam_id, device=device), ref_data) # rgb = 0, evs = 1
    ray_bundle.metadata["coords"] = fix_datashape(batch["indices"], ref_data)  # (frame_id, ys, xs)

    return ray_bundle