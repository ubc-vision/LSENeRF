import json
import os
import os.path as osp
import torch
import jax
import numpy as np
from PIL import Image

from lse_nerf.utils import plt_mapper
import glob
from nerfstudio.utils.rich_utils import CONSOLE

def get_num_item_in_dir(path):
    return len(os.listdir(path))

class LSEWriter:
    """
    write data to disk
    """

    def __init__(self, log_dir) -> None:
        self.log_dir = osp.join(log_dir, "eval_results")
        os.makedirs(self.log_dir, exist_ok=True)

        self.comb_res_dir = osp.join(self.log_dir, "comb_imgs")
        os.makedirs(self.comb_res_dir, exist_ok=True)

        self.skip_keys = ["accumulation", "ev_linear"]
        self.cnt=0
    
    
    def _write_image(self, path: str, image:torch.Tensor):
        img_data = torch.clamp(image*255,0,255).cpu().type(torch.uint8)
        img_data = img_data if img_data.shape[-1] == 3 else torch.tile(img_data, (1,1,3))
        img = Image.fromarray(img_data.numpy())
        img.save(path)

        return img_data
    


    def _make_grid(self, images:list):
        return torch.cat(images, dim=1)

    def log_images(self, output_img_dict:dict):
        all_imgs = []
        for k, v in output_img_dict.items():
            save_dir = osp.join(self.log_dir, f"{k}")
            save_path = osp.join(save_dir, str(self.cnt).zfill(3) + ".png")
            if k in self.skip_keys:
                continue

            if not osp.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            if k == "linear":
                v = v/v.max()
            
            img_data = self._write_image(save_path, v)
            all_imgs.append(img_data)
        
        all_imgs = self._make_grid(all_imgs) 
        self._write_image(osp.join(self.comb_res_dir, str(self.cnt).zfill(3) + ".png"), all_imgs)
        self.cnt += 1
    

    def log_metrics(self, metrics_dict_list):
        metrics_dict = jax.tree_map(lambda  *x: x, * metrics_dict_list)
        
        metric_f = osp.join(osp.dirname(self.log_dir), "eval_metrics.json")
        with open(metric_f, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        
        metric_mean_dict = {k: np.array(v).mean() for k, v in metrics_dict.items()}
        metric_mean_f = osp.join(osp.dirname(self.log_dir), "eval_mean.json")
        with open(metric_mean_f, "w") as f:
            json.dump(metric_mean_dict, f, indent=4)


    def plt_mapper(self, model, outputs = None):
        linear, ev_linear = None, None
        if hasattr(model, "rgb_mapper"):
        
            try:
                save_dir = osp.join(osp.dirname(self.log_dir), "rgb_mappers")
                os.makedirs(save_dir, exist_ok=True)

                num_saved = glob.glob(osp.join(save_dir, "learned_gamma*.png"))
                save_path = osp.join(save_dir, f"learned_gamma_{str(len(num_saved)).zfill(3)}.png")
                
                if outputs is not None:
                    linear = outputs.get("linear")
                plt_mapper(model.rgb_mapper, 1, 1, save_path, linear_img=linear)
            except:
                CONSOLE.print("[yellow] rgb mapper is not mappable!")
            
        

        if hasattr(model, "evs_mapper"):
            try:
            
                save_dir = osp.join(osp.dirname(self.log_dir), "evs_mappers")
                os.makedirs(save_dir, exist_ok=True)

                num_saved = glob.glob(osp.join(save_dir, "learned_gamma*.png"))
                save_path = osp.join(save_dir, f"learned_gamma_{str(len(num_saved)).zfill(3)}.png")

                if outputs is not None:
                    ev_linear = outputs.get("ev_linear")
                plt_mapper(model.evs_mapper, 1, 1, save_path, gt_fn=None, linear_img=ev_linear)
            except:
                CONSOLE.print("[yellow] evs mapper is not mappable!")