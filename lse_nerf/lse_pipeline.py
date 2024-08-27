import typing
import torch
from dataclasses import dataclass, field
from typing import Dict, Literal, Type, Optional
from time import time
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
    Pipeline
)
from nerfstudio.utils import profiler
from nerfstudio.models.base_model import Model
from nerfstudio.utils import profiler, writer

from lse_nerf.lse_datamanager import MultiCamManagerConfig, MultiCamManager
from lse_nerf.utils import to_gray, correct_img_scale, get_log_dir
from lse_nerf.lse_writer import LSEWriter
from lse_nerf.lsenerf import LSENeRFModel

import git
import os.path as osp

def write_git_hash():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    writer.put_config(name="commit_hash", config_dict={"commit_hash": sha}, step=0)

def write_git_hash_txt():
    save_dir = get_log_dir()
    txt_f = osp.join(save_dir, "commit_hash.txt")

    if osp.exists(txt_f):
        return
    
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    
    with open(txt_f, "w") as f:
        f.write(sha)

@dataclass
class LSENeRFPipelineConfig(VanillaPipelineConfig):

    _target: Type = field(default_factory=lambda: LSENeRFPipeline)
    datamanager:MultiCamManagerConfig = MultiCamManagerConfig()

class LSENeRFPipeline(VanillaPipeline):
    datamanager: MultiCamManager

    
    def __init__(
        self,
        config: VanillaPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        Pipeline.__init__(self)
        self.config = config
        self.test_mode = test_mode
        self.datamanager: MultiCamManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        write_git_hash()

        # NOTE revert to commented version if model fails to load
        # num_embd = len(self.datamanager.train_dataset) if not hasattr(self.datamanager, "num_embd") else self.datamanager.num_embd
        num_embd = self.datamanager.get_num_emb(config.model.embed_config.embedding_type)
        bin_size = self.datamanager.col_dataparser.get_bin_size() 
        self.datamanager.param_to_pipeline_config(config)
        self._model: LSENeRFModel = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=num_embd,
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            bin_size=bin_size
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    def flatten_metrics_dict(self, metrics_dict:dict):
        ### assume max 2 levels only
        flatten_dic = {}
        for k1, dic in metrics_dict.items():
            for k2, v in dic.items():
                flatten_dic[f"{k1}_{k2}"] = v
        
        return flatten_dic


    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        bundles, batch = self.datamanager.next_train(step)
        col_bundle, prev_bundle, next_bundle = bundles
        col_out, prev_out, next_out = None, None, None
        if not (col_bundle is None) and len(col_bundle)> 0:
            col_out = self._model(col_bundle, ev_out = False)
        if not (prev_bundle is None) and len(prev_bundle) > 0:
            prev_out = self._model(prev_bundle, raybd2=next_bundle, batch=batch, ev_out=True)

            if "denerf" in self._model.config.event_loss_type:
                next_out = prev_out # not used if using denerf
            else:
                next_out = self._model(next_bundle, raybd2=prev_bundle, batch=batch, keep=True, ev_out=True)
            
            if hasattr(self._model.evs_mapper, "calc_ev_out") and self._model.config.use_mapping:
                prev_out["ev_out"], next_out["ev_out"] = self._model.evs_mapper.calc_ev_out(prev_out["ev_linear"], next_out["ev_linear"])

        out_dict = {"col_out":col_out, "prev_out":prev_out, "next_out":next_out}
        metrics_dict = self.model.get_metrics_dict(out_dict, batch)

        if (not (metrics_dict == {})) and type(next(iter(metrics_dict.values()))) is dict:
            metrics_dict = self.flatten_metrics_dict(metrics_dict)

        if self.config.datamanager.camera_optimizer is not None:
            for i, cam_key in enumerate(self.datamanager.get_param_groups().keys()):
                # Report the camera optimization metrics
                metrics_dict[f"camera_opt_translation_{i}"] = (
                    self.datamanager.get_param_groups()[cam_key][0].data[:, :3].norm()
                )
                metrics_dict[f"camera_opt_rotation_{i}"] = (
                    self.datamanager.get_param_groups()[cam_key][0].data[:, 3:].norm()
                )
        
        loss_dict = self.model.get_loss_dict(out_dict, batch, metrics_dict)
        return out_dict, loss_dict, metrics_dict

    ## no need for eval_loss_dict because that will be rgb only

    def update_evs_only_metric(self, metrics_dict, gt_img, pred_img):
        pred_img[..., -1] = 0
        gray_gt, gray_pred = to_gray(gt_img), pred_img.sum(dim=-1, keepdims=True)#to_gray(pred_img)
        corr_pred = correct_img_scale(gray_gt, gray_pred)
        corr_pred, gray_gt = torch.concat([corr_pred]*3, dim=-1), torch.concat([gray_gt]*3, dim=-1)
        
        pred, gt = torch.moveaxis(corr_pred, -1, 0)[None], torch.moveaxis(gray_gt, -1, 0)[None]
        psnr, ssim, lpips = self.model.psnr(gt, pred), self.model.ssim(gt, pred), self.model.lpips(gt, pred)

        if metrics_dict is not None:
            metrics_dict["psnr"] = float(psnr.item())
            metrics_dict["ssim"] = float(ssim)
            metrics_dict["lpips"] = float(lpips)
        else:
            metrics_dict = {"psnr": float(psnr.item()), "ssim":float(ssim), "lpips": float(lpips)}
        return metrics_dict, gray_gt, corr_pred
        


    def get_eval_image_metrics_and_images(self, step: int):
        metrics_dict, images_dict = super().get_eval_image_metrics_and_images(step)

        if self.datamanager.config.rgb_frac == 0:
            img = images_dict["img"]
            cut_idx = img.shape[1]//2
            gt_img, pred_img = img[:, :cut_idx], img[:, cut_idx:]

            ## assume to be ednerf model only
            metrics_dict, gt_img, pred_img = self.update_evs_only_metric(metrics_dict, gt_img, pred_img)

            img = torch.concat([gt_img, pred_img.squeeze()], axis=1)
            
            images_dict["img"] = img
        
        return metrics_dict, images_dict
    
    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        """Iterate over all the images in the eval dataset and get the average.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []

        write_git_hash_txt()
        edwriter = LSEWriter(get_log_dir())

        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera_ray_bundle, batch in self.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                inner_start = time()
                height, width = camera_ray_bundle.shape
                num_rays = height * width
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                metrics_dict, image_dict = self.model.get_image_metrics_and_images(outputs, batch)
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
                metrics_dict_list.append(metrics_dict)
                edwriter.log_images(image_dict)
                progress.advance(task)

            edwriter.plt_mapper(self._model, outputs)
        
        edwriter.log_metrics(metrics_dict_list)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            metrics_dict[key] = float(
                torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
            )
        self.train()
        return metrics_dict
    

    def load_pipeline(self, loaded_state, step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)
        self.load_state_dict(state, strict=False)

