from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.models.instant_ngp import NGPModel, InstantNGPModelConfig
from dataclasses import dataclass, field
from typing import Type

import torch
from torch import Tensor
import nerfacc
import cv2

from nerfstudio.utils import colormaps
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.ray_samplers import VolumetricSampler
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.models.base_model import Model
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)


from lse_nerf.utils import to_gray, detect_edges
from lse_nerf.lse_field import LSEField
from lse_nerf.lse_renderer import LinearRenderer
from lse_nerf.intensity_mappers import MAPPERS_DICT
from lse_nerf.lse_grid_estimator import LSEOccGridEstimator
from lse_nerf.lse_embeddings import LSEEmbeddingConfig
from lse_nerf.utils import EPS, gbconfig

import numpy as np
import math

@dataclass
class LSENeRFModelConfig(InstantNGPModelConfig):
    _target: Type = field(default_factory=lambda: LSENeRFModel)
    evs_loss_weight: float = 1.0
    """event loss weight"""

    emb_norm_weight: float = 1e-2
    """weight for emb reg"""

    event_loss_type:str = "log_loss"
    """loss used for event supervision"""

    use_mapping:bool = False
    """use intensity mapper or not; used for mapping from linear to gamma"""

    mapping_method:str = "mlp"
    """mapping method to use for rgb"""

    evs_mapping_method:str = "None"
    """mapping method to use for events"""

    ev_one_dim:str = "learned"
    """map 3d rgb to 1d for event loss, one of [learned, gt]"""

    rgb_loss_type:str = "linspace"

    ###### mapper loss ###########
    use_mapper_loss:bool = False

    mapper_loss_weight:float = 0.25
    ###############################
    scaler_weight:float = 1.0

    map_mode: str = "ev_rgb"
    """one of [evs_rgb, rgb_evs, co_map]; see get_outputs function for details"""


    embed_config: LSEEmbeddingConfig = LSEEmbeddingConfig()

    def __post_init__(self):
        if self.evs_mapping_method is None or self.evs_mapping_method.lower() == "none":
            self.evs_mapping_method = None
        
        if self.map_mode.lower() == "none":
            self.map_mode = "evs_rgb"
        
        if self.ev_one_dim.lower() == "false" or self.ev_one_dim.lower() == "none":
            self.ev_one_dim = False
        elif self.ev_one_dim.lower() == "true":
            self.ev_one_dim = "learned"
        
        if self.rgb_loss_type.lower() == "none":
            self.rgb_loss_type = "linspace"


class ThreeToOne(nn.Module):
    def __init__(self) -> None:
        super(ThreeToOne,self).__init__()
        self.weights = nn.Parameter(torch.ones(1,3)/3)
        
    def forward(self, x):
        out1 = F.linear(x, F.softmax(self.weights, dim=-1), None)
        return out1


class ToGrayGT(nn.Module):
    def __init__(self) -> None:
        super(ToGrayGT, self).__init__()
        self.register_buffer("c2g_vec", torch.tensor([0.2989 , 0.5870 , 0.1140 ]).reshape(-1,1))
    
    def forward(self, img):
        grey_img = img@self.c2g_vec
        return grey_img



def lin_log(x, threshold=20):
    """
    linear mapping + logarithmic mapping.
    :param x: float or ndarray the input linear value in range 0-255
    :param threshold: float threshold 0-255 the threshold for transisition from linear to log mapping
    """
    # converting x into np.float32.
    if x.dtype is not torch.float64:
        x = x.double()
    f = (1./threshold) * math.log(threshold)
    y = torch.where(x <= threshold, x*f, torch.log(x))

    return y.float()


dis_lin = lambda x : x - 1/(x+1e-5)**0.8
format_linear = lambda x : torch.concatenate([x]*3, dim=-1) if x.shape[-1] == 1 else x

class LSENeRFModel(NGPModel):
    config: LSENeRFModelConfig
    def __init__(self, config: LSENeRFModelConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.log_losses_dict = {"log_loss": self.log_loss,
                                "enerf_norm_loss": self.enerf_norm_loss}

        self.rgb_losses_dic = {"linspace": self.mse_loss,
                               "deblur": self.mse_loss}
                             
        
        self.rgb_loss_fn = self.rgb_losses_dic[self.config.rgb_loss_type.lower()]
        self.event_loss = self.log_losses_dict[self.config.event_loss_type.lower()]


        self.kwargs = kwargs
    
    def populate_modules(self):

        """Set the fields and modules."""
        Model.populate_modules(self)

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # NOTE: gives n_evs/n_rgb ratio; useful for initializing seperate embeddings
        if self.kwargs.get("bin_size") is not None:
            self.config.embed_config.set_binsize(self.kwargs["bin_size"])

        self.field = LSEField(
            aabb=self.scene_box.aabb,
            num_images=self.num_train_data,
            log2_hashmap_size=self.config.log2_hashmap_size,
            max_res=self.config.max_res,
            spatial_distortion=scene_contraction,
            embd_config=self.config.embed_config,
            implementation="tcnn"
        )

        self.scene_aabb = torch.nn.Parameter(self.scene_box.aabb.flatten(), requires_grad=False)

        if self.config.render_step_size is None:
            # auto step size: ~1000 samples in the base level grid
            self.config.render_step_size = ((self.scene_aabb[3:] - self.scene_aabb[:3]) ** 2).sum().sqrt().item() / 1000
        # Occupancy Grid.
        self.occupancy_grid = LSEOccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )

        # Sampler
        self.sampler = VolumetricSampler(
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
        )

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
   
        if self.config.use_mapping:
            mapping_initializer = MAPPERS_DICT.get(self.config.mapping_method.lower())
            assert mapping_initializer is not None, f"{self.config.mapping_method} mapper is not supported"
            self.rgb_mapper = mapping_initializer()
            self.renderer_rgb = LinearRenderer(background_color=self.config.background_color)
        
        self.evs_mapper = None
        if self.config.evs_mapping_method is not None and self.config.map_mode == "co_map":
            mapping_initializer = MAPPERS_DICT.get(self.config.evs_mapping_method.lower())
            assert mapping_initializer is not None, f"{self.config.evs_mapping_method} mapper is not supported"
            
            if self.config.evs_mapping_method == "powbook":
                self.evs_mapper = mapping_initializer(self.num_train_data)
            else:
                self.evs_mapper = mapping_initializer()
        
        if self.config.ev_one_dim == "learned":
            self.rgb_to_one = ThreeToOne()
        elif self.config.ev_one_dim == "gt":
            self.rgb_to_one = ToGrayGT()


    def get_param_groups(self):
        param_groups = super().get_param_groups()
        if self.config.mapping_method == "gt":
            return param_groups
        
        if self.config.use_mapping:
            param_groups["fields"] = param_groups["fields"] + list(self.rgb_mapper.parameters())
        
        if self.config.ev_one_dim:
            param_groups["fields"] = param_groups["fields"] + list(self.rgb_to_one.parameters())
        
        if self.config.evs_mapping_method is not None and hasattr(self, "evs_mapper"):
            param_groups["fields"] = param_groups["fields"] + list(self.evs_mapper.parameters())
        
        
        if gbconfig.IS_EVAL:
            param_groups["fields"] = list(self.field.embedding_appearance.parameters())

        return param_groups
    
    def init_test_params(self):
        self.field.embedding_appearance.init_test_params()
    

    def correct_evs_dim(self, inp):
        """
        inp (torch.tensor) in R^(nx3), 
            map from R^3 --> R^1
        """
        if self.config.ev_one_dim:
            return self.rgb_to_one(inp)
        else:
            return inp
    
    def forward(self, ray_bundle: RayBundle, **kwargs):
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(ray_bundle, **kwargs)

    def exec_get_outputs(self, ray_bundle: RayBundle):
        assert self.field is not None
        num_rays = len(ray_bundle)

        # with torch.no_grad():
        ray_samples, ray_indices = self.sampler(
            ray_bundle=ray_bundle,
            near_plane=self.config.near_plane,
            far_plane=self.config.far_plane,
            render_step_size=self.config.render_step_size,
            alpha_thre=self.config.alpha_thre,
            cone_angle=self.config.cone_angle,
        )

        metadata = {}
        for k, v in ray_bundle.metadata.items():
            metadata[k] = v[ray_indices]

        ray_samples.metadata = metadata
        field_outputs = self.field(ray_samples)

        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights = nerfacc.render_weight_from_density(
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
            packed_info=packed_info,
        )[0]
        weights = weights[..., None]

        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        depth = self.renderer_depth(
            weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays
        )
        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "num_samples_per_ray": packed_info[:, 1],
        }
        return outputs    


    def get_outputs(self, ray_bundle: RayBundle, ev_out=False, **kwargs):
        """
        key, "rgb", is used for evaluation, need to replace it with correct color space
        """
        out_dict = self.exec_get_outputs(ray_bundle)
        clamp_out = torch.clamp(out_dict["rgb"], 1e-5)
        if self.config.use_mapping or self.config.map_mode == "rgb_evs":
            if self.config.map_mode == "rgb_evs":
                """
                vol_ren -> rgb -> evs
                """
                if ev_out or not self.training:
                    out_dict["ev_out"] = self.rgb_mapper(self.correct_evs_dim(clamp_out))
                    out_dict["linear"] = format_linear(out_dict["ev_out"])
            elif self.config.map_mode == "evs_rgb":
                """
                vol_ren -> evs -> rgb
                """
                out_dict["ev_out"] = self.correct_evs_dim(clamp_out)
                out_dict["linear"] = clamp_out
                out_dict["rgb"] = self.rgb_mapper(out_dict["linear"]).to(out_dict["linear"])
            elif self.config.map_mode == "co_map":
                """
                            /---> rgb_mapper ---> rgb
                NGP -> linear
                            \---> evs_mapper ---> evs
                """
                out_dict["rgb"] = self.rgb_mapper(clamp_out)

                # if self.training:
                if ev_out or not self.training:
                    ev_linear = self.correct_evs_dim(clamp_out)
                    out_dict["linear"] = clamp_out
                    out_dict["ev_linear"] = ev_linear
                    out_dict["ev_out"] = self.evs_mapper(ev_linear, raybd1=ray_bundle, **kwargs)
        
        if self.config.rgb_loss_type == "deblur" and self.training:
            try:
                out_dict["rgb"] = out_dict["rgb"].reshape(-1, 4, 3).mean(axis=1) # (-1, n_blur_rays, 3)
            except:
                # NOTE: for event forward, [rgb] value not used, so it can be as wrong as it wants
                pass

        if not self.training:
            out_dict["rgb"] = torch.clamp(out_dict["rgb"], 0, 1)    
        else:
            out_dict["rgb"] = torch.clamp(out_dict["rgb"], 1e-5)

        return out_dict
        

    def get_metrics_dict(self, outputs, batch):
        if (outputs.get("col_out") is None) and (outputs.get("prev_out") is None):
            return super().get_metrics_dict(outputs, batch)
        
        metric_dict = {}
        if outputs["col_out"] is not None:
            metric_dict["col"] = super().get_metrics_dict(outputs["col_out"], batch["col_batch"])

        
        return metric_dict


    def log_loss(self, evs, prev_rad, next_rad, evs_batch:dict):
        if prev_rad.shape[-1] != 1:
            prev_rad, next_rad = to_gray(prev_rad), to_gray(next_rad)

        prev_log, next_log = torch.log(prev_rad + EPS), torch.log(next_rad + EPS)
        delta_log = next_log - prev_log

        return self.rgb_loss(delta_log, evs)


    def mse_loss(self, rgb_gt, rgb_pred, rgb_out_dic = None):
        return self.rgb_loss(rgb_gt, rgb_pred)
    

    def enerf_norm_loss(self, evs, prev_rad, next_rad, evs_batch:dict):
        # assert prev_rad.shape[-1] == 1, "support grayscale only for now"
        if prev_rad.shape[-1] != 1:
            prev_rad, next_rad = to_gray(prev_rad), to_gray(next_rad)

        prev_log, next_log = torch.log(prev_rad + EPS), torch.log(next_rad + EPS)
        delta_log = next_log - prev_log

        log_norm_cnst = torch.linalg.norm(delta_log, dim=0, keepdim=True) + EPS
        with torch.no_grad():
            evs = evs/evs_batch["e_thresh"]
            evs_norm_cnst = torch.linalg.norm(evs, dim=0, keepdim=True) + EPS

        return self.rgb_loss(delta_log/log_norm_cnst, evs/evs_norm_cnst)
    

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        if ((batch.get("col_batch") is None) and (batch.get("evs_batch") is None)):
            return super().get_loss_dict(outputs, batch)
        loss_dict = {}
        col_batch, evs_batch = batch["col_batch"], batch["evs_batch"]
        col_out, prev_out, next_out = [outputs[e] for e in ["col_out", "prev_out", "next_out"]]

        if col_out is not None:
            loss_dict["rgb_loss"] = self.rgb_losses_dic[self.config.rgb_loss_type.lower()](col_batch["image"].to(self.device), col_out["rgb"], col_out)
        
        if prev_out is not None:
            ev_key = "rgb" if not self.config.use_mapping else "ev_out"
            prev_in, next_in = prev_out[ev_key], next_out[ev_key] 
            evs = evs_batch["image"].to(self.device)
            evs = evs if prev_in.shape[-1] == 1 else torch.concatenate([evs]*3, dim=-1)
            loss_dict["event_loss"] = self.config.evs_loss_weight*self.event_loss(evs , prev_in, next_in, evs_batch)

        return loss_dict


    def _make_error_map(self, rgb, pred):
        norm_cnst = 6
        rgb_gray, pred_gray = to_gray(rgb).squeeze(), to_gray(pred).squeeze()

        err = (rgb_gray - pred_gray)*norm_cnst
        err_img = torch.ones((*rgb.shape[:2], 3)).to(rgb_gray.device)

        err_thresh = 0 
        pos_cond = err > err_thresh
        neg_cond = err < -err_thresh

        err_img[..., 1][pos_cond] = 1 - err[pos_cond]
        err_img[..., 2][pos_cond] = 1 - err[pos_cond]

        err_img[..., 0][neg_cond] = 1 - torch.abs(err[neg_cond])
        err_img[..., 1][neg_cond] = 1 - torch.abs(err[neg_cond])


        return err_img

    def _make_overlay(self, rgb, pred):
        rgb_gray, pred_gray = to_gray(rgb).squeeze().cpu().numpy(), to_gray(pred).squeeze().cpu().numpy()
        rgb_gray, pred_gray = np.clip(rgb_gray*255, 0, 255).astype(np.uint8), np.clip(pred_gray*255, 0, 255).astype(np.uint8)
        gt_e, pred_e = detect_edges(rgb_gray), detect_edges(pred_gray)
        gt_cond = (gt_e != 0)
        pred_cond = (pred_e != 0)
        cond = gt_cond | pred_cond

        overlay = np.ones((*rgb_gray.shape[:2], 3), dtype=np.uint8)*255
        overlay[cond] = 0

        overlay[gt_cond,0] = 255
        overlay[pred_cond,2] = 255
        return torch.from_numpy(overlay/255).to(rgb.device)

    @torch.no_grad()
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        ori_rgb = outputs["rgb"]

        # rgb = linear_correction(image, rgb)

        if batch.get("msk") is not None:
            msk = batch["msk"].to(self.device)[...,None]
            image = image*msk
            rgb = rgb*msk

        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            colormap_options=colormaps.ColormapOptions(colormap="gray", invert=True)
        )
        err_map = self._make_error_map(image, rgb)
        overlay = self._make_overlay(image, rgb)

        # combined_rgb = torch.cat([image, rgb], dim=1)
        combined_rgb = torch.cat([image, ori_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)


        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)}  # type: ignore
        # TODO(ethan): return an image dictionary

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
            "err_map": err_map,
            "overlay": overlay
        }

        if outputs.get("ev_out") is not None:
            images_dict["ev_out"] = outputs["ev_out"]

        return metrics_dict, images_dict
