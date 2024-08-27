from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, TDataset, VanillaDataManager, variable_res_collate
from nerfstudio.data.datasets.base_dataset import InputDataset

from dataclasses import dataclass, field
from pathlib import Path
import os.path as osp
import os
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

import torch
from torch.nn import Parameter
import typing 


from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.data.pixel_samplers import (
    EquirectangularPixelSampler,
    PatchPixelSampler,
    PixelSampler,
)

from nerfstudio.data.utils.dataloaders import CacheDataloader
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils import writer

from lse_nerf.lse_parser import ColorDataParserConfig, EventDataParserConfig
from lse_nerf.lse_ray_generator import ConsecRayGenerator, PrevNextRayGenerator, DeblurRayGenerator
from lse_nerf.lse_pixel_sampler import EvPixelSampler
from lse_nerf.ns_camera_optimizer import CameraOptimizerConfig
from lse_nerf.lse_loaders import LSEFixedIndicesEvalDataloader, LSERandIndicesEvalDataloader
from lse_nerf.utils import add_metadata, gbconfig
from lse_nerf.lse_cameras import HardCamType
from lse_nerf.data_components import CameraIdxFixer

def get_orig_class(obj, default=None):
    """Returns the __orig_class__ class of `obj` even when it is not initialized in __init__ (Python>=3.8).

    Workaround for https://github.com/python/typing/issues/658.
    Inspired by https://github.com/Stewori/pytypes/pull/53.
    """
    try:
        return object.__getattribute__(obj, "__orig_class__")
    except AttributeError:
        cls = object.__getattribute__(obj, "__class__")
        try:
            is_type_generic = isinstance(cls, typing.GenericMeta)  # type: ignore
        except AttributeError:  # Python 3.8
            is_type_generic = issubclass(cls, typing.Generic)
        if is_type_generic:
            frame = currentframe().f_back.f_back  # type: ignore
            try:
                while frame:
                    try:
                        res = frame.f_locals["self"]
                        if res.__origin__ is cls:
                            return res
                    except (KeyError, AttributeError):
                        frame = frame.f_back
            finally:
                del frame
        return default


@dataclass
class MultiCamManagerConfig(DataManagerConfig):
    _target: Type = field(default_factory=lambda: MultiCamManager)
    col_dataparser: AnnotatedDataParserUnion = ColorDataParserConfig()
    evs_dataparser: AnnotatedDataParserUnion = EventDataParserConfig()


    train_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per training iteration."""

    rgb_frac: float = 0.66
    """ratio of rgb rays used for training"""

    train_num_images_to_sample_from: int = -1
    """Number of images to sample during training iteration."""
    train_num_times_to_repeat_images: int = 1000
    """When not training on all images, number of iterations before picking new
    images. If -1, never pick new images."""

    train_num_evs_images_to_sample_from: int = 1000
    """number of event images to sample from during training iterations"""

    eval_num_rays_per_batch: int = 64
    """Number of rays per batch to use per eval iteration."""
    eval_num_images_to_sample_from: int = -1
    """Number of images to sample during eval iteration."""
    eval_num_times_to_repeat_images: int = -1
    """When not evaluating on all images, number of iterations before picking
    new images. If -1, never pick new images."""
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    """Specifies the image indices to use during eval; if None, uses all."""
    col_cam_optimizer: CameraOptimizerConfig = CameraOptimizerConfig()
    """Specifies the camera pose optimizer for rgb camera used during training. Helpful if poses are noisy, such as for data from
    Record3D."""
    evs_cam_optimizer: CameraOptimizerConfig = CameraOptimizerConfig()
    """Specifies the camera pose optimizer for event camera used during training. Helpful if poses are noisy, such as for data from
    Record3D."""
    collate_fn: Callable[[Any], Any] = cast(Any, staticmethod(nerfstudio_collate))
    """Specifies the collate function to use for the train and eval dataloaders."""
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    patch_size: int = 1
    """Size of patch to sample from. If >1, patch-based sampling will be used."""

    rgb_loss_mode: str = "mse"
    """one of [mse, deblur]"""

    def __post_init__(self):
        self.rgb_loss_mode = self.rgb_loss_mode.lower()
        self.update_num_batch_rays()

    def update_num_batch_rays(self):
        
        # NOTE: times 0.5 because of events
        self.train_num_evs_rays_per_batch = int((1-self.rgb_frac)*self.train_num_rays_per_batch*0.5)

        # NOTE: n_sample = n_samples * (1/n_blur_rays)
        if self.rgb_loss_mode == "deblur":
            self.train_num_col_rays_per_batch = int((self.train_num_rays_per_batch - self.train_num_evs_rays_per_batch*2)*(1/4))
        else:
            self.train_num_col_rays_per_batch = self.train_num_rays_per_batch - self.train_num_evs_rays_per_batch*2


class MultiCamManager(DataManager, Generic[TDataset]):
    config: MultiCamManagerConfig
    col_train_dataset: TDataset
    evs_train_dataset: TDataset
    col_eval_dataset: TDataset

    col_train_dataparser_outputs: DataparserOutputs
    evs_train_dataparser_outputs: DataparserOutputs

    train_col_pixel_sampler: Optional[PixelSampler] = None
    train_evs_pixel_sampler: Optional[PixelSampler] = None
    eval_pixel_sampler: Optional[PixelSampler] = None

    def __init__(
        self,
        config: MultiCamManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        config.update_num_batch_rays()
        self.dataset_type: Type[TDataset] = kwargs.get("_dataset_type", getattr(TDataset, "__default__"))
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"

        # have 2 config & config outputs
        self.col_dataparser_config = self.config.col_dataparser
        self.evs_dataparser_config = self.config.evs_dataparser

        if self.config.data is not None:
            self.config.col_dataparser.data = Path(osp.join(self.config.data, "colcam_set"))
            self.config.evs_dataparser.data = Path(osp.join(self.config.data, "ecam_set"))
        else:
            self.config.data = self.config.col_dataparser.data
        self.col_dataparser = self.col_dataparser_config.setup()
        self.evs_dataparser = self.evs_dataparser_config.setup()

        if test_mode == "inference":
            self.col_dataparser.downscale_factor = 1
            self.evs_dataparser.downscale_factor = 1
        
        self.col_train_dataparser_outputs: DataparserOutputs = self.col_dataparser.get_dataparser_outputs(split="train")
        self.evs_train_dataparser_outputs: DataparserOutputs = self.evs_dataparser.get_dataparser_outputs(split="train")

        self.col_train_dataset = self.col_train_dataparser_outputs.dataset_cls(
            self.col_train_dataparser_outputs,
            self.config.camera_res_scale_factor
        )
        self.evs_train_dataset = self.evs_train_dataparser_outputs.dataset_cls(
            self.evs_train_dataparser_outputs,
            self.config.camera_res_scale_factor
        )

        self.cameraIdxFixer = CameraIdxFixer(self.col_dataparser.get_train_ts())
        
        # eval only on rgbs
        self.eval_dataset = self.create_eval_dataset()

        # NOTE: DO NOT REMOVE; VISUALIZER USES self.train_dataset TO VISUALIZE THE DATA;
        self.train_dataset = self.col_train_dataset
        
        self.num_embd = max(self.evs_dataparser.get_max_appearence_id(), 
                            self.col_dataparser.get_max_appearence_id())
        
        if gbconfig.IS_EVAL:
            self.config.rgb_frac = 1
            self.config.update_num_batch_rays()

        self.exclude_batch_keys_from_device = self.col_train_dataset.exclude_batch_keys_from_device
        if self.config.masks_on_gpu:
            self.exclude_batch_keys_from_device.remove("mask")
        
        self._check_camera_size(self.col_train_dataparser_outputs)
        self._check_camera_size(self.evs_train_dataparser_outputs)

        super().__init__()

    def _check_camera_size(self, dataparser_outputs):
        if dataparser_outputs is None:
            return
        
        cameras = dataparser_outputs.cameras
        if len(cameras) > 1:
            for i in range(1, len(cameras)):
                if cameras[0].width != cameras[i].width or cameras[0].height != cameras[i].height:
                    CONSOLE.print("Variable resolution, using variable_res_collate")
                    self.config.collate_fn = variable_res_collate
                    break
    
    def create_train_dataset(self) -> InputDataset:
        assert 0, "[NOT SUPPORTED], Multicam Manager creates 2 datasets at __init__, breaks creating 1 dataset assumption"
    
    def create_eval_dataset(self) -> InputDataset:
        ## eval only on rgb data
        return self.col_dataparser.config.dataset_cls(
            dataparser_outputs=self.col_dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor
        )


    def setup_train(self):
        assert (self.col_train_dataset is not None) or (self.evs_train_dataset is not None), "no train dataset specified"
        CONSOLE.print("Setting up training dataset")

        self.col_train_dataloader = CacheDataloader(
            self.col_train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )

        n_evs_img_to_sample_from = -1 if self.config.train_num_evs_images_to_sample_from >= len(self.evs_train_dataset) else self.config.train_num_evs_images_to_sample_from
        self.evs_train_dataloader = CacheDataloader(
            self.evs_train_dataset,
            num_images_to_sample_from=n_evs_img_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_col_train_dataloader, self.iter_evs_train_dataloader = iter(self.col_train_dataloader), iter(self.evs_train_dataloader)
        self.train_col_pixel_sampler = EvPixelSampler(self.config.train_num_col_rays_per_batch, device=self.device)

        self.train_evs_pixel_sampler = EvPixelSampler(self.config.train_num_evs_rays_per_batch, device=self.device)

        all_cams = None if self.config.col_cam_optimizer.optim_type != "spline" else self.col_dataparser.get_all_cameras()
        self.col_train_camera_optimizer = self.config.col_cam_optimizer.setup(
            num_cameras=self.col_train_dataset.cameras.size, device=self.device, 
            dM=self.col_train_dataparser_outputs.dM, cameras=all_cams
        )

        cam_interpolator = self.col_train_camera_optimizer if self.config.col_cam_optimizer.optim_type == "spline" else None
        self.col_cams = self.col_train_dataset.cameras.to(self.device)
        self.col_cams.set_interpolator(cam_interpolator)

        evs_cam_interpolator = cam_interpolator if self.config.evs_cam_optimizer.optim_type == "spline" and self.config.col_cam_optimizer.optim_type == "spline" else None
        if self.config.evs_cam_optimizer.optim_type == "spline":
            self.evs_train_camera_optimizer = self.col_train_camera_optimizer
        else:
            if self.evs_train_dataparser_outputs.prev_cameras is not None:
                self.config.evs_cam_optimizer.optim_type = "prevnext"

            self.evs_train_camera_optimizer = self.config.evs_cam_optimizer.setup(
                num_cameras=self.evs_train_dataset.cameras.size, device=self.device
            )

        self.col_ray_generator = self.build_col_ray_generator(self.config.rgb_loss_mode)
        self.col_train_dataset.cameras.set_interpolator(cam_interpolator)

        if self.evs_train_dataparser_outputs.prev_cameras is None:
            evs_cams = self.evs_train_dataset.cameras.to(self.device)
            evs_cams.set_hard_cam_type(HardCamType.EVS)
            self.evs_ray_generator = ConsecRayGenerator(
                evs_cams,
                lambda x : None   ## dummy for camera optimizer
            )
            evs_cams.set_interpolator(evs_cam_interpolator)
        else:
            prev_cams = self.evs_train_dataparser_outputs.prev_cameras.to(self.device)
            next_cams = self.evs_train_dataparser_outputs.next_cameras.to(self.device)
            prev_cams.set_hard_cam_type(HardCamType.EVS), next_cams.set_hard_cam_type(HardCamType.EVS)
            self.evs_ray_generator = PrevNextRayGenerator(
                prev_cameras=prev_cams, next_cameras=next_cams
            )
            prev_cams.set_interpolator(evs_cam_interpolator)
            next_cams.set_interpolator(evs_cam_interpolator)
            
    
    def build_col_ray_generator(self, rgb_loss_mode:str = "linspace"):
        if rgb_loss_mode == "deblur":
            self.col_raygenerator_cls = DeblurRayGenerator
        else:
            self.col_raygenerator_cls = RayGenerator
        
        return self.col_raygenerator_cls(self.col_cams, lambda x : None)

    
    def next_train(self, step:int):
        """Returns the next batch of data from the train dataloader."""

        self.col_train_camera_optimizer.update_mode(step)
        self.evs_train_camera_optimizer.update_mode(step)

        if self.get_train_col_rays_per_batch() > 0:
            col_batch = next(self.iter_col_train_dataloader)
            assert self.train_col_pixel_sampler is not None
            assert isinstance(col_batch, dict)
            col_batch = self.train_col_pixel_sampler.sample(col_batch)
            col_ray_indices = col_batch["indices"]
            col_ray_bundle = self.col_ray_generator(col_ray_indices)
            self.col_train_camera_optimizer.apply_to_raybundle(col_ray_bundle)
            col_ray_bundle = add_metadata(col_ray_bundle, col_batch, max_app_id=self.num_embd)
        else:
            col_ray_bundle, col_batch = None, None

        if self.get_train_evs_rays_per_batch() > 0:
            evs_batch = next(self.iter_evs_train_dataloader)
            assert self.train_evs_pixel_sampler is not None
            assert isinstance(evs_batch, dict)
            evs_batch = self.train_evs_pixel_sampler.sample(evs_batch)
            evs_ray_indices = evs_batch["indices"]
            prev_ray_bundle, next_ray_bundle = self.evs_ray_generator(evs_ray_indices)

            self.evs_train_camera_optimizer.apply_to_raybundle(prev_ray_bundle)
            self.evs_train_camera_optimizer.apply_to_raybundle(next_ray_bundle)
            prev_ray_bundle = add_metadata(prev_ray_bundle, evs_batch, cam_id=1)
            next_ray_bundle = add_metadata(next_ray_bundle, evs_batch, cam_id=1)
        else:
            prev_ray_bundle, next_ray_bundle, evs_batch = None, None, None
        
        prev_ray_bundle, next_ray_bundle = self.cameraIdxFixer(prev_ray_bundle), self.cameraIdxFixer(next_ray_bundle)

        return (col_ray_bundle, prev_ray_bundle, next_ray_bundle), {"col_batch":col_batch, "evs_batch":evs_batch}
    
    
    def get_train_col_rays_per_batch(self) -> int:
        return self.config.train_num_col_rays_per_batch
    
    def get_train_evs_rays_per_batch(self) -> int:
        return self.config.train_num_evs_rays_per_batch
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        param_groups = {}

        def add_params(params: list):
            if param_groups.get("camera_opt") is None:
                param_groups["camera_opt"] = params
            else:
                param_groups["camera_opt"] = param_groups["camera_opt"] + params

        col_camera_opt_params = list(self.col_train_camera_optimizer.parameters())
        evs_camera_opt_params = list(self.evs_train_camera_optimizer.parameters())
        if self.config.col_cam_optimizer.mode != "off":
            if self.get_train_col_rays_per_batch() > 0:
                assert len(col_camera_opt_params) > 0
                add_params(col_camera_opt_params)

        
        if self.config.evs_cam_optimizer.mode != "off" and (self.config.col_cam_optimizer.mode != "spline"):
            if self.get_train_evs_rays_per_batch() > 0:
                assert len(evs_camera_opt_params) > 0
                add_params(evs_camera_opt_params)


        return param_groups

    # Can't inherit correctly, just making a copy here
    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.eval_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        
        if gbconfig.IS_EVAL:
            self.col_train_camera_optimizer.apply_to_raybundle(ray_bundle)
        
        add_metadata(ray_bundle, batch, 0)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        for camera_ray_bundle, batch in self.eval_dataloader:
            assert camera_ray_bundle.camera_indices is not None
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
            return image_idx, camera_ray_bundle, batch
        raise ValueError("No more eval images")

    def get_train_rays_per_batch(self) -> int:
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        return self.config.eval_num_rays_per_batch
    

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_image_dataloader = CacheDataloader(
            self.eval_dataset,
            num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        self.eval_pixel_sampler = EvPixelSampler(self.config.eval_num_rays_per_batch)

        self.eval_ray_generator = RayGenerator(
            self.eval_dataset.cameras.to(self.device),
            lambda x : None ## dummy for camera optimizer
        ).to(self.device)

        cam_opt = None
        if gbconfig.IS_EVAL:
            cam_opt = self.col_train_camera_optimizer

        # for loading full images
        self.fixed_indices_eval_dataloader = LSEFixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
            camera_opt=cam_opt
        )
        self.eval_dataloader = LSERandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
            camera_opt=cam_opt
        )
    
    def get_num_emb(self, emb_type:str):
        if emb_type == "rgb_emb":
            return self.col_dataparser.get_num_train()
        else:
            return self.num_embd
    
    def param_to_pipeline_config(self, config):
        config.model.embed_config.metadata = {"col_train_idxs" : self.col_dataparser.get_train_ids(),
                                              "col_dataparser" : self.col_dataparser}
