from typing import Optional, Tuple, Union
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader, RandIndicesEvalDataloader
from lse_nerf.ns_camera_optimizer import CameraOptimizer
import torch

class LSEFixedIndicesEvalDataloader(FixedIndicesEvalDataloader):
    def __init__(
        self,
        input_dataset: InputDataset,
        image_indices: Optional[Tuple[int]] = None,
        device: Union[torch.device, str] = "cpu",
        camera_opt:CameraOptimizer = None,
        **kwargs,
    ):
        super().__init__(input_dataset, image_indices, device, **kwargs)
        self.camera_opt = camera_opt
    

    def __next__(self):
        ray_bundle, batch = super().__next__()
        if not (self.camera_opt is None):
            self.camera_opt.apply_to_raybundle(ray_bundle)
        
        ray_bundle.metadata["appearance_id"] = torch.full(ray_bundle.camera_indices.shape, 
                                                          batch["appearance_id"], device=ray_bundle.camera_indices.device)
        return ray_bundle, batch


class LSERandIndicesEvalDataloader(RandIndicesEvalDataloader):
    def __init__(
        self,
        input_dataset: InputDataset,
        device: Union[torch.device, str] = "cpu",
        camera_opt: CameraOptimizer = None,
        **kwargs,
    ):
        super().__init__(input_dataset, device, **kwargs)
        self.camera_opt = camera_opt
    
    def __next__(self):
        ray_bundle, batch = super().__next__()
        if not (self.camera_opt is None):
            self.camera_opt.apply_to_raybundle(ray_bundle)
        
        ray_bundle.metadata["appearance_id"] = torch.full(ray_bundle.camera_indices.shape, 
                                                          batch["appearance_id"], device=ray_bundle.camera_indices.device)
        return ray_bundle, batch
