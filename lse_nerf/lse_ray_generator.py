from jaxtyping import Int
from torch import Tensor

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.model_components.ray_generators import RayGenerator

# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Ray generator.
"""
from jaxtyping import Int
from torch import Tensor, nn
import torch
from types import MethodType

from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle

from lse_nerf.lse_cameras import EdCameras


class ConsecRayGenerator(RayGenerator):
    """torch.nn Module for generating rays.
    This class is the interface between the scene's cameras/camera optimizer and the ray sampler.

    Args:
        cameras: Camera objects containing camera info.
        pose_optimizer: pose optimization module, for optimizing noisy camera intrinsics/extrinsics.
    """


    def forward(self, ray_indices: Int[Tensor, "num_rays 3"]) -> RayBundle:
        """Index into the cameras to generate the rays.

        Args:
            ray_indices: Contains camera, row, and col indices for target rays.
        """
        c = ray_indices[:, 0]  # camera indices
        y = ray_indices[:, 1]  # row indices
        x = ray_indices[:, 2]  # col indices
        coords = self.image_coords[y, x]


        ray_bundle_prev = self.cameras.generate_rays(
            camera_indices=c.unsqueeze(-1),
            coords=coords
        )
        ray_bundle_next = self.cameras.generate_rays(
            camera_indices=c.unsqueeze(-1) + 1,
            coords=coords
        )

        # ray_bundle = concat_rbs(ray_bundle_prev, ray_bundle_next)
        return ray_bundle_prev, ray_bundle_next


class PrevNextRayGenerator(RayGenerator):

    def __init__(self, prev_cameras:Cameras, next_cameras:Cameras, pose_optimizer: CameraOptimizer = lambda x : None) -> None:
        super().__init__(prev_cameras, lambda x : None)
        self.prev_cameras = prev_cameras
        self.next_cameras = next_cameras
    

    def forward(self, ray_indices: Int[Tensor, "num_rays 3"]) -> RayBundle:
        """Index into the cameras to generate the rays.

        Args:
            ray_indices: Contains camera, row, and col indices for target rays.
        """
        c = ray_indices[:, 0]  # camera indices
        y = ray_indices[:, 1]  # row indices
        x = ray_indices[:, 2]  # col indices
        coords = self.image_coords[y, x]


        ray_bundle_prev = self.prev_cameras.generate_rays(
            camera_indices=c.unsqueeze(-1),
            coords=coords
        )
        ray_bundle_next = self.next_cameras.generate_rays(
            camera_indices=c.unsqueeze(-1),
            coords=coords
        )

        return ray_bundle_prev, ray_bundle_next


class DeblurRayGenerator(RayGenerator):
    cameras: EdCameras = None

    def generate_ray_bundle(self, ray_indices):

        c = ray_indices[:, 0]  # camera indices
        y = ray_indices[:, 1]  # row indices
        x = ray_indices[:, 2]  # col indices
        coords = self.image_coords[y, x]

        coords = torch.tile(coords[:, None], (1, 4, 1)).reshape(-1, coords.shape[-1])

        ray_bundle = self.cameras.generate_rays(
            camera_indices=c.unsqueeze(-1),
            coords=coords,
            camera_opt_to_camera=None,
        )

        return ray_bundle


    def forward(self, ray_indices: Tensor) -> RayBundle:
        assert self.cameras.interpolator is not None, "requires interpolator!!"

        # There is no config for camera class, so defining it here.
        def get_deblur_c2w(cameras:EdCameras, camera_indices):
            true_indices = [camera_indices[..., i] for i in range(camera_indices.shape[-1])]
            cam_ts = cameras.times[camera_indices].squeeze()[..., None]
            c2ws = self.cameras.interpolator.get_deblur_cameras(cam_ts)

            true_indices = [torch.stack(true_indices * 4, dim=1).reshape(-1)]

            return c2ws, true_indices

        
        ori_fn = self.cameras.get_c2w_fn
        self.cameras.get_c2w_fn = MethodType(get_deblur_c2w, self.cameras)


        ray_bundle = self.generate_ray_bundle(ray_indices)


        self.cameras.get_c2w_fn = ori_fn
        
        return ray_bundle
