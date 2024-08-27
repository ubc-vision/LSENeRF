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
Pose and Intrinsics Optimizers
"""

# taken from nerfstudio==1.0.0
# updated tyro in newer version of nerfstudio breaks the framework

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Literal, Optional, Type, Union

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
import torch
from typing_extensions import assert_never
import numpy as np


from nerfstudio.cameras.lie_groups import exp_map_SE3, exp_map_SO3xR3
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.utils import poses as pose_utils
from nerfstudio.cameras.cameras import CameraType

from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

from lse_nerf.interpolation_utils import (matrix_to_tangent_vector, 
                                         hom_exp_map_SO3xR3, 
                                         vectorized_generalized_interpolation,
                                         exp_map_to_quat_map,
                                         quat_map_to_mtx)
from lse_nerf.lse_cameras import HardCamType
from lse_nerf.lse_cameras import EdCameras as Cameras


class SplineCameraOptimizer(nn.Module):

    config: CameraOptimizer

    def __init__(
        self,
        config: CameraOptimizerConfig,
        num_cameras: int,
        device: Union[torch.device, str],
        cameras: Cameras,
        dM: torch.Tensor = None,
        non_trainable_camera_indices: Optional[Int[Tensor, "num_non_trainable_cameras"]] = None,
        **kwargs,
    )-> None:
        super().__init__()
        self.config = config
        self.device = device
        self.register_buffer("dM", dM)
        self.pnt_factor = self.config.control_pnt_factor
        self.cameras = cameras

        self.build_control_pnts(self.cameras, n_factor=self.pnt_factor)
        self.scale = nn.Parameter(torch.ones(1))
        self.get_fn_dic = {HardCamType.RGB : self.get_rgb_cameras,
                           HardCamType.EVS : self.get_evs_cameras}
        # self.exp_t = 14990  # exposure time
        self.exp_t = config.exp_t    # exposure time
        self.n_deblur_rays = 4

        self.is_on = True        
        self.ori_mode = self.config.mode
        if self.config.scheme == "delayed":
            self.config.mode = "off"
            self.is_on = False

    def turn_on(self):
        self.config.mode = self.ori_mode
        self.is_on = True
    
    def update_mode(self, step):
        if self.is_on:
            return
        
        if self.config.scheme == "delayed" and step > self.config.delay_cnt:
            self.turn_on()

    def build_control_pnts(self, cameras:Cameras, n_factor = 1):
        w2c = cameras.camera_to_worlds.numpy()
        cam_ts = cameras.times.numpy().squeeze()

        Rs = w2c[:,:3,:3]
        rot_interp = Slerp(cam_ts, Rotation.from_matrix(Rs))
        trans_interp = interp1d(cam_ts, w2c[:,:3,3], axis=0, kind="linear")

        max_err = np.abs(rot_interp(cam_ts[0]).as_matrix() - w2c[0][:3,:3]).max()
        assert max_err < 1e-5, f"ERROR {max_err}, w2cs are mirror transforms"

        ctrl_dts = (np.diff(cam_ts)/n_factor).reshape(-1, 1)
        i_s = np.arange(0, n_factor, dtype=np.int32).reshape(1, -1)
        ctrl_ts = np.concatenate([(cam_ts.reshape(-1, 1)[:-1] + ctrl_dts*i_s).reshape(-1).astype(np.float32), cam_ts[-1:]])

        intrp_Rs = rot_interp(ctrl_ts)
        intrp_Ts = trans_interp(ctrl_ts)
        ctrl_c2ws = np.concatenate([intrp_Rs.as_matrix(), intrp_Ts[..., None]], axis=-1)

        # to homogeneous
        ctrl_c2ws = np.concatenate([ctrl_c2ws, np.tile(np.array([0,0,0,1])[None, None], (len(ctrl_c2ws), 1, 1))], axis=1)
        ctrl_tangent = torch.stack([matrix_to_tangent_vector(torch.from_numpy(M)) for M in ctrl_c2ws])

        self.ctrl_tangents = torch.nn.Parameter(ctrl_tangent, requires_grad=True)
        self.register_buffer("ctrl_ts", torch.tensor(ctrl_ts, dtype=torch.float32, device=self.device))
        self.register_buffer("orig_cam_ts", cameras.times)

        assert len(self.ctrl_ts) == len(ctrl_c2ws)
        
    def get_rgb_cameras(self, times: torch.tensor):
        """
        return a (n, 3, 4) camrea matrix
        """

        def exec(times):
            ts = torch.clip(times, self.ctrl_ts[0], self.ctrl_ts[-1]).squeeze()
            ctrl_quats = exp_map_to_quat_map(self.ctrl_tangents)
            interp_vec = vectorized_generalized_interpolation(ctrl_quats, self.ctrl_ts, ts)
            interp_M = quat_map_to_mtx(interp_vec)
            
            return interp_M[:,:3,:4]
        
        if self.config.mode == "off":
            with torch.no_grad():
                return exec(times)
        else:
            return exec(times)

    
    def get_evs_cameras(self, times: torch.tensor):

        def get_rel_cam():
            if self.config.mode == "off":
                return self.dM

            updated_col = self.dM[:3, 3:4] * self.scale  # Multiply the specified part by s
            unchanged_col = self.dM[3:, 3:4]  # The part of the column not affected by s

            # Reconstruct dM without in-place modification
            return torch.cat((self.dM[:, :3], 
                              torch.cat((updated_col, unchanged_col), dim=0)), 
                              dim=1)


        def exec(times):
            interp_M = self.get_rgb_cameras(times)

            return interp_M @ get_rel_cam()
        
        if self.config.mode == "off":
            with torch.no_grad():
                return exec(times)
        else:
            return exec(times)
        

    
    def get_deblur_cameras(self, cam_ts):
        """
        cam_ts (torch.array): shape (n, 1)
        """

        def exec(cam_ts):
            st_t = cam_ts - self.exp_t / 2

            delta_t = self.exp_t/(self.n_deblur_rays - 1)
            t_steps = delta_t * torch.torch.arange(self.n_deblur_rays, device=cam_ts.device)
            cam_ts = (st_t + t_steps[None]).reshape(-1)
            rgb_c2w = self.get_rgb_cameras(cam_ts)

            return rgb_c2w

        if self.config.mode == "off":
            with torch.no_grad():
                return exec(cam_ts)
        else:
            return exec(cam_ts)


    def apply_to_raybundle(self, raybundle: RayBundle) -> None:
        """Apply the pose correction to the raybundle
        NOTE: If this class is initialized, the matrix is already in a correct state.
        """
        return raybundle
    
    def get_param_groups(self, param_groups: dict) -> None:
        """Get camera optimizer parameters"""
        camera_opt_params = list(self.parameters())
        if self.config.mode != "off":
            assert len(camera_opt_params) > 0
            param_groups["camera_opt"] = camera_opt_params


class CameraOptimizer(nn.Module):
    """Layer that modifies camera poses to be optimized as well as the field during training."""

    config: CameraOptimizerConfig

    def __init__(
        self,
        config: CameraOptimizerConfig,
        num_cameras: int,
        device: Union[torch.device, str],
        non_trainable_camera_indices: Optional[Int[Tensor, "num_non_trainable_cameras"]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_cameras = num_cameras
        self.device = device
        self.non_trainable_camera_indices = non_trainable_camera_indices

        # Initialize learnable parameters.
        if self.config.mode == "off":
            pass
        elif self.config.mode in ("SO3xR3", "SE3"):
            self.pose_adjustment = torch.nn.Parameter(torch.zeros((num_cameras, 6), device=device))
        else:
            assert_never(self.config.mode)

        self.is_on = True        
        self.ori_mode = self.config.mode
        if self.config.scheme == "delayed":
            self.config.mode = "off"
            self.is_on = False

    def turn_on(self):
        self.config.mode = self.ori_mode
        self.is_on = True
    
    def update_mode(self, step):
        if self.is_on:
            return
        
        if self.config.scheme == "delayed" and step > self.config.delay_cnt:
            self.turn_on()

    def forward(
        self,
        indices: Int[Tensor, "camera_indices"],
    ) -> Float[Tensor, "camera_indices 3 4"]:
        """Indexing into camera adjustments.
        Args:
            indices: indices of Cameras to optimize.
        Returns:
            Transformation matrices from optimized camera coordinates
            to given camera coordinates.
        """
        outputs = []

        # Apply learned transformation delta.
        if self.config.mode == "off":
            pass
        elif self.config.mode == "SO3xR3":
            outputs.append(exp_map_SO3xR3(self.pose_adjustment[indices, :]))
        elif self.config.mode == "SE3":
            outputs.append(exp_map_SE3(self.pose_adjustment[indices, :]))
        else:
            assert_never(self.config.mode)
        # Detach non-trainable indices by setting to identity transform
        if self.non_trainable_camera_indices is not None:
            if self.non_trainable_camera_indices.device != self.pose_adjustment.device:
                self.non_trainable_camera_indices = self.non_trainable_camera_indices.to(self.pose_adjustment.device)
            outputs[0][self.non_trainable_camera_indices] = torch.eye(4, device=self.pose_adjustment.device)[:3, :4]

        # Return: identity if no transforms are needed, otherwise multiply transforms together.
        if len(outputs) == 0:
            # Note that using repeat() instead of tile() here would result in unnecessary copies.
            return torch.eye(4, device=self.device)[None, :3, :4].tile(indices.shape[0], 1, 1)
        return functools.reduce(pose_utils.multiply, outputs)

    def _flatten_raybundle(self, raybundle: RayBundle):
        """
        flatten raybundle of shape (k1,k2..., d) to (n, d)
        """
        if len(raybundle.origins.shape) == 2:
            return
        
        self.raybundle_ori_shape = raybundle.origins.shape[:-1]
        
        flatten_fn = lambda x : torch.flatten(x, start_dim=0, end_dim=-2)
        raybundle.camera_indices = flatten_fn(raybundle.camera_indices)
        raybundle.origins = flatten_fn(raybundle.origins)
        raybundle.directions = flatten_fn(raybundle.directions)
    
    def _unflatten_raybundle(self, raybundle:RayBundle):
        """
        return flatten raybundle to original shape
        """

        if not hasattr(self, "raybundle_ori_shape") or (self.raybundle_ori_shape is None):
            return

        unflatten_fn = lambda x : x.reshape(*self.raybundle_ori_shape, x.shape[-1])
        raybundle.camera_indices = unflatten_fn(raybundle.camera_indices)
        raybundle.origins = unflatten_fn(raybundle.origins)
        raybundle.directions = unflatten_fn(raybundle.directions)

        self.raybundle_ori_shape = None


    def apply_to_raybundle(self, raybundle: RayBundle) -> None:
        """Apply the pose correction to the raybundle"""
        if self.config.mode != "off":
            self._flatten_raybundle(raybundle)
            correction_matrices = self(raybundle.camera_indices.squeeze())  # type: ignore
            raybundle.origins = raybundle.origins + correction_matrices[:, :3, 3]
            raybundle.directions = torch.bmm(correction_matrices[:, :3, :3], raybundle.directions[..., None]).squeeze()
            self._unflatten_raybundle(raybundle)

    def apply_to_camera(self, camera: Cameras) -> None:
        """Apply the pose correction to the raybundle"""
        if self.config.mode != "off":
            assert camera.metadata is not None, "Must provide id of camera in its metadata"
            assert "cam_idx" in camera.metadata, "Must provide id of camera in its metadata"
            camera_idx = camera.metadata["cam_idx"]
            adj = self([camera_idx])  # type: ignore
            adj = torch.cat([adj, torch.Tensor([0, 0, 0, 1])[None, None].to(adj)], dim=1)
            camera.camera_to_worlds = torch.bmm(camera.camera_to_worlds, adj)

    def get_loss_dict(self, loss_dict: dict, prefix="") -> None:
        """Add regularization"""
        if self.config.mode != "off":
            loss_dict[f"{prefix}camera_opt_regularizer"] = (
                self.pose_adjustment[:, :3].norm(dim=-1).mean() * self.config.trans_l2_penalty
                + self.pose_adjustment[:, 3:].norm(dim=-1).mean() * self.config.rot_l2_penalty
            )

    def get_correction_matrices(self):
        """Get optimized pose correction matrices"""
        return self(torch.arange(0, self.num_cameras).long())

    def get_metrics_dict(self, metrics_dict: dict, prefix: str = "") -> None:
        """Get camera optimizer metrics"""
        if self.config.mode != "off":
            metrics_dict[f"{prefix}camera_opt_translation"] = self.pose_adjustment[:, :3].norm()
            metrics_dict[f"{prefix}camera_opt_rotation"] = self.pose_adjustment[:, 3:].norm()

    def get_param_groups(self, param_groups: dict, prefix = "") -> None:
        """Get camera optimizer parameters"""
        camera_opt_params = list(self.parameters())
        if self.config.mode != "off":
            assert len(camera_opt_params) > 0
            param_groups[f"{prefix}camera_opt"] = camera_opt_params
        else:
            assert len(camera_opt_params) == 0

class PrevNextCamOptimizer(nn.Module):
    def __init__(
        self,
        config: CameraOptimizerConfig,
        num_cameras: int,
        device: Union[torch.device, str],
        non_trainable_camera_indices: Optional[Int[Tensor, "num_non_trainable_cameras"]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.prev_optim = CameraOptimizer(config, num_cameras,device, non_trainable_camera_indices, **kwargs)
        self.next_optim = CameraOptimizer(config, num_cameras,device, non_trainable_camera_indices, **kwargs)
        self.cnt_call = 0

    def turn_on(self):
        self.prev_optim.turn_on()
        self.next_optim.turn_on()
    
    def update_mode(self, step):
        self.prev_optim.update_mode(step)
        self.next_optim.update_mode(step)
    
    def forward(self, indices):
        assert 0, "not implemented"
    
    def applay_to_camera(self, camera):
        assert 0, "not implemented"
    
    def apply_to_raybundle(self, raybundle: RayBundle):
        if self.cnt_call%2 == 0:
            self.prev_optim.apply_to_raybundle(raybundle)
        else:
            self.next_optim.apply_to_raybundle(raybundle)
        
        self.cnt_call = (self.cnt_call + 1)%2
    
    def get_loss_dict(self, loss_dict):
        self.prev_optim.get_loss_dict(loss_dict, "prev_")
        self.next_optim.get_loss_dict(loss_dict, "next_")
    
    def get_metrics_dict(self, metrics_dict):
        self.prev_optim.get_metrics_dict(metrics_dict, "prev_")
        self.next_optim.get_metrics_dict(metrics_dict, "next_")
    
    def get_param_groups(self, param_groups):
        self.prev_optim.get_param_groups(param_groups, "prev_")
        self.next_optim.get_param_groups(param_groups, "next_")

CAM_OPTIM_DICT = {"ns": CameraOptimizer, 
                  "spline": SplineCameraOptimizer, 
                  "prevnext": PrevNextCamOptimizer}

@dataclass
class CameraOptimizerConfig(InstantiateConfig):
    """Configuration of optimization for camera poses."""

    _target: Type = field(default_factory=lambda: CameraOptimizer)

    mode: Literal["off", "SO3xR3", "SE3"] = "off"
    """Pose optimization strategy to use. If enabled, we recommend SO3xR3."""

    trans_l2_penalty: float = 1e-2
    """L2 penalty on translation parameters."""

    rot_l2_penalty: float = 1e-3
    """L2 penalty on rotation parameters."""

    optim_type: str = "ns"
    """one of [ns, spline] where ns is vanilla nerfstudio and spline"""

    control_pnt_factor: int = 1
    """number of control points to use"""

    scheme: Literal["active", "delayed"] = "active"
    """the optim scheme. If delayed, cam opt will be turned on after delay_cnt of train iter in datamanager"""

    delay_cnt: int = 10000
    """number of train iteration before turning on camera opt"""

    exp_t: float = 30000
    """exposure time of camera"""

    def __post_init__(self):
        if self.mode == "off":
            self.scheme = "active"
            self.delay_cnt = 1999999999 # set to large train_step that will never happen

    def setup(self, **kwargs) -> functools.Any:
        _target = CAM_OPTIM_DICT[self.optim_type]
        return _target(self, **kwargs)


###################################### TESTS BELOW ######################################
def create_spline(c2ws:np.ndarray, ts: np.ndarray, mode:str = "off"):
    # c2w = nx3x4
    cameras = Cameras(
        camera_to_worlds=torch.from_numpy(c2ws),
        fx = 100., fy = 100., cx = 64., cy = 64., width=128, height=128, distortion_params=None, 
        camera_type=CameraType.PERSPECTIVE, 
        times = torch.from_numpy(ts)
    )

    config = CameraOptimizerConfig(optim_type="spline", mode=mode)
    spline = config.setup(
        num_cameras=len(cameras), device = "cpu",
        dM = None, 
        cameras = cameras,
        scheme = "active"
    )

    return spline


class SciSpline:
    def __init__(self, w2cs, ts) -> None:
        self.w2cs = w2cs 
        self.ts = ts

        self.Rs = w2cs[:,:3,:3]
        self.trans = w2cs[:,:3,3:]

        self.rot_interp = Slerp(self.ts, Rotation.from_matrix(self.Rs))
        self.trans_interp = interp1d(self.ts, self.trans, axis=0, kind="linear")
    
    def interpolate(self, ts):
        ts = np.clip(ts, self.ts[0] + 1e-6, self.ts[-1] - 1e-6)
        r, t = self.rot_interp(ts).as_matrix(), self.trans_interp(ts)
        return np.concatenate([r, t], axis=2)
        
def gen_data(n_rnd = 10, n_bins=2, max_t = 10):
    np.random.seed(0)
    rots = Rotation.random(n_rnd).as_matrix()
    trans = np.random.rand(n_rnd, 3, 1) * 4
    w2cs = np.concatenate([rots, trans], axis= 2)
    ts = np.arange(0, max_t, max_t/n_rnd)
    return ts.astype(np.float32), w2cs.astype(np.float32)

def test_spline_imp(n_rnd = 10, n_bins=2, max_t = 10):
    ts, w2cs = gen_data(n_rnd, n_bins, max_t)
    # ts, w2cs = ts[:2], w2cs[:2]
    interp_ts = np.arange(0, max_t, max_t/(n_bins * n_rnd)).astype(np.float32)
    interp_ts = interp_ts[interp_ts <= ts.max()][:2]


    spline = create_spline(w2cs, ts)
    sspline = SciSpline(w2cs, ts)

    tcams = spline.get_rgb_cameras(torch.from_numpy(interp_ts))
    scams = sspline.interpolate(interp_ts)

    assert torch.abs(tcams - scams).mean() < 1e-5
    print("test pass")


def test_grad_exists(n_rnd = 10, n_bins=2, max_t = 10):
    ts, w2cs = gen_data(n_rnd, n_bins, max_t)
    # ts, w2cs = ts[:2], w2cs[:2]
    interp_ts = np.arange(0, max_t, max_t/(n_bins * n_rnd)).astype(np.float32)
    interp_ts = interp_ts[interp_ts <= ts.max()][:2]


    spline: SplineCameraOptimizer = create_spline(w2cs, ts, mode="on")

    tcams = spline.get_rgb_cameras(torch.from_numpy(interp_ts))

    loss = (tcams**2).sum()
    loss.backward()
    assert not (spline.ctrl_tangents.grad is None)

    assert 0


def test_learning(n_cams = 10, n_interp = 3, max_t=10):
    ts, w2cs = gen_data(n_cams, n_interp, max_t)
    interp_ts = np.arange(0, max_t, max_t/(n_interp * n_cams)).astype(np.float32)
    interp_ts = interp_ts[interp_ts <= ts.max()]
    
    tmp, eps = [], 1e-3
    from tqdm import tqdm
    for it in tqdm(interp_ts, desc="checking interp ts"):
        keep = True
        for st in ts:
            if np.abs(it - st) < eps:
                keep = False
        if keep:
            tmp.append(it)
    
    interp_ts = np.array(tmp)

    gt_spline: SplineCameraOptimizer = create_spline(w2cs, ts, mode="on")

    def perterb_angle(mtx, var):
        angle = np.random.normal(loc=0, scale=var, size=3)
        delta = Rotation.from_euler("xyz", angle).as_matrix()
        return mtx @ delta
    
    p_w2cs = np.copy(w2cs)
    for i in range(len(p_w2cs)):
        R, t = p_w2cs[i, :3, :3], p_w2cs[i, :3, 3:]
        Ro = perterb_angle(R, np.pi / 180 * 10)
        to = t + ((np.random.rand(*t.shape) - 0.5) * 2)*0.5
        p_w2cs[i] = np.concatenate([Ro, to], axis=-1)
    
    init_spline: SplineCameraOptimizer = create_spline(p_w2cs, ts, mode="on")

    with torch.no_grad():
        interp_ts = torch.from_numpy(interp_ts).float()
        gt_cams = gt_spline.get_rgb_cameras(interp_ts).float()

    optimizer = torch.optim.Adam(init_spline.parameters(), lr=1e-4)

    with torch.no_grad():
        initial_error = (gt_spline.ctrl_tangents - init_spline.ctrl_tangents).abs().mean()
        print(f"initial error: {initial_error}")

    print(gt_spline.ctrl_tangents[0])
    print(init_spline.ctrl_tangents[0])
    interp_ts, gt_cams, init_spline, gt_spline = list(map(lambda x: x.to("cuda"), [interp_ts, gt_cams, init_spline, gt_spline]))
    for i in range(10000):
        optimizer.zero_grad()
        cams = init_spline.get_rgb_cameras(interp_ts)
        loss = torch.nn.functional.mse_loss(cams, gt_cams)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"{i} - loss: {loss.item()}")
    
    final_error = (gt_spline.ctrl_tangents - init_spline.ctrl_tangents).abs().mean()
    print(f"final error: {final_error.item()}")
    print(gt_spline.ctrl_tangents[0])
    print(init_spline.ctrl_tangents[0])


if __name__ == "__main__":
    # test_spline_imp()
    # test_grad_exists()
    test_learning()
    

