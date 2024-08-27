from typing import Dict, List
from jaxtyping import Float, Int, Shaped
from torch import Tensor
import torch
from torch.nn import Parameter

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.cameras.cameras import CameraType, Cameras
import nerfstudio.utils.poses as pose_utils
import nerfstudio
from nerfstudio.data.scene_box import SceneBox

from typing import Dict, List, Literal, Optional, Tuple, Union
import math

from lse_nerf.utils import fix_datashape

class HardCamType:
    """
    hardware camera type
    """
    RGB = 0
    EVS = 1

class EdCameras(Cameras):

    def __init__(
        self,
        camera_to_worlds: Float[Tensor, "*batch_c2ws 3 4"],
        fx: Union[Float[Tensor, "*batch_fxs 1"], float],
        fy: Union[Float[Tensor, "*batch_fys 1"], float],
        cx: Union[Float[Tensor, "*batch_cxs 1"], float],
        cy: Union[Float[Tensor, "*batch_cys 1"], float],
        width: Optional[Union[Shaped[Tensor, "*batch_ws 1"], int]] = None,
        height: Optional[Union[Shaped[Tensor, "*batch_hs 1"], int]] = None,
        distortion_params: Optional[Float[Tensor, "*batch_dist_params 6"]] = None,
        camera_type: Union[
            Int[Tensor, "*batch_cam_types 1"],
            int,
            List[CameraType],
            CameraType,
        ] = CameraType.PERSPECTIVE,
        times: Optional[Float[Tensor, "num_cameras"]] = None,
        metadata: Optional[Dict] = None,
    ):
        super().__init__(camera_to_worlds, fx, fy, cx, cy, width, height, distortion_params, camera_type, times, metadata)
        self.interpolator = None
        self.hard_cam_type = HardCamType.RGB
        self.get_c2w_fn = self.get_c2w
    
    def get_c2w(self, camera_indices):
        true_indices = [camera_indices[..., i] for i in range(camera_indices.shape[-1])]
        if self.interpolator is not None:
            c2w = self.interpolator.get_fn_dic[self.hard_cam_type](self.times[camera_indices])
        else:
            c2w = self.camera_to_worlds[true_indices]
        
        return c2w, true_indices


    
    def set_interpolator(self, interpolator) -> None:
        self.interpolator = interpolator
    
    def set_hard_cam_type(self, hard_cam_type: int) -> None:
        self.hard_cam_type = hard_cam_type
    
    def get_image_coords(
        self, pixel_offset: float = 0.5, index: Optional[Tuple] = None
    ) -> Float[Tensor, "height width 2"]:
        # WE REMOVE THIS IMAGE COORDINET OFFSET
        return super().get_image_coords(0, index)
    
    def generate_rays(
        self,
        camera_indices: Union[Int[Tensor, "*num_rays num_cameras_batch_dims"], int],
        coords: Optional[Float[Tensor, "*num_rays 2"]] = None,
        camera_opt_to_camera: Optional[Float[Tensor, "*num_rays 3 4"]] = None,
        distortion_params_delta: Optional[Float[Tensor, "*num_rays 6"]] = None,
        keep_shape: Optional[bool] = None,
        disable_distortion: bool = False,
        aabb_box: Optional[SceneBox] = None,
    ) -> RayBundle:
        """Generates rays for the given camera indices.

        This function will standardize the input arguments and then call the _generate_rays_from_coords function
        to generate the rays. Our goal is to parse the arguments and then get them into the right shape:
            - camera_indices: (num_rays:..., num_cameras_batch_dims)
            - coords: (num_rays:..., 2)
            - camera_opt_to_camera: (num_rays:..., 3, 4) or None
            - distortion_params_delta: (num_rays:..., 6) or None

        Read the docstring for _generate_rays_from_coords for more information on how we generate the rays
        after we have standardized the arguments.

        We are only concerned about different combinations of camera_indices and coords matrices, and the following
        are the 4 cases we have to deal with:
            1. isinstance(camera_indices, int) and coords == None
                - In this case we broadcast our camera_indices / coords shape (h, w, 1 / 2 respectively)
            2. isinstance(camera_indices, int) and coords != None
                - In this case, we broadcast camera_indices to the same batch dim as coords
            3. not isinstance(camera_indices, int) and coords == None
                - In this case, we will need to set coords so that it is of shape (h, w, num_rays, 2), and broadcast
                    all our other args to match the new definition of num_rays := (h, w) + num_rays
            4. not isinstance(camera_indices, int) and coords != None
                - In this case, we have nothing to do, only check that the arguments are of the correct shape

        There is one more edge case we need to be careful with: when we have "jagged cameras" (ie: different heights
        and widths for each camera). This isn't problematic when we specify coords, since coords is already a tensor.
        When coords == None (ie: when we render out the whole image associated with this camera), we run into problems
        since there's no way to stack each coordinate map as all coordinate maps are all different shapes. In this case,
        we will need to flatten each individual coordinate map and concatenate them, giving us only one batch dimension,
        regardless of the number of prepended extra batch dimensions in the camera_indices tensor.


        Args:
            camera_indices: Camera indices of the flattened cameras object to generate rays for.
            coords: Coordinates of the pixels to generate rays for. If None, the full image will be rendered.
            camera_opt_to_camera: Optional transform for the camera to world matrices.
            distortion_params_delta: Optional delta for the distortion parameters.
            keep_shape: If None, then we default to the regular behavior of flattening if cameras is jagged, otherwise
                keeping dimensions. If False, we flatten at the end. If True, then we keep the shape of the
                camera_indices and coords tensors (if we can).
            disable_distortion: If True, disables distortion.
            aabb_box: if not None will calculate nears and fars of the ray according to aabb box intersection

        Returns:
            Rays for the given camera indices and coords.
        """
        # Check the argument types to make sure they're valid and all shaped correctly
        assert isinstance(camera_indices, (torch.Tensor, int)), "camera_indices must be a tensor or int"
        assert coords is None or isinstance(coords, torch.Tensor), "coords must be a tensor or None"
        assert camera_opt_to_camera is None or isinstance(camera_opt_to_camera, torch.Tensor)
        assert distortion_params_delta is None or isinstance(distortion_params_delta, torch.Tensor)
        if isinstance(camera_indices, torch.Tensor) and isinstance(coords, torch.Tensor):
            num_rays_shape = camera_indices.shape[:-1]
            errormsg = "Batch dims of inputs must match when inputs are all tensors"
            # assert coords.shape[:-1] == num_rays_shape, errormsg  # assumption broken with deblur sampling
            assert camera_opt_to_camera is None or camera_opt_to_camera.shape[:-2] == num_rays_shape, errormsg
            assert distortion_params_delta is None or distortion_params_delta.shape[:-1] == num_rays_shape, errormsg

        # If zero dimensional, we need to unsqueeze to get a batch dimension and then squeeze later
        if not self.shape:
            cameras = self.reshape((1,))
            assert torch.all(
                torch.tensor(camera_indices == 0) if isinstance(camera_indices, int) else camera_indices == 0
            ), "Can only index into single camera with no batch dimensions if index is zero"
        else:
            cameras = self

        # If the camera indices are an int, then we need to make sure that the camera batch is 1D
        if isinstance(camera_indices, int):
            assert (
                len(cameras.shape) == 1
            ), "camera_indices must be a tensor if cameras are batched with more than 1 batch dimension"
            camera_indices = torch.tensor([camera_indices], device=cameras.device)

        assert camera_indices.shape[-1] == len(
            cameras.shape
        ), "camera_indices must have shape (num_rays:..., num_cameras_batch_dims)"

        # If keep_shape is True, then we need to make sure that the camera indices in question
        # are all the same height and width and can actually be batched while maintaining the image
        # shape
        if keep_shape is True:
            assert torch.all(cameras.height[camera_indices] == cameras.height[camera_indices[0]]) and torch.all(
                cameras.width[camera_indices] == cameras.width[camera_indices[0]]
            ), "Can only keep shape if all cameras have the same height and width"

        # If the cameras don't all have same height / width, if coords is not none, we will need to generate
        # a flat list of coords for each camera and then concatenate otherwise our rays will be jagged.
        # Camera indices, camera_opt, and distortion will also need to be broadcasted accordingly which is non-trivial
        if cameras.is_jagged and coords is None and (keep_shape is None or keep_shape is False):
            index_dim = camera_indices.shape[-1]
            camera_indices = camera_indices.reshape(-1, index_dim)
            _coords = [cameras.get_image_coords(index=tuple(index)).reshape(-1, 2) for index in camera_indices]
            camera_indices = torch.cat(
                [index.unsqueeze(0).repeat(coords.shape[0], 1) for index, coords in zip(camera_indices, _coords)],
            )
            coords = torch.cat(_coords, dim=0)
            assert coords.shape[0] == camera_indices.shape[0]
            # Need to get the coords of each indexed camera and flatten all coordinate maps and concatenate them

        # The case where we aren't jagged && keep_shape (since otherwise coords is already set) and coords
        # is None. In this case we append (h, w) to the num_rays dimensions for all tensors. In this case,
        # each image in camera_indices has to have the same shape since otherwise we would have error'd when
        # we checked keep_shape is valid or we aren't jagged.
        if coords is None:
            index_dim = camera_indices.shape[-1]
            index = camera_indices.reshape(-1, index_dim)[0]
            coords = cameras.get_image_coords(index=tuple(index))  # (h, w, 2)
            coords = coords.reshape(coords.shape[:2] + (1,) * len(camera_indices.shape[:-1]) + (2,))  # (h, w, 1..., 2)
            coords = coords.expand(coords.shape[:2] + camera_indices.shape[:-1] + (2,))  # (h, w, num_rays, 2)
            camera_opt_to_camera = (  # (h, w, num_rays, 3, 4) or None
                camera_opt_to_camera.broadcast_to(coords.shape[:-1] + (3, 4))
                if camera_opt_to_camera is not None
                else None
            )
            distortion_params_delta = (  # (h, w, num_rays, 6) or None
                distortion_params_delta.broadcast_to(coords.shape[:-1] + (6,))
                if distortion_params_delta is not None
                else None
            )

        # If camera indices was an int or coords was none, we need to broadcast our indices along batch dims
        try:
            camera_indices = camera_indices.broadcast_to(coords.shape[:-1] + (len(cameras.shape),)).to(torch.long)
        except:
            # NOTE: deblur Raygenerator breaks the above, but the camera indicies is correct
            pass

        # Checking our tensors have been standardized
        assert isinstance(coords, torch.Tensor) and isinstance(camera_indices, torch.Tensor)
        assert camera_indices.shape[-1] == len(cameras.shape)
        assert camera_opt_to_camera is None or camera_opt_to_camera.shape[:-2] == coords.shape[:-1]
        assert distortion_params_delta is None or distortion_params_delta.shape[:-1] == coords.shape[:-1]

        # This will do the actual work of generating the rays now that we have standardized the inputs
        # raybundle.shape == (num_rays) when done

        raybundle = cameras._generate_rays_from_coords(
            camera_indices, coords, camera_opt_to_camera, distortion_params_delta, disable_distortion=disable_distortion
        )

        # If we have mandated that we don't keep the shape, then we flatten
        if keep_shape is False:
            raybundle = raybundle.flatten()

        if aabb_box:
            with torch.no_grad():
                tensor_aabb = Parameter(aabb_box.aabb.flatten(), requires_grad=False)

                rays_o = raybundle.origins.contiguous()
                rays_d = raybundle.directions.contiguous()

                tensor_aabb = tensor_aabb.to(rays_o.device)
                shape = rays_o.shape

                rays_o = rays_o.reshape((-1, 3))
                rays_d = rays_d.reshape((-1, 3))

                t_min, t_max = nerfstudio.utils.math.intersect_aabb(rays_o, rays_d, tensor_aabb)

                t_min = t_min.reshape([shape[0], shape[1], 1])
                t_max = t_max.reshape([shape[0], shape[1], 1])

                raybundle.nears = t_min
                raybundle.fars = t_max

        # TODO: We should have to squeeze the last dimension here if we started with zero batch dims, but never have to,
        # so there might be a rogue squeeze happening somewhere, and this may cause some unintended behaviour
        # that we haven't caught yet with tests
        return raybundle

    
    def _generate_rays_from_coords(
        self,
        camera_indices: Int[Tensor, "*num_rays num_cameras_batch_dims"],
        coords: Float[Tensor, "*num_rays 2"],
        camera_opt_to_camera: Optional[Float[Tensor, "*num_rays 3 4"]] = None,
        distortion_params_delta: Optional[Float[Tensor, "*num_rays 6"]] = None,
        disable_distortion: bool = False,
    ) -> RayBundle:
        """Generates rays for the given camera indices and coords where self isn't jagged

        This is a fairly complex function, so let's break this down slowly.

        Shapes involved:
            - num_rays: This is your output raybundle shape. It dictates the number and shape of the rays generated
            - num_cameras_batch_dims: This is the number of dimensions of our camera

        Args:
            camera_indices: Camera indices of the flattened cameras object to generate rays for.
                The shape of this is such that indexing into camera_indices["num_rays":...] will return the
                index into each batch dimension of the camera in order to get the correct camera specified by
                "num_rays".

                Example:
                    >>> cameras = Cameras(...)
                    >>> cameras.shape
                        (2, 3, 4)

                    >>> camera_indices = torch.tensor([0, 0, 0]) # We need an axis of length 3 since cameras.ndim == 3
                    >>> camera_indices.shape
                        (3,)
                    >>> coords = torch.tensor([1,1])
                    >>> coords.shape
                        (2,)
                    >>> out_rays = cameras.generate_rays(camera_indices=camera_indices, coords = coords)
                        # This will generate a RayBundle with a single ray for the
                        # camera at cameras[0,0,0] at image coordinates (1,1), so out_rays.shape == ()
                    >>> out_rays.shape
                        ()

                    >>> camera_indices = torch.tensor([[0,0,0]])
                    >>> camera_indices.shape
                        (1, 3)
                    >>> coords = torch.tensor([[1,1]])
                    >>> coords.shape
                        (1, 2)
                    >>> out_rays = cameras.generate_rays(camera_indices=camera_indices, coords = coords)
                        # This will generate a RayBundle with a single ray for the
                        # camera at cameras[0,0,0] at point (1,1), so out_rays.shape == (1,)
                        # since we added an extra dimension in front of camera_indices
                    >>> out_rays.shape
                        (1,)

                If you want more examples, check tests/cameras/test_cameras and the function check_generate_rays_shape

                The bottom line is that for camera_indices: (num_rays:..., num_cameras_batch_dims), num_rays is the
                output shape and if you index into the output RayBundle with some indices [i:...], if you index into
                camera_indices with camera_indices[i:...] as well, you will get a 1D tensor containing the batch
                indices into the original cameras object corresponding to that ray (ie: you will get the camera
                from our batched cameras corresponding to the ray at RayBundle[i:...]).

            coords: Coordinates of the pixels to generate rays for. If None, the full image will be rendered, meaning
                height and width get prepended to the num_rays dimensions. Indexing into coords with [i:...] will
                get you the image coordinates [x, y] of that specific ray located at output RayBundle[i:...].

            camera_opt_to_camera: Optional transform for the camera to world matrices.
                In terms of shape, it follows the same rules as coords, but indexing into it with [i:...] gets you
                the 2D camera to world transform matrix for the camera optimization at RayBundle[i:...].

            distortion_params_delta: Optional delta for the distortion parameters.
                In terms of shape, it follows the same rules as coords, but indexing into it with [i:...] gets you
                the 1D tensor with the 6 distortion parameters for the camera optimization at RayBundle[i:...].

            disable_distortion: If True, disables distortion.

        Returns:
            Rays for the given camera indices and coords. RayBundle.shape == num_rays
        """
        # Make sure we're on the right devices
        camera_indices = camera_indices.to(self.device)
        coords = coords.to(self.device)

        # Checking to make sure everything is of the right shape and type
        num_rays_shape = coords.shape[:-1] #camera_indices.shape[:-1]
        # assert camera_indices.shape == num_rays_shape + (self.ndim,)  # NOTE: DEBLUR BREAKS THIS ASSUMPTION
        # assert coords.shape == num_rays_shape + (2,)                  # NOTE: DEBLUR BREAKS THIS ASSUMPTION
        assert coords.shape[-1] == 2
        assert camera_opt_to_camera is None or camera_opt_to_camera.shape == num_rays_shape + (3, 4)
        assert distortion_params_delta is None or distortion_params_delta.shape == num_rays_shape + (6,)

        # Here, we've broken our indices down along the num_cameras_batch_dims dimension allowing us to index by all
        # of our output rays at each dimension of our cameras object
        # true_indices = [camera_indices[..., i] for i in range(camera_indices.shape[-1])]
        c2w, true_indices = self.get_c2w_fn(camera_indices)
            
        assert c2w.shape == num_rays_shape + (3, 4)

        # Get all our focal lengths, principal points and make sure they are the right shapes
        y = coords[..., 0]  # (num_rays,) get rid of the last dimension
        x = coords[..., 1]  # (num_rays,) get rid of the last dimension
        # fx, fy = self.fx[true_indices].squeeze(-1), self.fy[true_indices].squeeze(-1)  # (num_rays,)
        # cx, cy = self.cx[true_indices].squeeze(-1), self.cy[true_indices].squeeze(-1)  # (num_rays,)

        # NOTE: Assume there is only one camera; doing this because true_indices shape is incorrect when making deblur samples
        make_intrxs = lambda x : torch.full(y.shape, x, device=y.device)
        fx, fy = make_intrxs(self.fx[0].item()), make_intrxs(self.fy[0].item())
        cx, cy = make_intrxs(self.cx[0].item()), make_intrxs(self.cy[0].item())

        assert (
            y.shape == num_rays_shape
            and x.shape == num_rays_shape
            and fx.shape == num_rays_shape
            and fy.shape == num_rays_shape
            and cx.shape == num_rays_shape
            and cy.shape == num_rays_shape
        ), (
            str(num_rays_shape)
            + str(y.shape)
            + str(x.shape)
            + str(fx.shape)
            + str(fy.shape)
            + str(cx.shape)
            + str(cy.shape)
        )

        # Get our image coordinates and image coordinates offset by 1 (offsets used for dx, dy calculations)
        # Also make sure the shapes are correct
        coord = torch.stack([(x - cx) / fx, -(y - cy) / fy], -1)  # (num_rays, 2)
        coord_x_offset = torch.stack([(x - cx + 1) / fx, -(y - cy) / fy], -1)  # (num_rays, 2)
        coord_y_offset = torch.stack([(x - cx) / fx, -(y - cy + 1) / fy], -1)  # (num_rays, 2)
        assert (
            coord.shape == num_rays_shape + (2,)
            and coord_x_offset.shape == num_rays_shape + (2,)
            and coord_y_offset.shape == num_rays_shape + (2,)
        )

        # Stack image coordinates and image coordinates offset by 1, check shapes too
        coord_stack = torch.stack([coord, coord_x_offset, coord_y_offset], dim=0)  # (3, num_rays, 2)
        assert coord_stack.shape == (3,) + num_rays_shape + (2,)

        # Undistorts our images according to our distortion parameters
        if not disable_distortion:
            distortion_params = None
            if self.distortion_params is not None:
                distortion_params = self.distortion_params[true_indices]
                if distortion_params_delta is not None:
                    distortion_params = distortion_params + distortion_params_delta
            elif distortion_params_delta is not None:
                distortion_params = distortion_params_delta

            # Do not apply distortion for equirectangular images
            if distortion_params is not None:
                mask = (self.camera_type[true_indices] != CameraType.EQUIRECTANGULAR.value).squeeze(-1)  # (num_rays)
                coord_mask = torch.stack([mask, mask, mask], dim=0)
                if mask.any() and (distortion_params != 0).any():
                    coord_stack[coord_mask, :] = camera_utils.radial_and_tangential_undistort(
                        coord_stack[coord_mask, :].reshape(3, -1, 2),
                        distortion_params[mask, :],
                    ).reshape(-1, 2)

        # Make sure after we have undistorted our images, the shapes are still correct
        assert coord_stack.shape == (3,) + num_rays_shape + (2,)

        # Gets our directions for all our rays in camera coordinates and checks shapes at the end
        # Here, directions_stack is of shape (3, num_rays, 3)
        # directions_stack[0] is the direction for ray in camera coordinates
        # directions_stack[1] is the direction for ray in camera coordinates offset by 1 in x
        # directions_stack[2] is the direction for ray in camera coordinates offset by 1 in y
        cam_types = torch.unique(self.camera_type, sorted=False)
        directions_stack = torch.empty((3,) + num_rays_shape + (3,), device=self.device)


        def _compute_rays_for_omnidirectional_stereo(
            eye: Literal["left", "right"]
        ) -> Tuple[Float[Tensor, "num_rays_shape 3"], Float[Tensor, "3 num_rays_shape 3"]]:
            """Compute the rays for an omnidirectional stereo camera

            Args:
                eye: Which eye to compute rays for.

            Returns:
                A tuple containing the origins and the directions of the rays.
            """
            # Directions calculated similarly to equirectangular
            ods_cam_type = (
                CameraType.OMNIDIRECTIONALSTEREO_R.value if eye == "right" else CameraType.OMNIDIRECTIONALSTEREO_L.value
            )
            mask = (self.camera_type[true_indices] == ods_cam_type).squeeze(-1)
            mask = torch.stack([mask, mask, mask], dim=0)
            theta = -torch.pi * coord_stack[..., 0]
            phi = torch.pi * (0.5 - coord_stack[..., 1])

            directions_stack[..., 0][mask] = torch.masked_select(-torch.sin(theta) * torch.sin(phi), mask).float()
            directions_stack[..., 1][mask] = torch.masked_select(torch.cos(phi), mask).float()
            directions_stack[..., 2][mask] = torch.masked_select(-torch.cos(theta) * torch.sin(phi), mask).float()

            vr_ipd = 0.064  # IPD in meters (note: scale of NeRF must be true to life and can be adjusted with the Blender add-on)
            isRightEye = 1 if eye == "right" else -1

            # find ODS camera position
            c2w = self.get_c2w_fn(camera_indices)

            assert c2w.shape == num_rays_shape + (3, 4)
            transposedC2W = c2w[0][0].t()
            ods_cam_position = transposedC2W[3].repeat(c2w.shape[1], 1)

            rotation = c2w[..., :3, :3]

            ods_theta = -torch.pi * ((x - cx) / fx)[0]

            # local axes of ODS camera
            ods_x_axis = torch.tensor([1, 0, 0], device=c2w.device)
            ods_z_axis = torch.tensor([0, 0, -1], device=c2w.device)

            # circle of ODS ray origins
            ods_origins_circle = (
                ods_cam_position
                + isRightEye * (vr_ipd / 2.0) * (ods_x_axis.repeat(c2w.shape[1], 1)) * (torch.cos(ods_theta))[:, None]
                + isRightEye * (vr_ipd / 2.0) * (ods_z_axis.repeat(c2w.shape[1], 1)) * (torch.sin(ods_theta))[:, None]
            )

            # rotate origins to match the camera rotation
            for i in range(ods_origins_circle.shape[0]):
                ods_origins_circle[i] = rotation[0][0] @ ods_origins_circle[i] + ods_cam_position[0]
            ods_origins_circle = ods_origins_circle.unsqueeze(0).repeat(c2w.shape[0], 1, 1)

            # assign final camera origins
            c2w[..., :3, 3] = ods_origins_circle

            return ods_origins_circle, directions_stack

        for cam in cam_types:
            if CameraType.PERSPECTIVE.value in cam_types:
                mask = (self.camera_type[true_indices] == CameraType.PERSPECTIVE.value).squeeze(-1)  # (num_rays)
                mask = torch.stack([mask, mask, mask], dim=0)
                directions_stack[..., 0][mask] = torch.masked_select(coord_stack[..., 0], mask).float()
                directions_stack[..., 1][mask] = torch.masked_select(coord_stack[..., 1], mask).float()
                directions_stack[..., 2][mask] = -1.0

            elif CameraType.FISHEYE.value in cam_types:
                mask = (self.camera_type[true_indices] == CameraType.FISHEYE.value).squeeze(-1)  # (num_rays)
                mask = torch.stack([mask, mask, mask], dim=0)

                theta = torch.sqrt(torch.sum(coord_stack**2, dim=-1))
                theta = torch.clip(theta, 0.0, math.pi)

                sin_theta = torch.sin(theta)

                directions_stack[..., 0][mask] = torch.masked_select(
                    coord_stack[..., 0] * sin_theta / theta, mask
                ).float()
                directions_stack[..., 1][mask] = torch.masked_select(
                    coord_stack[..., 1] * sin_theta / theta, mask
                ).float()
                directions_stack[..., 2][mask] = -torch.masked_select(torch.cos(theta), mask).float()

            elif CameraType.EQUIRECTANGULAR.value in cam_types:
                mask = (self.camera_type[true_indices] == CameraType.EQUIRECTANGULAR.value).squeeze(-1)  # (num_rays)
                mask = torch.stack([mask, mask, mask], dim=0)

                # For equirect, fx = fy = height = width/2
                # Then coord[..., 0] goes from -1 to 1 and coord[..., 1] goes from -1/2 to 1/2
                theta = -torch.pi * coord_stack[..., 0]  # minus sign for right-handed
                phi = torch.pi * (0.5 - coord_stack[..., 1])
                # use spherical in local camera coordinates (+y up, x=0 and z<0 is theta=0)
                directions_stack[..., 0][mask] = torch.masked_select(-torch.sin(theta) * torch.sin(phi), mask).float()
                directions_stack[..., 1][mask] = torch.masked_select(torch.cos(phi), mask).float()
                directions_stack[..., 2][mask] = torch.masked_select(-torch.cos(theta) * torch.sin(phi), mask).float()

            elif CameraType.OMNIDIRECTIONALSTEREO_L.value in cam_types:
                ods_origins_circle, directions_stack = _compute_rays_for_omnidirectional_stereo("left")
                # assign final camera origins
                c2w[..., :3, 3] = ods_origins_circle

            elif CameraType.OMNIDIRECTIONALSTEREO_R.value in cam_types:
                ods_origins_circle, directions_stack = _compute_rays_for_omnidirectional_stereo("right")
                # assign final camera origins
                c2w[..., :3, 3] = ods_origins_circle
            else:
                raise ValueError(f"Camera type {cam} not supported.")

        assert directions_stack.shape == (3,) + num_rays_shape + (3,)

        if camera_opt_to_camera is not None:
            c2w = pose_utils.multiply(c2w, camera_opt_to_camera)
        rotation = c2w[..., :3, :3]  # (..., 3, 3)
        assert rotation.shape == num_rays_shape + (3, 3)

        directions_stack = torch.sum(
            directions_stack[..., None, :] * rotation, dim=-1
        )  # (..., 1, 3) * (..., 3, 3) -> (..., 3)
        directions_stack, directions_norm = camera_utils.normalize_with_norm(directions_stack, -1)
        assert directions_stack.shape == (3,) + num_rays_shape + (3,)

        origins = c2w[..., :3, 3]  # (..., 3)
        assert origins.shape == num_rays_shape + (3,)

        directions = directions_stack[0]
        assert directions.shape == num_rays_shape + (3,)

        # norms of the vector going between adjacent coords, giving us dx and dy per output ray
        dx = torch.sqrt(torch.sum((directions - directions_stack[1]) ** 2, dim=-1))  # ("num_rays":...,)
        dy = torch.sqrt(torch.sum((directions - directions_stack[2]) ** 2, dim=-1))  # ("num_rays":...,)
        assert dx.shape == num_rays_shape and dy.shape == num_rays_shape

        pixel_area = (dx * dy)[..., None]  # ("num_rays":..., 1)
        assert pixel_area.shape == num_rays_shape + (1,)

        times = self.times[camera_indices, 0] if self.times is not None else None

        metadata = (
            self._apply_fn_to_dict(self.metadata, lambda x: x[true_indices]) if self.metadata is not None else None
        )
        if metadata is not None:
            metadata["directions_norm"] = directions_norm[0].detach()
        else:
            metadata = {"directions_norm": directions_norm[0].detach()}

        times = self.times[camera_indices, 0] if self.times is not None else None

        times, camera_indices = fix_datashape(times, origins), fix_datashape(camera_indices, origins)
        metadata = {k: fix_datashape(v, origins) for k,v in metadata.items()}

        return RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=pixel_area,
            camera_indices=camera_indices,
            times=times,
            metadata=metadata,
        )
