import torch
from nerfstudio.cameras.rays import RayBundle


def find_closest_idxs(ref_tensor: torch.Tensor, srch_tensor: torch.Tensor):
    """
    find the closest element in ref_tensor for every element in srch_tensor
    """

    # Use searchsorted to find the insertion points
    insert_points = torch.searchsorted(ref_tensor, srch_tensor)
    
    # Clamp insert points to be within the array bounds
    insert_points = insert_points.clamp(max=len(ref_tensor) - 1)

    # Calculate distances to the current closest elements
    closest_distances = torch.abs(ref_tensor[insert_points] - srch_tensor)

    # Prepare to check previous elements (if any)
    prev_points = (insert_points - 1).clamp(min=0)
    prev_distances = torch.abs(ref_tensor[prev_points] - srch_tensor)

    # Use a mask to determine if the previous element is closer
    mask = prev_distances < closest_distances

    # Apply mask to determine final indices
    closest_indices = torch.where(mask, prev_points, insert_points)

    return closest_indices


def find_closest_idxs_exclu(ref_tensor: torch.Tensor, srch_tensor: torch.Tensor):
    """
    Find the closest element in ref_tensor for every element in srch_tensor,
    excluding the element itself if present in ref_tensor.
    """

    # Use searchsorted to find the insertion points
    insert_points = torch.searchsorted(ref_tensor, srch_tensor)
    
    # Clamp insert points to be within the array bounds
    insert_points = insert_points.clamp(max=len(ref_tensor) - 1)

    # Calculate distances to the current closest elements
    closest_distances = torch.abs(ref_tensor[insert_points] - srch_tensor)

    # Prepare to check previous elements (if any)
    prev_points = (insert_points - 1).clamp(min=0)
    prev_distances = torch.abs(ref_tensor[prev_points] - srch_tensor)

    # Determine next points (if any)
    next_points = (insert_points + 1).clamp(max=len(ref_tensor) - 1)
    next_distances = torch.abs(ref_tensor[next_points] - srch_tensor)

    # Avoid selecting the same element by setting distance to infinity if indices are the same
    same_element_mask = ref_tensor[insert_points] == srch_tensor
    closest_distances = torch.where(same_element_mask, torch.inf, closest_distances)

    # Compare previous and next distances to find truly closest
    mask_prev = (prev_distances <= closest_distances) & (prev_distances <= next_distances)
    mask_next = (next_distances < closest_distances) & (next_distances < prev_distances)

    # Apply mask to determine final indices
    closest_indices = torch.where(mask_prev, prev_points, insert_points)
    closest_indices = torch.where(mask_next, next_points, closest_indices)

    return closest_indices


class CameraIdxFixer:
    """
    align evs camera idx to rgb cameras
    """

    def __init__(self, rgb_ts) -> None:
        self.rgb_ts = rgb_ts

        if self.rgb_ts is not None:
            self.rgb_ts = rgb_ts.to("cuda")
    
    def __call__(self, ray_bundle: RayBundle) -> torch.Any:
        return self.fix_idxs(ray_bundle)

    def fix_idxs(self, ray_bundle: RayBundle):
        if ray_bundle is None or (self.rgb_ts is None):
            return ray_bundle
        
        corr_camera_idxs = find_closest_idxs(self.rgb_ts, ray_bundle.times)
        ray_bundle.camera_indices = corr_camera_idxs
        return ray_bundle