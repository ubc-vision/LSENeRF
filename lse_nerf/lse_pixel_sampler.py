from typing import Dict
from nerfstudio.data.pixel_samplers import PixelSampler
import torch

class EvPixelSampler(PixelSampler):
    def collate_image_dataset_batch(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape

        if "mask" in batch:
            indices = self.sample_method(
                num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
            )
        else:
            indices = self.sample_method(num_rays_per_batch, num_images, image_height, image_width, device=device)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()
        collated_batch = {
            key: value[c, y, x] for key, value in batch.items() if (key != "image_idx" and key != "e_thresh" and key != "appearance_id") and value is not None
        }

        assert collated_batch["image"].shape[0] == num_rays_per_batch

        if batch.get("e_thresh") is not None:
            collated_batch["e_thresh"] = batch["e_thresh"][c]

        # Needed to correct the random indices to their actual camera idx locations.
        indices = indices.to(batch["image_idx"].device)
        collated_batch["appearance_id"] = batch["appearance_id"][c]
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch
