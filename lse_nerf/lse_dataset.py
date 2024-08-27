from __future__ import annotations
from typing import Dict
from PIL import Image

import numpy as np
import numpy.typing as npt
import numpy as np

import torch
from torch import Tensor
from torch import Tensor

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset 
from nerfstudio.utils.rich_utils import CONSOLE


class ColorDataset(InputDataset):
    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1, use_gray: bool = False):
        super().__init__(dataparser_outputs, scale_factor)
        self.appearance_ids = self._dataparser_outputs.appearance_ids
        self.use_gray = use_gray
        if self._dataparser_outputs.msk is not None:
            self.msk = torch.from_numpy(self._dataparser_outputs.msk)
        else:
            self.msk = self._dataparser_outputs.msk
    
    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        pil_image = Image.open(image_filename)
        if self.scale_factor != 1.0:
            width, height = pil_image.size
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
        
        if self.use_gray:
            pil_image = pil_image.convert("L")
            
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def get_data(self, image_idx: int) -> Dict:
        data = super().get_data(image_idx)
        data["appearance_id"] = self.appearance_ids[image_idx]
        if self.msk is not None:
            data["msk"] = self.msk[image_idx]
        return data

class EventFrameDataset(ColorDataset):

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1):
        super().__init__(dataparser_outputs, scale_factor)
        self.e_thresh = torch.tensor([self._dataparser_outputs.e_thresh], dtype=torch.float32)
        self.evs = self._dataparser_outputs.events

        if len(self.evs) > 1000:
            self.evs = self.evs[:-8]
            

        if self._dataparser_outputs.e_thresh == 1 and self.msk is not None:
            self.evs = np.clip(self.evs/255, 0,1)

        self.evs = torch.from_numpy(self.evs)

    def __len__(self):
        return len(self.evs)
    
    def get_data(self, image_idx: int) -> Dict:
        data = super().get_data(image_idx)
        data["e_thresh"] = self.e_thresh
        return data

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        assert 0, "no images in event frames dataset"

    def get_image(self, image_idx: int) -> Tensor:
        ev_img = self.evs[image_idx]
        return (ev_img * self.e_thresh).float()

    ### NOTE: for debug, replace the above and replace self.train
    # def get_image(self, image_idx: int) -> Tensor:
    #     ev_img = self.evs[image_idx] #.squeeze()
    #     h, w = ev_img.shape[:2]

    #     img_r = torch.zeros((h, w, 1))
    #     img_g = torch.zeros((h, w, 1))
    #     img_b = torch.zeros((h, w, 1))
        
    #     neg_cond = ev_img < 0
    #     pos_cond = ev_img > 0
    #     img_r[neg_cond] = 1 #ev_img[neg_cond]*self.e_thresh
    #     img_g[pos_cond] = 1 #ev_img[pos_cond]*self.e_thresh
    #     img_g[pos_cond] = 1
        
    #     return torch.concat([img_r, img_g, img_b], dim=-1)