from nerfstudio.model_components.renderers import RGBRenderer
from torch import Tensor

class LinearRenderer(RGBRenderer):

    def forward(self, rgb, weights, ray_indices, num_rays) -> Tensor:
        tmp = self.training
        self.training = True
        out = super().forward(rgb, weights, ray_indices, num_rays)
        self.training = tmp
        return out