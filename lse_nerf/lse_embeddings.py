from dataclasses import dataclass, field
from nerfstudio.configs.base_config import InstantiateConfig
from typing import Any, Type, Dict, Literal
import os

from torch import nn
import torch
from torch.nn.functional import cosine_similarity

from nerfstudio.field_components.embedding import Embedding
from nerfstudio.cameras.rays import RaySamples

from lse_nerf.lse_cameras import HardCamType
from lse_nerf.data_components import find_closest_idxs, find_closest_idxs_exclu
from lse_nerf.utils import gbconfig


# treat this as a abstract class
class EvsFrameEmbedding(Embedding):
    def __init__(self, config, num_imgs, num_dims, n_bins) -> None:
        self.config = config

        super().__init__(num_imgs, self.config.emb_dim)
        self.eval_emb_fn_dict = {"zero": self.get_zero_emb,
                                 "first": self.get_first_emb,
                                 "opt": self.get_opt_emb,
                                 "mean": self.get_mean_emb,
                                 "param": self.get_test_param_emb}
        self.test_emb = None
        self.get_eval_emb_fn = self.eval_emb_fn_dict[self.config.eval_mode]


        if gbconfig.IS_EVAL and not gbconfig.DO_PRETRAIN:
            self.init_test_params()

    
    ## MUST IMPLEMENT
    def forward(self, x: RaySamples, call_from_test=False):
        if gbconfig.IS_EVAL and not call_from_test:
            return self.get_test_emb(x)

        idxs = x.metadata["appearance_id"] 
        return super().forward(idxs)

    ## MUST IMPLEMENT
    def get_test_emb(self, x:RaySamples):
        return self.get_eval_emb_fn(x)
    
    def get_first_emb(self, x:RaySamples):
        idxs = x.metadata["appearance_id"]
        idxs[:] = 3
        return Embedding.forward(self, idxs)

    def get_opt_emb(self, x:RaySamples):
        idxs = x.metadata["appearance_id"]
        return Embedding.forward(self, idxs)
    
    def get_mean_emb(self, x:RaySamples):
        return torch.ones((len(x), self.out_dim), device = x.frustums.directions.device) * self.mean(dim=0)
    
    def get_zero_emb(self, x:RaySamples):
        embedded_appearance = torch.zeros(
                        (*x.frustums.directions.shape[:-1], self.out_dim), device=x.frustums.directions.device
                    )
        return embedded_appearance

    def get_test_param_emb(self, x:RaySamples):
        assert self.test_emb is not None, "for deblur pretrain test only! need to init test_emb!"
        
        idxs = x.metadata["appearance_id"]
        idxs[:] = 0
        return self.test_emb(idxs)

    def init_test_params(self):
        if self.test_emb is not None or (self.embedding.weight.shape[0] <= 1):
            # NOTE: if test_emb already initialized or is global embedding, skip
            return
        self.test_emb = nn.Embedding(1, self.out_dim).to(self.embedding.weight.device)
        self.test_emb.weight = nn.Parameter(self.embedding(torch.tensor([21]).to(self.embedding.weight.device)))

    # MUST IMPLEMENT
    def get_emb_dim(self):
        return self.out_dim



class GlobalEmbedding(EvsFrameEmbedding):
    def __init__(self, config, num_imgs, num_dims, n_bins=None) -> None:
        super().__init__(config, 1, num_dims, n_bins)
    
    def forward(self, x:RaySamples):
        idxs = x.camera_indices * 0
        return Embedding.forward(self, idxs)
    
    def get_test_emb(self, x:RaySamples):
        return self.forward(x)




EMBEDDING_TYPE_DICT = {"global_emb": GlobalEmbedding,
                       "evs_emb": EvsFrameEmbedding,}


@dataclass
class LSEEmbeddingConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: GlobalEmbedding)
    embedding_type: str = "global_emb"
    n_bins: int = 1
    metadata:  str = "dummy"
    skip_base: str = "evs_emb"

    emb_dim:int = 32
    """dimension of embedding; PLEASE CONFIGURE IT HERE"""

    eval_mode: Literal["zero", "first", "opt", "mean"] = "zero"

    def setup(self, **kwargs):
        _target = EMBEDDING_TYPE_DICT[self.embedding_type.lower()]
        # return super().setup(n_bins=self.n_bins, **kwargs)
        return _target(self, n_bins=self.n_bins, **kwargs)

    def set_binsize(self, size):
        self.n_bins=size