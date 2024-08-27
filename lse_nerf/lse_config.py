from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from lse_nerf.lse_trainer import LSETranerConfig as TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from lse_nerf.lse_datamanager import MultiCamManagerConfig
from lse_nerf.lsenerf import LSENeRFModelConfig
from lse_nerf.lse_pipeline import LSENeRFPipelineConfig
from lse_nerf.lse_parser import ColorDataParserConfig, EventDataParserConfig



lsenerf_method =MethodSpecification(TrainerConfig(
    method_name="lsenerf",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=False,
    pipeline=LSENeRFPipelineConfig(
        datamanager=MultiCamManagerConfig(
            col_dataparser=ColorDataParserConfig(),
            evs_dataparser=EventDataParserConfig(),
            train_num_rays_per_batch=3512,
            eval_num_rays_per_batch=1024,
        ),
        model=LSENeRFModelConfig(eval_num_rays_per_chunk=3512),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
        }
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
    vis="tensorboard",
), description="base lsenerf"
)