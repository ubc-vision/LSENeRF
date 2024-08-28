from nerfstudio.engine.trainer import TrainerConfig, Trainer
from nerfstudio.utils import profiler
from nerfstudio.engine.optimizers import Optimizers

from typing import Type, Literal
from pathlib import Path

import torch
from nerfstudio.utils.rich_utils import CONSOLE

from lse_nerf.lse_pipeline import LSENeRFPipeline
from lse_nerf.utils import gbconfig
from dataclasses import dataclass, field
import os


@dataclass
class LSETranerConfig(TrainerConfig):
    _target: Type = field(default_factory=lambda: LSETrainer)
    is_eval: bool = False
    emb_eval_mode: Literal["zero", "mean", "param"] = "zero"
    do_pretrain: bool = False
    is_render: bool = False

    def get_base_dir(self) -> Path:
        dataset_type = ["train", "spiral", "panel"]
        path = super().get_base_dir()
        if self.is_eval:
            to_add = f"_{self.pipeline.datamanager.col_dataparser.image_type}" \
                     if self.pipeline.datamanager.col_dataparser.image_type in dataset_type else ""
            path = Path(f"{self.output_dir}/{self.experiment_name}/{self.method_name}/{self.timestamp}" + to_add)

        return path

class LSETrainer(Trainer):
    config: LSETranerConfig
    pipeline : LSENeRFPipeline

    
    def setup_pretrain(self):
        # 1) INIT EMB_PARAM
        # 2) UPDATE OPTIMIZER
        self.pipeline._model.init_test_params()
        self.optimizers = self.setup_optimizers(emb_eval_mode="opt")


    def setup_optimizers(self, emb_eval_mode=None) -> Optimizers:
        """Helper to set up the optimizers

        Returns:
            The optimizers object given the trainer config.
        """
        # NOTE: make sure the camera update weights are never loaded
        optimizer_config = self.config.optimizers.copy()

        # NOTE: eval mode will only return emb_weights for field
        param_groups = self.pipeline.get_param_groups()
        emb_eval_mode = self.config.emb_eval_mode if emb_eval_mode is None else emb_eval_mode

        if gbconfig.IS_EVAL and emb_eval_mode != "opt":
            del optimizer_config["fields"], param_groups["fields"]
        
        if gbconfig.IS_RENDER:
            param_groups = {}

        return Optimizers(optimizer_config, param_groups)
    
    def _modify_states_for_eval(self, loaded_state:dict):
        """
        remove learned cameras from state
        """
        if gbconfig.DO_PRETRAIN:
            # shape is consistent when DO_PRETRAIN, so don't delete the camera
            return

        pipeline_dic = loaded_state["pipeline"]
        for k in list(pipeline_dic.keys()):
            if "camera_optimizer" in k:
                del pipeline_dic[k]

        if not (loaded_state["optimizers"].get("camera_opt") is None):
            del loaded_state["optimizers"]["camera_opt"]

    
    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir = self.config.load_dir
        load_checkpoint = self.config.load_checkpoint
        if load_dir is not None:
            load_step = self.config.load_step
            if load_step is None:
                print("Loading latest Nerfstudio checkpoint from load_dir...")
                # NOTE: this is specific to the checkpoint name format
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")
            self._start_step = loaded_state["step"] + 1

            if gbconfig.IS_EVAL:
                self._modify_states_for_eval(loaded_state)

            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            # self.optimizers.load_optimizers(loaded_state["optimizers"])
            # self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_path}")
        elif load_checkpoint is not None:
            assert load_checkpoint.exists(), f"Checkpoint {load_checkpoint} does not exist"
            loaded_state = torch.load(load_checkpoint, map_location="cpu")
            self._start_step = loaded_state["step"] + 1

            if gbconfig.IS_EVAL:
                self._modify_states_for_eval(loaded_state)

            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_checkpoint}")
        else:
            CONSOLE.print("No Nerfstudio checkpoint to load, so training from scratch.")
