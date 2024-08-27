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

#!/usr/bin/env python
"""Train a radiance field with nerfstudio.
For real captures, we recommend using the [bright_yellow]nerfacto[/bright_yellow] model.

Nerfstudio allows for customizing your training and eval configs from the CLI in a powerful way, but there are some
things to understand.

The most demonstrative and helpful example of the CLI structure is the difference in output between the following
commands:

    ns-train -h
    ns-train nerfacto -h nerfstudio-data
    ns-train nerfacto nerfstudio-data -h

In each of these examples, the -h applies to the previous subcommand (ns-train, nerfacto, and nerfstudio-data).

In the first example, we get the help menu for the ns-train script.
In the second example, we get the help menu for the nerfacto model.
In the third example, we get the help menu for the nerfstudio-data dataparser.

With our scripts, your arguments will apply to the preceding subcommand in your command, and thus where you put your
arguments matters! Any optional arguments you discover from running

    ns-train nerfacto -h nerfstudio-data

need to come directly after the nerfacto subcommand, since these optional arguments only belong to the nerfacto
subcommand:

    ns-train nerfacto {nerfacto optional args} nerfstudio-data
"""

from __future__ import annotations


import random
import socket
import traceback
from datetime import timedelta
from typing import Any, Callable, Literal, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tyro
import yaml
import os
import os.path as osp
from pathlib import Path

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.configs.method_configs import AnnotatedBaseConfigUnion
from nerfstudio.utils import comms, profiler
from nerfstudio.utils.rich_utils import CONSOLE

from lse_nerf.lse_trainer import LSETranerConfig as TrainerConfig
from lse_nerf.lse_trainer import LSETrainer
from lse_nerf.utils import gbconfig

DEFAULT_TIMEOUT = timedelta(minutes=30)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore


def _find_free_port() -> str:
    """Finds a free port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_loop(local_rank: int, world_size: int, config: TrainerConfig, global_rank: int = 0):
    """Main training function that sets up and runs the trainer per process

    Args:
        local_rank: current rank of process
        world_size: total number of gpus available
        config: config file specifying training regimen
    """
    _set_random_seed(config.machine.seed + global_rank)
    trainer:LSETrainer = config.setup(local_rank=local_rank, world_size=world_size)
    trainer.setup()

    if config.do_pretrain:
        trainer.setup_pretrain()

    trainer.train()


def _distributed_worker(
    local_rank: int,
    main_func: Callable,
    world_size: int,
    num_devices_per_machine: int,
    machine_rank: int,
    dist_url: str,
    config: TrainerConfig,
    timeout: timedelta = DEFAULT_TIMEOUT,
    device_type: Literal["cpu", "cuda", "mps"] = "cuda",
) -> Any:
    """Spawned distributed worker that handles the initialization of process group and handles the
       training process on multiple processes.

    Args:
        local_rank: Current rank of process.
        main_func: Function that will be called by the distributed workers.
        world_size: Total number of gpus available.
        num_devices_per_machine: Number of GPUs per machine.
        machine_rank: Rank of this machine.
        dist_url: URL to connect to for distributed jobs, including protocol
            E.g., "tcp://127.0.0.1:8686".
            It can be set to "auto" to automatically select a free port on localhost.
        config: TrainerConfig specifying training regimen.
        timeout: Timeout of the distributed workers.

    Raises:
        e: Exception in initializing the process group

    Returns:
        Any: TODO: determine the return type
    """
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_devices_per_machine + local_rank

    dist.init_process_group(
        backend="nccl" if device_type == "cuda" else "gloo",
        init_method=dist_url,
        world_size=world_size,
        rank=global_rank,
        timeout=timeout,
    )
    assert comms.LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_devices_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_devices_per_machine, (i + 1) * num_devices_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comms.LOCAL_PROCESS_GROUP = pg

    assert num_devices_per_machine <= torch.cuda.device_count()
    output = main_func(local_rank, world_size, config, global_rank)
    comms.synchronize()
    dist.destroy_process_group()
    return output


def launch(
    main_func: Callable,
    num_devices_per_machine: int,
    num_machines: int = 1,
    machine_rank: int = 0,
    dist_url: str = "auto",
    config: Optional[TrainerConfig] = None,
    timeout: timedelta = DEFAULT_TIMEOUT,
    device_type: Literal["cpu", "cuda", "mps"] = "cuda",
) -> None:
    """Function that spawns multiple processes to call on main_func

    Args:
        main_func (Callable): function that will be called by the distributed workers
        num_devices_per_machine (int): number of GPUs per machine
        num_machines (int, optional): total number of machines
        machine_rank (int, optional): rank of this machine.
        dist_url (str, optional): url to connect to for distributed jobs.
        config (TrainerConfig, optional): config file specifying training regimen.
        timeout (timedelta, optional): timeout of the distributed workers.
        device_type: type of device to use for training.
    """
    assert config is not None
    world_size = num_machines * num_devices_per_machine
    if world_size == 0:
        raise ValueError("world_size cannot be 0")
    elif world_size == 1:
        # uses one process
        try:
            main_func(local_rank=0, world_size=world_size, config=config)
        except KeyboardInterrupt:
            # print the stack trace
            CONSOLE.print(traceback.format_exc())
        finally:
            profiler.flush_profiler(config.logging)
    elif world_size > 1:
        # Using multiple gpus with multiple processes.
        if dist_url == "auto":
            assert num_machines == 1, "dist_url=auto is not supported for multi-machine jobs."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"
        if num_machines > 1 and dist_url.startswith("file://"):
            CONSOLE.log("file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://")

        process_context = mp.spawn(
            _distributed_worker,
            nprocs=num_devices_per_machine,
            join=False,
            args=(main_func, world_size, num_devices_per_machine, machine_rank, dist_url, config, timeout, device_type),
        )
        # process_context won't be None because join=False, so it's okay to assert this
        # for Pylance reasons
        assert process_context is not None
        try:
            process_context.join()
        except KeyboardInterrupt:
            for i, process in enumerate(process_context.processes):
                if process.is_alive():
                    CONSOLE.log(f"Terminating process {i}...")
                    process.terminate()
                process.join()
                CONSOLE.log(f"Process {i} finished.")
        finally:
            profiler.flush_profiler(config.logging)


def modify_config(config):
    if config.load_config:
        ori_config = config
        CONSOLE.log(f"Loading pre-set config from: {config.load_config}")
        config = yaml.load(config.load_config.read_text(), Loader=yaml.Loader)
        config.load_dir = ori_config.load_dir
        config.max_num_iterations = ori_config.max_num_iterations
        config.steps_per_eval_image = ori_config.steps_per_eval_image
        config.steps_per_eval_all_images = ori_config.steps_per_eval_all_images
        config.steps_per_save = ori_config.steps_per_save
        config.timestamp = ori_config.timestamp
        config.emb_eval_mode = ori_config.emb_eval_mode
        config.pipeline.datamanager.col_dataparser.image_type = ori_config.pipeline.datamanager.col_dataparser.image_type
        config.pipeline.datamanager.col_dataparser.quality = ori_config.pipeline.datamanager.col_dataparser.quality
        config.output_dir = ori_config.output_dir if ori_config.output_dir != Path("outputs") else config.output_dir

        config.is_eval = ori_config.is_eval
        config.do_pretrain = ori_config.do_pretrain
        config.is_render = ori_config.is_render


        if not ori_config.data is None:
            config.pipeline.datamanager.data = ori_config.data


        gbconfig.IS_EVAL = config.is_eval
        gbconfig.DO_PRETRAIN = config.do_pretrain
        gbconfig.IS_RENDER = config.is_render

        if gbconfig.IS_EVAL:
            config.method_name = osp.join(config.method_name, f"{osp.basename(osp.dirname(ori_config.load_dir))}_eval_{config.emb_eval_mode}")
            load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(config.load_dir))[-1]
            config.steps_per_eval_all_images = load_step + config.max_num_iterations - 5
            config.pipeline.model.embed_config.eval_mode = config.emb_eval_mode

            if not config.do_pretrain:
                config.pipeline.model.rgb_loss_type = "linspace"
                config.pipeline.datamanager.rgb_loss_mode = "mse"
            else:
                # not eval if pretrain
                config.steps_per_eval_all_images = load_step + config.max_num_iterations + 1000
        else:
            config.method_name = osp.join(config.method_name, f"{osp.basename(osp.dirname(ori_config.load_dir))}_camopt")

        config.pipeline.datamanager.col_cam_optimizer.mode = "SO3xR3"        # always optimize camera if eval

        if config.do_pretrain and config.pipeline.model.embed_config.eval_mode == "param":
            config.pipeline.model.rgb_loss_type = "deblur"
            config.pipeline.datamanager.rgb_loss_mode = "deblur"
        elif config.do_pretrain and (not config.pipeline.model.embed_config.eval_mode == "param"):
            assert 0, "Pretrain only make sense with eval_mode = param"
        else:    
            config.pipeline.datamanager.col_cam_optimizer.optim_type = "ns"
            
        config.pipeline.datamanager.col_cam_optimizer.scheme = "active"
        
        config.pipeline.model.eval_num_rays_per_chunk = ori_config.pipeline.model.eval_num_rays_per_chunk

    if config.pipeline.model.rgb_loss_type == "deblur":
        config.pipeline.datamanager.rgb_loss_mode = "deblur"
        config.pipeline.datamanager.col_cam_optimizer.optim_type = "spline"
    
    
    # gbconfig.IS_EVAL = config.is_eval
    # gbconfig.DO_PRETRAIN = config.do_pretrain
    # gbconfig.IS_RENDER = config.is_render

    return config



def main(config: TrainerConfig) -> None:
    """Main function."""

    config.set_timestamp()
    if config.data:
        CONSOLE.log("Using --data alias for --data.pipeline.datamanager.data")
        config.pipeline.datamanager.data = config.data

    if config.prompt:
        CONSOLE.log("Using --prompt alias for --data.pipeline.model.prompt")
        config.pipeline.model.prompt = config.prompt

    config = modify_config(config)

    # print and save config
    config.print_to_terminal()
    config.save_config()

    launch(
        main_func=train_loop,
        num_devices_per_machine=config.machine.num_devices,
        device_type=config.machine.device_type,
        num_machines=config.machine.num_machines,
        machine_rank=config.machine.machine_rank,
        dist_url=config.machine.dist_url,
        config=config,
    )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    )


if __name__ == "__main__":
    entrypoint()
