# LSENeRF
implementation of the paper LSENeRF

[paper](https://arxiv.org/pdf/2409.06104) | [webpage](https://ubc-vision.github.io/LSENeRF/) | [data](https://github.com/ubc-vision/LSENeRF-data-formatter)

# Setup
1. Create an environment with python=3.8 or from environment.yml
2. Install with the below:
```
python -m pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install -e .
```
3. Reinstall tiny-cuda-nn torch extension from [source](https://github.com/NVlabs/tiny-cuda-nn) with float32 enabled following [here](https://github.com/NVlabs/tiny-cuda-nn/issues/51#issuecomment-1054565404).


# Training
Refer to [data repo](https://github.com/ubc-vision/LSENeRF-data-formatter) to format either a EVIMOv2 or LSENeRF scene. To train a model, update the `--data` in the training script and run them:
```bash
# to train a LSENeRF scene
bash scripts/train_lse_data.sh

# to train a EVIMOv2 scene
bash scripts/train_evimo.sh
```
You can choose which method to run by change the configurations at the top of the *train_evimo.sh* and *train_lse_data.sh*.

To see all available parameters do:
```bash
ns-train lsenerf -h
```

# Evaluation
These script run camera optimization before evaluation. Please update the experiment path before running. The example path in each script should give a sense of what to put down. To evaluate non-emb methods, do:
```
bash scripts/eval.sh
```

To evaluate a embedding method do, update the experiment path and do:
```
bash scripts/emb_eval.sh
```
