mamba activate nerfstudio
which python

MAX_NUM_ITER=30000

# python train.py ednerf  --data /scratch/matthew/projects/e-nerf_synth_datapipeline/adapt_carpet_enerf \
#                         --method-name evs_only_ngp \
#                         --pipeline.datamanager.evs-dataparser.scale-factor 0.4 \
#                         --pipeline.datamanager.col-dataparser.scale-factor 0.4 \
#                         --pipeline.datamanager.rgb_frac 0.0

# python train.py ednerf  --data /ubc/cs/research/kmyi/matthew/projects/DyNeRF/datasets/mean_t_rgb_checker \
#                         --method-name evs_only \
#                         --pipeline.datamanager.evs-dataparser.scale-factor 0.1 \
#                         --pipeline.datamanager.col-dataparser.scale-factor 0.1 \
#                         --pipeline.datamanager.rgb_frac 0.0


# python train.py ednerf  --data /ubc/cs/research/kmyi/matthew/projects/DyNeRF/datasets/adapt_carpet_formatted \
#                         --method-name evs_only_ngp \
#                         --pipeline.datamanager.evs-dataparser.scale-factor 1 \
#                         --pipeline.datamanager.col-dataparser.scale-factor 1 \
#                         --pipeline.datamanager.rgb_frac 0.0
 

# python train.py ednerf  --data data/ShakeCarpet1_formatted \
#                         --method-name evs_only_ngp \
#                         --pipeline.datamanager.evs-dataparser.scale-factor 0.1 \
#                         --pipeline.datamanager.col-dataparser.scale-factor 0.1 \
#                         --pipeline.datamanager.col-dataparser.scene-scale 0.8 \
#                         --pipeline.datamanager.rgb_frac 0.0


python train.py ednerf --data data/synth_robo \
                       --method-name r1_rgb_clear \
                       --pipeline.datamanager.evs-dataparser.scale-factor 0.1 \
                       --pipeline.datamanager.col-dataparser.scale-factor 0.1 \
                       --pipeline.datamanager.col-dataparser.scene-scale 0.8 \
                       --pipeline.datamanager.col-dataparser.image-type linear \
                       --pipeline.datamanager.col-dataparser.quality clear \
                       --steps-per-eval-all-images $MAX_NUM_ITER \
                       --max-num-iterations $MAX_NUM_ITER \
                       --pipeline.datamanager.rgb_frac 1.0

# python train.py ednerf-rgb --data data/ShakeCarpet1_formatted/colcam_set \
#                            --pipeline.datamanager.dataparser.scale-factor 0.15 \
#                            --pipeline.datamanager.dataparser.scene-scale 1
# python train.py ednerf-rgb --data data/synth_robo/colcam_set \
#                            --method-name static_batch\
#                            --pipeline.datamanager.dataparser.scale-factor 0.1 \
#                            --pipeline.datamanager.dataparser.scene-scale 0.8 \



### reference
# ns-train  instant-ngp --data data/ns-robo/ --vis viewer+tensorboard
