#### choose your configurations
source exp_configs/BADNERF_config.sh
# source exp_configs/BADNERF_emb_config.sh
# source exp_configs/lsenerf_config.sh
# source exp_configs/lsenerf_emb_config.sh

seed=96
MAX_NUM_ITER=200010
OUTPUT_DIR=outputs_tcnn

########## DEFAULT CONFIGS ##########
## RGB loss function
rgb_loss_fn=deblur

## event loss weight
evs_w=1.0

## camera optimization on of [off, SO3xR3]
col_cam_opt=SO3xR3
evs_cam_opt=SO3xR3

## one of [ns, spline]; col_cam_opt_type will be overwrite to spline if rgb_loss_fn=deblur
# ns    : default nerfstudio camera-opt, each camera is parameterized
# spline: assumes linear trajectory between 2 cameras
col_cam_opt_type=spline
evs_cam_opt_type=ns

## one of [learned, gt]; gt to use classic rgb to gray; learned for learning based rgb to gray
ev_one=gt

## scene related param
scale_fac=1.0
scene_scale=1.0

########## END DEFAULT CONFIGS ##########


# PLEASE UPDATE THE --data for training
python train.py lsenerf --data "formatted_data/depth_var_1_lr_000000" \
                        --method-name $exp_name \
                        --output-dir $OUTPUT_DIR \
                        --pipeline.datamanager.evs-dataparser.scale-factor $scale_fac \
                        --pipeline.datamanager.col-dataparser.scale-factor $scale_fac \
                        --pipeline.datamanager.evs-dataparser.scene-scale $scene_scale \
                        --pipeline.datamanager.col-dataparser.scene-scale $scene_scale \
                        --pipeline.datamanager.col-cam-optimizer.mode $col_cam_opt \
                        --pipeline.datamanager.evs-cam-optimizer.mode $evs_cam_opt \
                        --pipeline.datamanager.col-cam-optimizer.optim-type $col_cam_opt_type \
                        --pipeline.datamanager.col-cam-optimizer.exp-t 30000 \
                        --pipeline.datamanager.evs-cam-optimizer.optim-type $evs_cam_opt_type \
                        --max-num-iterations $MAX_NUM_ITER \
                        --steps-per-eval-image 20000 \
                        --steps-per-save 10000 \
                        --steps-per-eval-all-images 100000 \
                        --pipeline.datamanager.rgb_frac $rgb_frac \
                        --vis tensorboard \
                        --optimizers.fields.optimizer.lr 1e-2 \
                        --pipeline.model.evs-loss-weight $evs_w \
                        --pipeline.model.event-loss-type $evs_loss_fn \
                        --pipeline.model.rgb-loss-type $rgb_loss_fn \
                        --pipeline.model.ev-one-dim $ev_one \
                        --pipeline.model.use-mapping $use_map \
                        --pipeline.model.mapping-method $mapping_method \
                        --pipeline.model.evs-mapping-method $evs_mapping_method \
                        --pipeline.model.map-mode $map_mode \
                        --pipeline.model.embed-config.embedding-type $emb_type \
                        --machine.seed $seed 