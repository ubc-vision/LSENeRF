
### INPUT HERE: replace $EXP_PATH with the experiment you want to evaluate with
EXP_PATH="outputs_tcnn/Dragon Max/LSENeRF/2024-08-24_021027"

echo $EXP_PATH

#### Add the data path if there is issues
# python train.py lsenerf --data formatted_data/depth_var_1_lr_000000 \
python train.py lsenerf --max-num-iterations $((6000 + 10)) \
                        --steps-per-eval-image 10000 \
                        --load-dir  "$EXP_PATH/nerfstudio_models" \
                        --load-config "$EXP_PATH/config.yml" \
                        --is_eval True \
                        --emb_eval_mode zero \
                        --pipeline.datamanager.col-dataparser.image-type clear \
                        --pipeline.datamanager.col-dataparser.quality "" \
                        --pipeline.model.eval-num-rays-per-chunk 128 

# NOTE:
# Global emb always return the same embedding regardless of --emb_eval_mode