
### INPUT HERE: replace $EXP_PATH with the experiment you want to evaluate with
EXP_PATH="outputs_tcnn/Dragon Max/BADNeRF_emb/2024-08-24_021025"

# add the datapath as in eval.sh if not working
# learn a test embedding using training images
# NOTE: --do_pretrain True  == learn a testing emb
python train.py lsenerf \
  --max-num-iterations 3010 \
  --steps-per-eval-image 10000 \
  --load-dir "$EXP_PATH/nerfstudio_models" \
  --load-config "$EXP_PATH/config.yml" \
  --is_eval True \
  --emb_eval_mode param \
  --do_pretrain True \
  --pipeline.datamanager.col-dataparser.image-type clear \
  --pipeline.datamanager.col-dataparser.quality "" \
  --pipeline.model.eval-num-rays-per-chunk 2048 


PARAM_EXP_PATH="${EXP_PATH}_eval_param"
LAST_DIR=$(ls -1 "$PARAM_EXP_PATH" | grep -v '_eval_param' | sort | tail -n 1)
FULL_DIR="$PARAM_EXP_PATH/$LAST_DIR"


# run camera opt and eval with the learned test emb
python train.py lsenerf \
  --max-num-iterations $((6000 + 10)) \
  --steps-per-eval-image 10000 \
  --is_eval True \
  --emb_eval_mode param \
  --pipeline.datamanager.col-dataparser.image-type clear \
  --pipeline.datamanager.col-dataparser.quality "" \
  --load-dir "$FULL_DIR/nerfstudio_models" \
  --load-config "$FULL_DIR/config.yml" \
  --pipeline.model.eval-num-rays-per-chunk 2048 
