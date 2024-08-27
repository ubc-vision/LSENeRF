# fraction of batch to use for rgb, 1 for using rgb only
rgb_frac=1.0
evs_loss_fn=log_loss

## rgb mapping
use_map=False
mapping_method=identity

## evs mapping; map_mode=None for turning event mapping off
map_mode=None 
evs_mapping_method=None

# experiment name
exp_name=BADNeRF

# global_emb is no embedding
emb_type=global_emb