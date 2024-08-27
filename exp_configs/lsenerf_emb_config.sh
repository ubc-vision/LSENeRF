# fraction of batch to use for rgb, 1 for using rgb only
rgb_frac=0.66
evs_loss_fn=log_loss

## rgb mapping
use_map=True
mapping_method=identity

## evs mapping
# map_mode, one of [None, co_map], co_map to turn on event mapping
map_mode=co_map
evs_mapping_method=powpow # powpow is learnable gamma map


# experiment name
exp_name=LSENeRF_emb

# evs_emb is using embedding
emb_type=evs_emb