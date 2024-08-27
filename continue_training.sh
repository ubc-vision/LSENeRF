

path_exist() {
    if [ -e "$1" ]; then
        echo "Path '$1' exists!"
    else
        echo "Path '$1' does not exist."
    fi
}

EXP_DIR=outputs/synth_robo/debug_cont/2023-07-27_180832
TIMESTAMP=$(basename $EXP_DIR)
METHODNAME=$(basename $(dirname $EXP_DIR))

CONFIG_FILE=$EXP_DIR/config.yml
CKPNT_DIR=$EXP_DIR/nerfstudio_models/

path_exist $CONFIG_FILE
path_exist $CKPNT_DIR


python train.py  ednerf --data data/synth_robo \
                        --load-config $CONFIG_FILE \
                        --load-dir $CKPNT_DIR \
                        --timestamp $TIMESTAMP \

