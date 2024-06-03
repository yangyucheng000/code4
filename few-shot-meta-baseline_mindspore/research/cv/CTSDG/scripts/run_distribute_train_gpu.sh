#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

if [ $# != 7 ]
then
    echo "Usage: bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [CFG_PATH] [SAVE_PATH] [VGG_PRETRAIN] [IMAGES_PATH] [MASKS_PATH] [ANNO_PATH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}


DEVICE_NUM=$1
CFG_PATH=$(get_real_path $2)
SAVE_PATH=$(get_real_path $3)
VGG_PRETRAIN=$(get_real_path $4)
IMAGES_PATH=$(get_real_path $5)
MASKS_PATH=$(get_real_path $6)
ANNO_PATH=$(get_real_path $7)

if [ -d "$SAVE_PATH" ];
then
    rm -rf "$SAVE_PATH"
fi
mkdir -p "$SAVE_PATH"

cp "$CFG_PATH" "$SAVE_PATH"

mpirun --allow-run-as-root -n "$DEVICE_NUM" --map-by socket:pe=4 --output-filename "$SAVE_PATH" --merge-stderr-to-stdout \
python train.py \
  --is_distributed=1 \
  --device_target=GPU \
  --gen_lr_train=0.002 \
  --gen_lr_finetune=0.0005 \
  --train_iter=43750 \
  --finetune_iter=18750 \
  --save_checkpoint_steps=1250 \
  --log_frequency_step=125 \
  --config_path="$CFG_PATH" \
  --pretrained_vgg="$VGG_PRETRAIN" \
  --data_root="$IMAGES_PATH" \
  --train_masks_root="$MASKS_PATH" \
  --anno_path="$ANNO_PATH" \
  --save_path="$SAVE_PATH" > "$SAVE_PATH"/log.txt 2>&1 &
