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

if [ $# != 3 ]
then
    echo "Usage: bash run_distribute_train_gpu.sh [CFG_PATH] [SAVE_PATH] [DEVICE_NUM]"
exit 1
fi

CFG_PATH=$1
SAVE_PATH=$2
DEVICE_NUM=$3

if [ -d $SAVE_PATH ];
then
    rm -rf $SAVE_PATH
fi
mkdir -p $SAVE_PATH

cp $CFG_PATH $SAVE_PATH

mpirun --allow-run-as-root -n $DEVICE_NUM --map-by socket:pe=3 --output-filename log_output --merge-stderr-to-stdout \
python train.py \
  --is_distributed=1 \
  --device_target=GPU \
  --cfg_path=$CFG_PATH \
  --save_path=$SAVE_PATH > $SAVE_PATH/log.txt 2>&1 &
