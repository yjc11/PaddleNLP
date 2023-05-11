#!/bin/bash

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export CUDA_VISIBLE_DEVICES=5

QUESTION=$1

if [ ! -d output ]; then
    mkdir output
fi
if [ ! -d log ]; then
    mkdir log
fi

# python3 change_to_rerank.py ${QUESTION}

python3 -u ./src/train_ce.py \
                   --use_cuda true \
                   --verbose true \
                   --do_train false \
                   --do_val true\
                   --do_test false\
                   --batch_size 128 \
                   --init_checkpoint "./output/step_11601" \
                   --dev_set "/home/youjiachen/workspace/longtext_ie/datasets/contract_v1.1/preprocess_ds/val.tsv" \
                   --max_seq_len 512 \
                   --for_cn true \
                   --vocab_path "config/ernie_base_1.0_CN/vocab.txt" \
                   --ernie_config_path "config/ernie_base_1.0_CN/ernie_config.json"
                   1>>log/train.log 2>&1

# bash run_test_longtext.sh 买方