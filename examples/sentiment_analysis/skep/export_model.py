# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import argparse

import paddle

from paddlenlp.transformers import SkepForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt_dir",
    type=str,
    required=True,
    default="./checkpoint/model_100",
    help="The directory of saved model checkpoint.",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="./static_graph_params",
    help="The path of model parameter in static graph to be saved.",
)
parser.add_argument(
    "--model_name",
    choices=["skep_ernie_1.0_large_ch", "skep_ernie_2.0_large_en"],
    default="skep_ernie_1.0_large_ch",
    help="Select which model to train, defaults to skep_ernie_1.0_large_ch.",
)
args = parser.parse_args()

if __name__ == "__main__":
    # The number of labels should be in accordance with the training dataset.
    label_map = {0: "negative", 1: "positive"}
    model = SkepForSequenceClassification.from_pretrained(args.ckpt_dir, num_labels=len(label_map))
    print("Loaded model from %s" % args.ckpt_dir)

    model.eval()

    # Convert to static graph with specific input description
    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # segment_ids
        ],
    )
    # Save in static graph model.
    paddle.jit.save(model, args.output_path)
    print("Static Model has been saved to: {}".format(args.output_path))
