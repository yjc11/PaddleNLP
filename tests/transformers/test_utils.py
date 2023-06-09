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

import json
import unittest

from paddlenlp.transformers import (
    AlbertForTokenClassification,
    AlbertModel,
    BertModel,
    utils,
)
from paddlenlp.transformers.bert.modeling import BertForTokenClassification


class TestUtils(unittest.TestCase):
    """Unittest for paddlenlp.transformers.utils.py module"""

    def test_find_transformer_model_type(self):
        """test for `find_transformer_model_type`"""
        self.assertEqual(utils.find_transformer_model_type(AlbertModel), "albert")
        self.assertEqual(utils.find_transformer_model_type(AlbertForTokenClassification), "albert")
        self.assertEqual(utils.find_transformer_model_type(BertModel), "bert")
        self.assertEqual(utils.find_transformer_model_type(BertForTokenClassification), "bert")


def check_json_file_has_correct_format(file_path):
    with open(file_path, "r") as f:
        try:
            json.load(f)
        except Exception as e:
            raise Exception(f"{e}: the json file should be a valid json")
