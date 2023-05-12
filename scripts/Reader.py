import os
import json
import logging
import math
import cv2
import random
import sys
import copy

import networkx as nx
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from pprint import pprint
from pathlib import Path
from itertools import chain
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

sys.path.append('/home/youjiachen/PaddleNLP/paddlenlp')
sys.path.append(Path(__file__) / '..')
from utils.doc_match_label import match_label_v1
from preprocess import rotate_box


class DataProcess:
    def __init__(self, ocr_result, output_path, cls_file_path):
        self.ocr_result = Path(ocr_result)
        self.output_path = Path(output_path)
        self.all_ocr_pages = sorted(set(i.stem for i in self.ocr_result.glob('[!.]*')))
        self.all_pdf_names = sorted({i.split('_page_')[0] for i in self.all_ocr_pages})
        self.reader_output = None
        with open(cls_file_path, 'r') as f:
            self.cls = json.load(f)
        self.max_prompt_len = max([len(i) for i in self.cls])

    def __len__(self):
        return len(self.all_ocr_pages)

    @staticmethod
    def reader(data_path, max_seq_len=512):
        '''
        read json
        '''
        data_image_path = os.path.splitext(data_path)[0] + '_image.txt'
        if os.path.exists(data_image_path):
            with open(data_image_path, 'r', encoding='utf-8') as f:
                all_images = json.load(f)
        else:
            all_images = None
        json_lines = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                json_line = json.loads(line)
                page = json_line['pagename']
                content = json_line['content'].strip()
                prompt = json_line['prompt']
                boxes = json_line.get('bbox', None)
                image = json_line.get('image', None)
                if image is not None and all_images is not None:
                    image = all_images[image]
                    json_line['image'] = image
                # image_file = json_line.get('image_file', '')
                # Model Input is aslike: [CLS] prompt [SEP] [SEP] text [SEP] for UIE-X
                if boxes is not None and image is not None:
                    summary_token_num = 4
                else:
                    summary_token_num = 3  #
                if max_seq_len <= len(prompt) + summary_token_num:
                    raise ValueError(
                        'The value of max_seq_len is too small, please set a larger value'
                    )
                max_content_len = max_seq_len - len(prompt) - summary_token_num

                if len(content) <= max_content_len:
                    json_lines.append(json_line)
                    # yield json_line
                else:
                    result_list = json_line['result_list']

                    accumulate = 0
                    while True:
                        cur_result_list = []
                        for result in result_list:
                            if (
                                result['end'] - result['start'] > max_content_len
                            ):  # value超过max_content_len
                                logging.warning(
                                    'result["end"] - result ["start"] exceeds max_content_len, which will result in no valid instance being returned'
                                )
                            if (
                                result['start'] + 1
                                <= max_content_len
                                < result['end']  # value在max_content_len范围内或者部分在范围内
                                and result['end'] - result['start'] <= max_content_len
                            ):
                                max_content_len = result['start']
                                break

                        cur_content = content[:max_content_len]
                        res_content = content[max_content_len:]
                        # if boxes is not None and image is not None:
                        #     cur_boxes = boxes[:max_content_len]
                        #     res_boxes = boxes[max_content_len:]

                        while True:
                            # 如果prompt有多个start和end时，默认从小到大
                            if len(result_list) == 0:
                                break
                            elif result_list[0]['end'] <= max_content_len:
                                if result_list[0]['end'] > 0:
                                    cur_result = result_list.pop(0)
                                    cur_result_list.append(cur_result)
                                else:
                                    cur_result_list = [result for result in result_list]
                                    break
                            else:
                                break

                        if boxes is not None and image is not None:
                            json_line = {
                                'content': cur_content,
                                'result_list': cur_result_list,
                                'prompt': prompt,
                                'bbox': cur_boxes,
                                'image': image,
                                'pagename': page,
                            }
                        else:
                            json_line = {
                                'content': cur_content,
                                'result_list': cur_result_list,
                                'prompt': prompt,
                                'pagename': page,
                            }
                        json_lines.append(json_line)

                        for result in result_list:
                            if result['end'] <= 0:
                                break
                            result['start'] -= max_content_len
                            result['end'] -= max_content_len
                        accumulate += max_content_len
                        max_content_len = max_seq_len - len(prompt) - summary_token_num
                        if len(res_content) == 0:
                            break
                        elif len(res_content) < max_content_len:
                            if boxes is not None and image is not None:
                                json_line = {
                                    'content': res_content,
                                    'result_list': result_list,
                                    'prompt': prompt,
                                    'bbox': res_boxes,
                                    'image': image,
                                    'pagename': page,
                                }
                            else:
                                json_line = {
                                    'content': res_content,
                                    'result_list': result_list,
                                    'prompt': prompt,
                                    'pagename': page,
                                }

                            json_lines.append(json_line)
                            break
                        else:
                            content = res_content
                            # boxes = res_boxes
            return json_lines

    def match_label(self, label_file, merge=None):
        with open(label_file, 'r', encoding='utf-8') as f:
            raw_example = json.loads(f.read())

        tmp_dict = defaultdict(dict)
        c = 0
        empty = 0
        gt_pages = set()
        for line in tqdm(raw_example):
            # cur_pages = list()
            for e in line['annotations']:
                if not len(e):  # 无标签则跳过
                    continue

                pagename = e['page_name']
                with open(self.ocr_result / f'{pagename}.json', 'r') as f:
                    ocr_results = json.load(f)
                    ocr_bboxes = ocr_results['bboxes']
                    ocr_texts = ocr_results['texts']
                    image_size = ocr_results['image_size']
                    rotate_angle = ocr_results['rotate_angle']

                if pagename not in tmp_dict:
                    # cur_pages.append(pagename)
                    gt_pages.add(pagename)

                    # 初始化当前page的结果
                    tmp_dict[pagename][e['label'][0]] = {
                        'content': ''.join(ocr_texts),
                        'result_list': [],
                        'prompt': e['label'][0],
                        'pagename': pagename,
                        'image': None,
                        'bbox': None,
                    }

                elif e['label'][0] not in tmp_dict[pagename]:
                    tmp_dict[pagename][e['label'][0]] = {
                        'content': ''.join(ocr_texts),
                        'result_list': [],
                        'prompt': e['label'][0],
                        'pagename': pagename,
                        'image': None,
                        'bbox': None,
                    }

                # match by gt and ocr rotate box and text
                gt_bbox = e['box']
                _gt_box = rotate_box(np.array(gt_bbox), image_size, rotate_angle)
                gt_text = e['text'][0]
                offsets = match_label_v1(
                    deepcopy(_gt_box), deepcopy(gt_text), ocr_bboxes, ocr_texts
                )

                # 写入匹配结果
                if len(offsets) > 0:
                    c += 1
                    tmp_dict[pagename][e['label'][0]]['result_list'].append(
                        {
                            'id': e['id'],
                            'text': gt_text,
                            'start': offsets[0][0],
                            'end': offsets[0][1],
                        }
                    )
                else:
                    empty += 1

            # todo:此处为具有关系的字段start和end合并代码，待完成
            # if merge:
            #     # 1.得到当前pdf的关系组
            #     relation_sets = self._get_relation_set(line['relations'])

            #     # 2.逐页根据关系合并gt
            #     for page in cur_pages:
            #         raw_result_list = tmp_dict[page][e['label'][0]]['result_list']
            #         id_2_gt = {gt['id']: gt for gt in raw_result_list}
            #         cur_id = set(id_2_gt.keys())
            #         for relation_set in relation_sets:
            #             if cur_id.intersection(set(relation_set)):
            #                 pass

        # 添加无gt页面内容
        final_dict = self._add_negative_examples(tmp_dict, gt_pages)

        # convert format to reader format
        res = [j for i in final_dict.values() for j in i.values()]

        print('匹配上的标注：', c)
        print('未匹配上的标注：', empty)

        with open(self.output_path / 'reader_input.txt', 'w') as f:
            for i in res:
                f.write(json.dumps(i, ensure_ascii=False) + "\n")

        return res

    def cut_and_save_data(self, save_path=None):
        data_generator = self.reader_v2(f'{self.output_path}/reader_input.txt')

        if not save_path:
            self.reader_output = str(self.output_path / 'reader_output.txt')
            with open(self.reader_output, 'w') as f:
                for i in tqdm(data_generator):
                    f.write(json.dumps(i, ensure_ascii=False) + "\n")
        else:
            self.reader_output = str(Path(save_path) / 'reader_output.txt')
            with open(self.reader_output, 'w') as f:
                for i in tqdm(data_generator):
                    f.write(json.dumps(i, ensure_ascii=False) + "\n")

    def _add_negative_examples(self, tmp_dict, gt_pages):
        neg_pages = set(self.all_ocr_pages) - gt_pages
        for neg_page in neg_pages:
            with open(os.path.join(self.ocr_result, f'{neg_page}.json'), 'r') as f:
                ocr_results = json.load(f)
                ocr_texts = ocr_results['texts']

                tmp_dict[neg_page]['无gt'] = {
                    'content': ''.join(ocr_texts),
                    'result_list': [],
                    'prompt': '无' * self.max_prompt_len,
                    'pagename': neg_page,
                    'image': None,
                    'bbox': None,
                }

        return tmp_dict

    def reader_v2(self, data_path, max_seq_len=512):
        # todo: 优化点1:单页内容的gt合并处理; 优化点2:根据bbox切分
        json_lines = list()
        c = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            for ids, line in enumerate(f):
                json_line = json.loads(line)
                page = json_line['pagename']
                content = json_line['content'].strip()
                prompt = json_line['prompt']

                if not len(content):
                    continue

                summary_token_num = 3  # [CLS] + [SEP] + [SEP]
                max_content_len = max_seq_len - self.max_prompt_len - summary_token_num

                arr = np.arange(0, len(content))
                content_idx_sets = [
                    arr[i : i + max_content_len]
                    for i in range(0, len(arr), max_content_len)
                ]  #
                c += len(content_idx_sets)

                if not len(json_line['result_list']):  # 对无gt的page切片存储
                    for idc, interval in enumerate(content_idx_sets):
                        cur_content = content[interval[0] : interval[-1] + 1]
                        _json_line = {
                            'content': cur_content,
                            'result_list': [],
                            'prompt': prompt,
                            'pagename': page,
                            'interval_id': idc,
                        }
                        json_lines.append(_json_line)

                else:  # 对有gt的page进行切片存储
                    prompt_se_sets = [
                        np.arange(i['start'], i['end'])
                        for i in json_line['result_list']
                    ]
                    results = list()
                    for ids, content_interval in enumerate(content_idx_sets):
                        for prompt_interval in prompt_se_sets:
                            intersection = sorted(
                                set(prompt_interval) & set(content_interval)
                            )
                            if not len(intersection):
                                continue
                            else:
                                results.append(
                                    (ids, content_interval, intersection)
                                )  # 片段id，内容区间，        gt区间

                    # 对无gt的碎片进行存储
                    gt_fragment = set(_[0] for _ in results)
                    total_fragment = set(_ for _ in range(len(content_idx_sets)))
                    no_gt_fragment = total_fragment - gt_fragment
                    for frag_id in no_gt_fragment:
                        no_gt_interval = content_idx_sets[frag_id]
                        s, e = no_gt_interval[0], no_gt_interval[-1]
                        cur_content = content[s : e + 1]
                        _json_line = {
                            'content': cur_content,
                            'result_list': [],
                            'prompt': prompt,
                            'pagename': page,
                            'interval_id': frag_id,
                        }
                        json_lines.append(_json_line)

                    cur_gt_id = 0
                    _map = dict()
                    tmp_json_lines = list()
                    for res in results:
                        cur_content = content[res[1][0] : res[1][-1] + 1]
                        start = res[2][0] - res[0] * (max_content_len - 1)
                        end = res[2][-1] + 1 - res[0] * (max_content_len - 1)
                        cur_result_list = [{'start': int(start), 'end': int(end)}]
                        if res[0] in _map:
                            json_id = _map[res[0]]
                            tmp_json_lines[json_id]['result_list'].extend(
                                cur_result_list
                            )

                        else:
                            _map[res[0]] = cur_gt_id
                            _json_line = {
                                'content': cur_content,
                                'result_list': cur_result_list,
                                'prompt': prompt,
                                'pagename': page,
                                'interval_id': res[0],
                            }
                            tmp_json_lines.append(_json_line)
                            cur_gt_id += 1

                    json_lines.extend(tmp_json_lines)

        print('理论上的段数：', c)
        print('实际的段数', len(json_lines))
        return json_lines

    def create_ds(self):
        if self.reader_output is None:
            self.reader_output = self.output_path / 'reader_output.txt'

        train_ds, val_ds = train_test_split(
            self.all_pdf_names,
            train_size=0.8,
            test_size=0.2,
            shuffle=True,
            random_state=42,
        )

        train_p = list()
        train_n = list()
        val_p = list()
        val_n = list()
        train_pos_dict = defaultdict(lambda: defaultdict(list))
        train_neg_dict = defaultdict(list)
        tag_stats_dict = self.tag_statistics()
        with open(self.reader_output, 'r') as f:
            for i in f:
                data = json.loads(i)
                cur_pagename = data['pagename']
                cur_pdf = cur_pagename.split('_page_')[0]
                cur_frag = cur_pagename + '~$~' + str(data['interval_id'])
                cur_content = data['content']

                if not len(cur_content):
                    continue

                # 训练候选集构造逻辑
                cur_pdf_gt_page_frags = tag_stats_dict[cur_pdf]
                if cur_pdf in train_ds:
                    # 如果page不存在于字段统计字典中，判断是否为纯负例
                    if cur_frag not in cur_pdf_gt_page_frags:
                        for tag in self.cls:
                            train_neg_dict[tag].append(f'{tag}\t\t{cur_content}\t0\n')
                    # 如果page存在于字段统计字典中,则根据差集构建负例
                    elif cur_frag in cur_pdf_gt_page_frags:
                        # 构造负例
                        cur_page_gt_tag = cur_pdf_gt_page_frags[cur_frag]
                        redundants = list(set(cur_page_gt_tag) ^ set(self.cls))
                        for tag in redundants:
                            train_neg_dict[tag].append(f'{tag}\t\t{cur_content}\t0\n')

                        # 构造正例
                        for tag in cur_page_gt_tag:
                            train_pos_dict[cur_pdf][tag].append(
                                f'{tag}\t\t{cur_content}\t1\n'
                            )
                            train_p.append(f'{tag}\t\t{cur_content}\t1\n')

                # 验证集构造逻辑
                elif cur_pdf in val_ds:
                    # 若不是gt页，则构造所有字段的负例
                    if cur_frag not in cur_pdf_gt_page_frags:
                        for tag in self.cls:
                            val_n.append(f'{tag}\t\t{cur_content}\t0\n')
                    # 若是gt页，则对gt字段构造正例，对所有非gt字段构造负例
                    elif cur_frag in cur_pdf_gt_page_frags:
                        cur_page_gt_tag = cur_pdf_gt_page_frags[cur_frag]
                        for tag in cur_pdf_gt_page_frags[cur_frag]:
                            val_p.append(f'{tag}\t\t{cur_content}\t1\n')

                        redundants = list(set(cur_page_gt_tag) ^ set(self.cls))
                        for tag in redundants:
                            val_n.append(f'{tag}\t\t{cur_content}\t0\n')

            # 训练集构造逻辑(按字段 1:1 构造负例)
            # 1.统计正例的各字段数量
            stats_pos_num = dict()
            for tag2data in train_pos_dict.values():
                for tag, data in tag2data.items():
                    stats_pos_num[tag] = stats_pos_num.get(tag, 0) + len(data)

            # 2. 根据字段和正例数采样负例
            for tag, num in stats_pos_num.items():
                sample_neg = random.sample(train_neg_dict[tag], k=num)
                train_n.extend(sample_neg)

        train_tsv = train_p + train_n
        val_tsv = val_p + val_n
        random.shuffle(train_tsv)
        random.shuffle(val_tsv)
        with open(self.output_path / 'train.tsv', 'w') as f:
            f.writelines(train_tsv)
        with open(self.output_path / 'val.tsv', 'w') as f:
            f.writelines(val_tsv)

        print('---PDF---')
        print('train:', len(train_ds))
        print('val:', len(val_ds))

        print('---page_frag_ds---')
        print('train_p:', len(train_p))
        print('train_n:', len(train_n))
        print('val_p:', len(val_p))
        print('val_n:', len(val_n))

        print('---pos_tag_nums---')
        pprint(stats_pos_num)

    def tag_statistics(self):
        stats_dict = defaultdict(lambda: defaultdict(list))
        with open(self.output_path / 'reader_output.txt', 'r') as f:
            for line in f:
                data = json.loads(line)
                pagename = data['pagename']
                pdf = pagename.split('_page_')[0]
                frag = pagename + '~$~' + str(data['interval_id'])
                gt_list = data['result_list']

                if gt_list:
                    prompt = data['prompt']
                    stats_dict[pdf][frag].append(prompt)
            # pprint(stats_dict)

        return stats_dict

    @staticmethod
    def compute_angle(cos, sin):
        angle = math.atan2(sin, cos) * 180 / math.pi
        return angle

    @staticmethod
    def refine_box(r, img_w, img_h):
        """
        box = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        """
        angle = r['rotation']
        R = cv2.getRotationMatrix2D(angle=angle, center=(img_w / 2, img_h / 2), scale=1)
        r['box'] = np.array(r['box']).reshape(-1, 2)
        box_hom = np.hstack((r['box'], np.ones((4, 1))))
        box_rotated = np.dot(R, box_hom.T).T[:, :2]
        refined_box = box_rotated.tolist()

        return refined_box

    @staticmethod
    def _get_relation_set(relations):
        G = nx.DiGraph()
        for relation in relations:
            from_id, to_id = relation['from_id'], relation['to_id']
            G.add_edge(from_id, to_id)
        r_set = []
        for component in nx.connected_components(G.to_undirected()):
            # 对每个连通分量进行拓扑排序
            sorted_nodes = list(nx.topological_sort(G.subgraph(component)))
            r_set.append(sorted_nodes)
        return r_set


if __name__ == "__main__":
    ocr_file_path = '/home/youjiachen/workspace/longtext_ie/datasets/contract_v1.0/dataelem_ocr_res_rotateupright_true'
    output_path = (
        '/home/youjiachen/workspace/longtext_ie/datasets/contract_v1.1/preprocess_ds'
    )
    cls_path = '/home/youjiachen/workspace/longtext_ie/datasets/contract_v1.1/cls.json'
    label_file = '/home/youjiachen/workspace/longtext_ie/datasets/contract_v1.1/processed_labels_5_7.json'
    data_processer = DataProcess(ocr_file_path, output_path, cls_path)
    data_processer.match_label(label_file)  # 匹配标注
    data_processer.cut_and_save_data()  # 512 切分后保存
    data_processer.create_ds()  # 构造train val
    data_processer.tag_statistics()  # 统计字段
