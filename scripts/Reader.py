import os
import json
import logging
import sys
import math
import cv2

import numpy as np

from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy

sys.path.append('/home/youjiachen/PaddleNLP/paddlenlp')
sys.path.append(Path(__file__) / '..')
from utils.doc_match_label import match_label_v1
from preprocess import rotate_box


class DataProcess:
    def __init__(self, ocr_result, output_path):
        self.ocr_result = Path(ocr_result)
        self.output_path = Path(output_path)

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
        refined_list = []
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
                    summary_token_num = 3
                if max_seq_len <= len(prompt) + summary_token_num:
                    raise ValueError(
                        'The value of max_seq_len is too small, please set a larger value'
                    )
                max_content_len = max_seq_len - len(prompt) - summary_token_num

                if len(content) <= max_content_len:
                    refined_list.append(json_line)
                    # yield json_line
                else:
                    result_list = json_line['result_list']

                    accumulate = 0
                    while True:
                        cur_result_list = []
                        for result in result_list:
                            if result['end'] - result['start'] > max_content_len:
                                logging.warning(
                                    'result["end"] - result ["start"] exceeds max_content_len, which will result in no valid instance being returned'
                                )
                            if (
                                result['start'] + 1 <= max_content_len < result['end']
                                and result['end'] - result['start'] <= max_content_len
                            ):
                                # 训练时确保字段的start和end不会被截断，预估如何保证？
                                max_content_len = result['start']
                                break

                        cur_content = content[:max_content_len]
                        res_content = content[max_content_len:]
                        if boxes is not None and image is not None:
                            cur_boxes = boxes[:max_content_len]
                            res_boxes = boxes[max_content_len:]

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

        return refined_list, json_lines

    def match_label(self, label_file):
        with open(label_file, 'r', encoding='utf-8') as f:
            raw_example = json.loads(f.read())

        tmp_dict = defaultdict(dict)
        c = 0
        empty = 0
        for line in tqdm(raw_example):
            for e in line['annotations']:
                if not len(e):  # 无标签则跳过
                    continue
                pagename = e['page_name']
                if pagename not in tmp_dict:
                    with open(self.ocr_result / f'{pagename}.json', 'r') as f:
                        ocr_results = json.load(f)
                        ocr_bboxes = ocr_results['bboxes']
                        ocr_texts = ocr_results['texts']
                        image_size = ocr_results['image_size']
                        rotate_angle = ocr_results['rotate_angle']

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

        # convert format to reader format
        res = [j for i in tmp_dict.values() for j in i.values()]

        print('匹配上的标注：', c)
        print('未匹配上的标注：', empty)

        with open(self.output_path / 'reader_input.txt', 'w') as f:
            for i in res:
                f.write(json.dumps(i, ensure_ascii=False) + "\n")

        return res

    @staticmethod
    def compute_angle(cos, sin):
        angle = math.atan2(sin, cos) * 180 / math.pi
        return angle

    @staticmethod
    def refine_box(r, w, h):
        """
        box = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        """
        angle = r['rotation']
        R = cv2.getRotationMatrix2D(angle=angle, center=(w / 2, h / 2), scale=1)
        r['box'] = np.array(r['box']).reshape(-1, 2)
        box_hom = np.hstack((r['box'], np.ones((4, 1))))
        box_rotated = np.dot(R, box_hom.T).T[:, :2]
        refined_box = box_rotated.tolist()

        return refined_box


if __name__ == "__main__":
    ocr_file_path = '/home/youjiachen/workspace/longtext_ie/datasets/contract_v1.0/dataelem_ocr_res_rotateupright_true'
    label_file = '/home/youjiachen/workspace/longtext_ie/datasets/contract_v1.1/processed_labels_5_7.json'
    output_path = (
        '/home/youjiachen/workspace/longtext_ie/datasets/contract_v1.1/preprocess_ds'
    )
    data_processer = DataProcess(ocr_file_path, output_path)
    res = data_processer.match_label(label_file)
    # print(len(res))
    # with open(
    #     '/home/youjiachen/workspace/longtext_ie/datasets/reader_input.json', 'w'
    # ) as f:
    #     json.dump(res, f, indent=4, ensure_ascii=False)

    # with open('/home/youjiachen/workspace/longtext_ie/datasets/ceshi.txt', 'w') as f:
    #     for i in res:
    #         f.write(json.dumps(i, ensure_ascii=False) + "\n")

    # output_path = '/home/youjiachen/workspace/longtext_ie/datasets/contract_v1.1'
    # file_path = '/home/youjiachen/workspace/longtext_ie/datasets/ceshi.txt'
    # refined_list, json_lines = data_processer.reader(file_path)

    # with open(output_path + '/reader_output.json', 'w') as f:
    #     json.dump(json_lines, f, indent=4, ensure_ascii=False)
    # print(refined_list)
    # print(json_lines)
