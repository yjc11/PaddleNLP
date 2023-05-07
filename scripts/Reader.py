import os
import json
from pathlib import Path
from collections import defaultdict

import logging
import sys

sys.path.append('/home/youjiachen/PaddleNLP_baidu/workspace')
from utils.doc_match_label import match_label_v1


class DataProcess:
    def __init__(self, ocr_result) -> None:
        self.ocr_result = Path(ocr_result)

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

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                json_line = json.loads(line)
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
                    yield json_line
                else:
                    result_list = json_line['result_list']
                    json_lines = []
                    accumulate = 0
                    while True:
                        cur_result_list = []
                        for result in result_list:
                            if result['end'] - result['start'] > max_content_len:
                                logging.warning(
                                    'result['end'] - result ['start'] exceeds max_content_len, which will result in no valid instance being returned'
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
                            }
                        else:
                            json_line = {
                                'content': cur_content,
                                'result_list': cur_result_list,
                                'prompt': prompt,
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
                                }
                            else:
                                json_line = {
                                    'content': res_content,
                                    'result_list': result_list,
                                    'prompt': prompt,
                                }

                            json_lines.append(json_line)
                            break
                        else:
                            content = res_content
                            boxes = res_boxes

                    for json_line in json_lines:
                        yield json_line

    def match_label(self, label_file):
        # 打开label文件
        with open(label_file, 'r', encoding='utf-8') as f:
            raw_example= json.loads(f.read())

        rs = list()
        tmp_dict = dict()
        for line in raw_example:
            items = defaultdict(list)
            cur_pagename = ''

            for e in line['annotations']:
                if not len(e):  # 无标签则跳过
                    continue
                
                pagename = e['page_name']
                # todo:需要判断与上次page是否相同，相同则不需要重新读取ocr结果
                if pagename != cur_pagename:
                    with open(self.ocr_result / f'{pagename}.json', 'r') as f:
                        ocr_results = json.load(f)
                        ocr_bboxes = ocr_results['bboxes']
                        ocr_texts = ocr_results['texts']
                    
                    # 初始化当前page的结果
                    tmp_dict[pagename] = {
                        'content': ocr_texts,
                        'result_list': [],
                        'prompt': '',
                        'image': pagename,
                        'bbox': None,
                    }
                
                # match by gt and ocr rotate box and text
                gt_bbox = e['box']
                gt_text = e['text'][0]
                offsets = match_label_v1(gt_bbox, gt_text, ocr_bboxes, ocr_texts)

                if len(offsets) > 0:
                    items['entities'].append(
                        {
                            'id': e['id'],
                            'text': gt_text,
                            'start_offset': offsets[0][0],
                            'end_offset': offsets[0][1],
                            'label': e['label'],
                        }
                    )
                    # ent_ids.append(e['id'])

                cur_pagename = pagename

            # for r in line['relations']:
            #     if r['from_id'] in ent_ids and r['to_id'] in ent_ids:
            #         items['relations'].append(
            #             {
            #                 'id': r['from_id'] + '-' + r['to_id'],
            #                 'from_id': r['from_id'],
            #                 'to_id': r['to_id'],
            #             }
            #         )

            rs.append(items)
        return rs


if __name__=="__main__":
        
    ocr_file = '/home/youjiachen/PaddleNLP_baidu/workspace/longtext_ie/datasets/contract/dataelem_ocr_res_rotateupright_true'
    label_file = '/home/youjiachen/PaddleNLP_baidu/workspace/longtext_ie/datasets/contract_v1.1/processed_labels.json'
    data_processer = DataProcess(ocr_file)
    res = data_processer.match_label(label_file)
    print(len(res))
    with open('./xxxx.json', 'w') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
