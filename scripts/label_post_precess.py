import json
import re
import sys
import cv2
import math
import base64
import shutil

import numpy as np

from tqdm import tqdm
from pprint import pprint
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split


def convert_rect(rrect):
    x, y, w, h, theta = rrect
    norm_theta = theta * math.pi / 180
    if w > 0 and h > 0:
        w_angle, h_angle = [norm_theta, norm_theta + math.pi / 2]
    elif w < 0 and h > 0:
        w, h = [h, -w]
        w_angle, h_angle = [norm_theta + math.pi / 2, norm_theta + math.pi]
    elif w < 0 and h < 0:
        [w, h] = [-w, -h]
        [w_angle, h_angle] = [norm_theta + math.pi, norm_theta + 1.5 * math.pi]
    else:
        [w, h] = [-h, w]
        [w_angle, h_angle] = [norm_theta + 1.5 * math.pi, norm_theta + 2 * math.pi]

    horiV = np.array([math.cos(w_angle), math.sin(w_angle)]) / np.linalg.norm(
        [math.cos(w_angle), math.sin(w_angle)]
    )
    vertV = np.array([math.cos(h_angle), math.sin(h_angle)]) / np.linalg.norm(
        [math.cos(h_angle), math.sin(h_angle)]
    )

    p0 = np.array([x, y])
    p1 = (p0 + w * horiV).astype(np.float32)
    p2 = (p1 + h * vertV).astype(np.float32)
    p3 = (p2 - w * horiV).astype(np.float32)

    return [p0.tolist(), p1.tolist(), p2.tolist(), p3.tolist()]


# convert from LS percent units to pixels
def convert_from_ls(result):
    if 'original_width' not in result or 'original_height' not in result:
        return None

    value = result['value']
    w, h = result['original_width'], result['original_height']

    if all([key in value for key in ['x', 'y', 'width', 'height']]):
        return (
            w * value['x'] / 100.0,
            h * value['y'] / 100.0,
            w * value['width'] / 100.0,
            h * value['height'] / 100.0,
        )


# convert from pixels to LS percent units
def convert_to_ls(x, y, width, height, original_width, original_height):
    return (
        x / original_width * 100.0,
        y / original_height * 100.0,
        width / original_width * 100.0,
        height / original_height * 100,
    )


def rotate_bbox(x, y, width, height, angle=0):
    """_summary_

    Args:
        x : 左上角x
        y : 左上角y
        width : box的宽
        height : box的高
        angle : 弧度制旋转角

    """
    xc = x + width / 2
    yc = y + height / 2
    w = width / 2
    h = height / 2
    angle = np.deg2rad(angle)
    cos_ang = math.cos(angle)
    sin_ang = math.sin(angle)

    x1 = xc + (-w) * cos_ang - (-h) * sin_ang
    y1 = yc + (-w) * sin_ang + (-h) * cos_ang

    x2 = xc + (w) * cos_ang - (-h) * sin_ang
    y2 = yc + (w) * sin_ang + (-h) * cos_ang

    x3 = xc + (w) * cos_ang - (h) * sin_ang
    y3 = yc + (w) * sin_ang + (h) * cos_ang

    x4 = xc + (-w) * cos_ang - (h) * sin_ang
    y4 = yc + (-w) * sin_ang + (h) * cos_ang

    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]


def process_label_studio(label_path: str, img_path: str, output_path: str) -> None:
    with open(label_path, 'r') as f:
        raw_result = json.load(f)

    post_processed_result = []
    for task in tqdm(raw_result):
        task_folder = task['data']['Name']
        anno_dict = {'task_name': task_folder, 'annotations': [], 'relations': []}

        for label in task['annotations'][0]['result']:
            # relation
            if 'from_id' in label and 'to_id' in label:
                anno_dict['relations'].append(
                    {'from_id': label['from_id'], 'to_id': label['to_id']}
                )
            if label['type'] != 'labels':
                continue
            num = int(re.search(r'_\d+', label['to_name']).group(0)[1:])
            page = f"page_{num:03d}"

            # refine ori_width and ori_height
            # todo：不清楚是否pdf内所有图片都是一样的尺寸，如果是，需要改进此处
            img = cv2.imread(f'{img_path}/{task_folder}_{page}.png')
            height, width, _ = img.shape
            label['original_width'], label['original_height'] = width, height

            # covert cor
            x, y, w, h = convert_from_ls(label)
            angle = label['value']['rotation']
            box = convert_rect([x, y, w, h, angle])
            task_row = {
                'id': label['id'],
                'page_name': f'{task_folder}_{page}',
                'box': box,
                'rotation': label['value']['rotation'],
                'text': label['meta']['text'] if label.get('meta') else [],  # ocr res
                'label': label['value']['labels'],  # 此处报错说明漏选标签
            }

            anno_dict['annotations'].append(task_row)

        post_processed_result.append(anno_dict)

    with open(Path(output_path) / 'processed_labels.json', 'w') as f:
        json.dump(post_processed_result, f, ensure_ascii=False, indent=2)

    # 保存ori weight和ori height修正后的label studio结果
    with open(Path(output_path) / 'refined_labels.json', 'w') as f:
        json.dump(raw_result, f, ensure_ascii=False, indent=2)


def split_ocr_res_trianval(
    output_path: str, precessed_label_path: str, ocr_res_path: str, seed: int = 144
):
    """根据train val划分ocr结果"""
    with open(precessed_label_path, 'r') as f:
        raw_result = json.load(f)

    # Get pdf name from label file
    pdfname = sorted([i['task_name'] for i in raw_result])
    ocr_res_files = list(Path(ocr_res_path).glob('*.json'))

    # Split train and val data
    train_pdf, val_pdf = train_test_split(
        pdfname, train_size=0.8, test_size=0.2, random_state=seed
    )
    train_output_path = Path(output_path) / 'train' / 'ocr_res'
    val_output_path = Path(output_path) / 'val' / 'ocr_res'
    train_output_path.mkdir(exist_ok=True, parents=True)
    val_output_path.mkdir(exist_ok=True, parents=True)

    # Copy ocr file to train or val folder
    train_count = 0
    val_count = 0
    for file in ocr_res_files:
        file_name = file.stem.split('_page_')[0]
        if file_name in train_pdf:
            shutil.copy(
                file,
                train_output_path,
            )
            train_count += 1
        elif file_name in val_pdf:
            shutil.copy(
                file,
                val_output_path,
            )
            val_count += 1
        else:
            raise ValueError(f'{file_name} not in train or val')
    print(f'train: {train_count}, val: {val_count}')


if __name__ == "__main__":
    # covert label studio json to post processed json
    label_path = '/home/youjiachen/PaddleNLP_baidu/workspace/longtext_ie/datasets/contract_v1.1/labels.json'
    img_path = '/home/youjiachen/PaddleNLP_baidu/workspace/longtext_ie/datasets/contract/Images'
    output_path = (
        '/home/youjiachen/PaddleNLP_baidu/workspace/longtext_ie/datasets/contract_v1.1'
    )

    post_processed_result = process_label_studio(label_path, img_path, output_path)

    # precessed_label_path = '/home/youjiachen/PaddleNLP_baidu/workspace/longtext_ie/datasets/contract_v1.1/processed_labels.json'
    # ocr_res_path = '/home/youjiachen/PaddleNLP_baidu/workspace/longtext_ie/datasets/contract/dataelem_ocr_res_rotateupright_true'

    # split_ocr_res_trianval(output_path, precessed_label_path, ocr_res_path)
