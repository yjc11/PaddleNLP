import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import re
import shutil
import cv2
import numpy as np
import json
import copy
import math
# from socr_dataset.data import SceneDataset

field_map = {'住所': 'address', '经营范围': 'business_range', '注册资本': 'capital', '名称': 'company_name',
             '类型': 'company_type', '组成形式': 'composing_form', '营业期限': 'opration_period',
             '法定代表人': 'owner', '成立日期': 'register_date', '注册号': 'register_number'}
en_ch_map = {val: key for key, val in field_map.items()}


def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)


def list_image(directory, ext='jpg|jpeg|bmp|png|tif|tiff|JPG|PNG|TIF|TIFF'):
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(directory):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    pattern = ext + r'\Z'
    res = [f for f in listOfFiles if re.findall(pattern, f)]
    return res


def rotate_box(box, image_size, angle):
    assert box.shape == (4, 2)
    w, h = image_size
    box_copy = copy.deepcopy(box)
    if angle == 0:
        return box
    if angle == -90:
        box[:, 0] = w - 1 - box_copy[:, 1]
        box[:, 1] = box_copy[:, 0]
        return box
    if angle == 90:
        box[:, 0] = box_copy[:, 1]
        box[:, 1] = h - 1 - box_copy[:, 0]
        return box
    if angle == 180:
        box[:, 0] = w - 1 - box_copy[:, 0]
        box[:, 1] = h - 1 - box_copy[:, 1]
        return box


def rotate_image_only(im, angle):
    """
    rotate image in range[-10,10]
    :param polys:
    :param tags:
    :return:
    """
    def rotate(src, angle, scale=1.0): #1
        w = src.shape[1]
        h = src.shape[0]
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0,2] += rot_move[0]
        rot_mat[1,2] += rot_move[1]
        rotated_image = cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
        return rotated_image

    old_h, old_w, _ = im.shape
    old_center = (old_w / 2, old_h / 2)

    image = rotate(im, angle)
    new_h, new_w, _ = image.shape
    new_center = (new_w / 2, new_h / 2)

    return image, old_center, new_center


def plot_bbox_on_image(image, bboxes, color=(255, 0, 0), show_order=False, rectangle=True):
    for index, box in enumerate(bboxes):
        box = np.array(box)
        if rectangle:
            x1, y1, x2, y2 = min(box[:, 0]), min(box[:, 1]), max(box[:, 0]), max(box[:, 1])
            box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        first_point = (int(float(box[0, 0])), int(float(box[0, 1])))
        cv2.circle(image, first_point, 4, (0, 0, 255), 2)
        cv2.polylines(image, [box.astype(np.int32).reshape((-1, 1, 2))], True,
                      color=color, thickness=2)
        if show_order:
            cv2.putText(image, str(index), (int(float(box[0, 0])), int(float(box[0, 1]))),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1, color=(0, 0, 255))


def vis_image(dataset_folder):
    base_name = os.path.basename(dataset_folder)
    dir_name = os.path.dirname(dataset_folder)
    save_folder = os.path.join(dir_name, base_name + '_show')
    check_folder(save_folder)

    label_or_ocr_not_exist = []
    image_files = list_image(os.path.join(dataset_folder, 'Images', 'train')) + \
                  list_image(os.path.join(dataset_folder, 'Images', 'val'))

    for image_file in image_files:
        image = cv2.imread(image_file)
        image_name = os.path.basename(image_file)
        train_or_val = os.path.basename(os.path.dirname(image_file))

        json_name = os.path.splitext(image_name)[0] + '.json'
        json_file = os.path.join(dataset_folder, 'Images', 'ocr_results', train_or_val, json_name)
        label_file = os.path.join(dataset_folder, 'Labels', json_name)

        if (not os.path.exists(label_file)) or (not os.path.exists(json_file)):
            label_or_ocr_not_exist.append(image_file)
            continue

        with open(json_file, 'r') as f:
            ocr_results = json.load(f)
            rotate_angle = ocr_results['rotate_angle']
            rotateupright = ocr_results['rotateupright']
            text_direction = ocr_results['text_direction']
            bboxes = ocr_results['bboxes']
            texts = ocr_results['texts']
            image_size = ocr_results['image_size']

        with open(label_file, 'r') as f:
            labels = json.load(f)
            categorys = []
            values = []
            gt_bboxes = []
            for label in labels:
                category = label['category']
                value = label['value']
                points = label['points']
                categorys.append(category)
                values.append(value)
                gt_bboxes.append(points)

        for index, bbox in enumerate(gt_bboxes):
            bbox = np.array(bbox)
            gt_bboxes[index] = rotate_box(bbox, image_size, rotate_angle)

        image, _, _ = rotate_image_only(image, rotate_angle)
        plot_bbox_on_image(image, bboxes, show_order=True, rectangle=False)
        plot_bbox_on_image(image, gt_bboxes, color=(0, 0, 255), rectangle=False)
        cv2.imwrite(os.path.join(save_folder, image_name), image)


def convert_label(dataset_folder):
    sd = SceneDataset(dataset_folder, reunzip=False)
    field_def = sd.field_def
    train_ocr_files = sd.tr_ocr_files
    val_ocr_files = sd.val_ocr_files
    train_lab_files = sd.tr_lab_files
    val_lab_files = sd.val_lab_files

    base_name = os.path.basename(dataset_folder)
    dir_name = os.path.dirname(dataset_folder)
    save_folder = os.path.join(dir_name, base_name + '_convert')
    save_image_folder = os.path.join(save_folder, 'images')
    save_ocr_folder = os.path.join(save_folder, 'ocr_results')
    save_label_folder = os.path.join(save_folder, 'Labels')
    save_val_image_folder = os.path.join(save_folder, 'val_images')
    check_folder(save_folder)
    check_folder(save_image_folder)
    check_folder(save_val_image_folder)
    check_folder(save_ocr_folder)
    check_folder(save_label_folder)

    label_train_studio = []
    label_val_studio = []
    label_or_ocr_not_exist = []
    image_files = list_image(os.path.join(dataset_folder, 'Images', 'train')) + \
                  list_image(os.path.join(dataset_folder, 'Images', 'val'))
    for image_file in image_files:
        image = cv2.imread(image_file)
        image_name = os.path.basename(image_file)
        train_or_val = os.path.basename(os.path.dirname(image_file))

        json_name = os.path.splitext(image_name)[0] + '.json'
        json_file = os.path.join(dataset_folder, 'Images', 'ocr_results', train_or_val, json_name)
        label_file = os.path.join(dataset_folder, 'Labels', json_name)

        if (json_file not in train_ocr_files) and (json_file not in val_ocr_files):
            label_or_ocr_not_exist.append(image_file)
            continue

        with open(json_file, 'r') as f:
            ocr_results = json.load(f)
            rotate_angle = ocr_results['rotate_angle']
            rotateupright = ocr_results['rotateupright']
            text_direction = ocr_results['text_direction']
            bboxes = ocr_results['bboxes']
            texts = ocr_results['texts']
            image_size = ocr_results['image_size']

        with open(label_file, 'r') as f:
            labels = json.load(f)
            categorys = []
            values = []
            gt_bboxes = []
            for label in labels:
                category = label['category']
                value = label['value']
                points = label['points']
                categorys.append(category)
                values.append(value)
                gt_bboxes.append(points)

        shutil.copy(json_file, os.path.join(save_ocr_folder, json_name))
        shutil.copy(label_file, os.path.join(save_label_folder, json_name))

        # rotate gt_boxes
        for index, bbox in enumerate(gt_bboxes):
            bbox = np.array(bbox)
            gt_bboxes[index] = rotate_box(bbox, image_size, rotate_angle)

        # rotate image
        image, _, _ = rotate_image_only(image, rotate_angle)
        cv2.imwrite(os.path.join(save_image_folder, image_name), image)
        if train_or_val == 'val':
            cv2.imwrite(os.path.join(save_val_image_folder, image_name), image)

        original_height, original_width, _ = image.shape
        convert_label = {}
        convert_label['annotations'] = [dict()]
        result = []
        for index, bbox in enumerate(gt_bboxes):
            # 处理成水平box
            bbox = np.array(bbox)
            x1, y1, x2, y2 = min(bbox[:, 0]), min(bbox[:, 1]), max(bbox[:, 0]), max(bbox[:, 1])
            info = dict()
            info['original_width'] = int(original_width)
            info['original_height'] = int(original_height)
            info['value'] = dict()
            if categorys[index] not in field_def:
                print(image_file, f'{categorys[index]} does not exist in field_def.')
                continue
            info['value']['rectanglelabels'] = [categorys[index]]
            # info['value']['rectanglelabels'] = [en_ch_map[categorys[index]]]
            info['value']['x'] = float(x1 / original_width * 100)
            info['value']['y'] = float(y1 / original_height * 100)
            info['value']['width'] = float((x2 - x1) / original_width * 100)
            info['value']['height'] = float((y2 - y1) / original_height * 100)
            info['id'] = os.path.splitext(image_name)[0] + '_' + str(index)
            info['type'] = 'rectanglelabels'
            info['origin_bbox'] = bbox.tolist()
            info['origin_text'] = values[index]
            result.append(info)
        convert_label['annotations'][0]['result'] = result
        convert_label['data'] = {'image': 'prefix-' + image_name}

        if train_or_val == 'train':
            label_train_studio.append(convert_label)
        else:
            label_val_studio.append(convert_label)

    with open(os.path.join(save_folder, 'label_train_studio.json'), 'w') as f:
        f.write(json.dumps(label_train_studio, ensure_ascii=False))
    with open(os.path.join(save_folder, 'label_val_studio.json'), 'w') as f:
        f.write(json.dumps(label_val_studio, ensure_ascii=False))

    print('total image: {}, label_or_ocr_not_exist image: {}'.format(len(image_files), len(label_or_ocr_not_exist)))


# def sample_train_label(dataset_folder, sample_num=20):
#     sd = SceneDataset(dataset_folder, reunzip=False)
#     field_def = sd.field_def
#     train_ocr_files = sd.tr_ocr_files
#     val_ocr_files = sd.val_ocr_files
#     train_lab_files = sd.tr_lab_files
#     val_lab_files = sd.val_lab_files

#     base_name = os.path.basename(dataset_folder)
#     dir_name = os.path.dirname(dataset_folder)
#     save_folder = os.path.join(dir_name, base_name + '_convert')

#     label_train_studio = []
#     image_files = list_image(os.path.join(dataset_folder, 'Images', 'train'))
#     np.random.seed(42)
#     np.random.shuffle(image_files)
#     num = 0
#     for image_file in image_files:
#         image = cv2.imread(image_file)
#         image_name = os.path.basename(image_file)
#         train_or_val = os.path.basename(os.path.dirname(image_file))

#         json_name = os.path.splitext(image_name)[0] + '.json'
#         json_file = os.path.join(dataset_folder, 'Images', 'ocr_results', train_or_val, json_name)
#         label_file = os.path.join(dataset_folder, 'Labels', json_name)

#         if json_file not in train_ocr_files:
#             continue

#         with open(json_file, 'r') as f:
#             ocr_results = json.load(f)
#             rotate_angle = ocr_results['rotate_angle']
#             rotateupright = ocr_results['rotateupright']
#             text_direction = ocr_results['text_direction']
#             bboxes = ocr_results['bboxes']
#             texts = ocr_results['texts']
#             image_size = ocr_results['image_size']

#         with open(label_file, 'r') as f:
#             labels = json.load(f)
#             categorys = []
#             values = []
#             gt_bboxes = []
#             for label in labels:
#                 category = label['category']
#                 value = label['value']
#                 points = label['points']
#                 categorys.append(category)
#                 values.append(value)
#                 gt_bboxes.append(points)

#         # rotate gt_boxes
#         for index, bbox in enumerate(gt_bboxes):
#             bbox = np.array(bbox)
#             gt_bboxes[index] = rotate_box(bbox, image_size, rotate_angle)

#         # rotate image
#         image, _, _ = rotate_image_only(image, rotate_angle)

#         original_height, original_width, _ = image.shape
#         convert_label = {}
#         convert_label['annotations'] = [dict()]
#         result = []
#         for index, bbox in enumerate(gt_bboxes):
#             # 处理成水平box
#             bbox = np.array(bbox)
#             x1, y1, x2, y2 = min(bbox[:, 0]), min(bbox[:, 1]), max(bbox[:, 0]), max(bbox[:, 1])
#             info = dict()
#             info['original_width'] = int(original_width)
#             info['original_height'] = int(original_height)
#             info['value'] = dict()
#             if categorys[index] not in field_def:
#                 print(image_file, f'{categorys[index]} does not exist in field_def.')
#                 continue
#             info['value']['rectanglelabels'] = [categorys[index]]
#             # info['value']['rectanglelabels'] = [en_ch_map[categorys[index]]]
#             info['value']['x'] = float(x1 / original_width * 100)
#             info['value']['y'] = float(y1 / original_height * 100)
#             info['value']['width'] = float((x2 - x1) / original_width * 100)
#             info['value']['height'] = float((y2 - y1) / original_height * 100)
#             info['id'] = os.path.splitext(image_name)[0] + '_' + str(index)
#             info['type'] = 'rectanglelabels'
#             info['origin_bbox'] = bbox.tolist()
#             info['origin_text'] = values[index]
#             result.append(info)
#         convert_label['annotations'][0]['result'] = result
#         convert_label['data'] = {'image': 'prefix-' + image_name}

#         label_train_studio.append(convert_label)
#         num += 1
#         if num == sample_num:
#             break

#     with open(os.path.join(save_folder, f'label_train_studio_sample_{sample_num}.json'), 'w') as f:
#         f.write(json.dumps(label_train_studio, ensure_ascii=False))


if __name__ == '__main__':
    # dataset_folder = '/Users/gulixin/Desktop/数据/智能结构化数据/socr_dataset/身份证'
    # vis_image(dataset_folder)
    # convert_label(dataset_folder)

    # dataset_folder = '/Users/gulixin/Desktop/数据/智能结构化数据/socr_dataset/营业执照'
    # convert_label(dataset_folder)

    # dataset_folder = '/Users/gulixin/Desktop/数据/智能结构化数据/socr_dataset/普通发票'
    # convert_label(dataset_folder)

    # dataset_folder = '/Users/gulixin/Desktop/数据/智能结构化数据/socr_dataset/研报'
    # convert_label(dataset_folder)

    # dataset_folder = '/Users/gulixin/Desktop/数据/智能结构化数据/socr_dataset/简历'
    # convert_label(dataset_folder)

    # dataset_folder = '/Users/gulixin/Desktop/数据/智能结构化数据/socr_dataset/EPHOIE'
    # convert_label(dataset_folder)

    # dataset_folder = '/Users/gulixin/Desktop/数据/智能结构化数据/socr_dataset/第一类医疗器械生产备案凭证'
    # convert_label(dataset_folder)

    # dataset_folder = '/Users/gulixin/Desktop/数据/智能结构化数据/socr_dataset/出租车发票'
    # convert_label(dataset_folder)

    # dataset_folder = '/Users/gulixin/Desktop/数据/智能结构化数据/socr_dataset/航空行程单'
    # convert_label(dataset_folder)

    # dataset_folder = '/Users/gulixin/Desktop/数据/智能结构化数据/socr_dataset/安全生产许可证'
    # convert_label(dataset_folder)

    # dataset_folder = '/Users/gulixin/Desktop/数据/智能结构化数据/socr_dataset/驾驶证'
    # convert_label(dataset_folder)

    # dataset_folder = '/Users/gulixin/Desktop/数据/智能结构化数据/socr_dataset/火车票'
    # convert_label(dataset_folder)

    # dataset_folder = '/Users/gulixin/Desktop/数据/智能结构化数据/socr_dataset/专利证书'
    # convert_label(dataset_folder)

    # dataset_folder = '/Users/gulixin/Desktop/数据/智能结构化数据/socr_dataset/信用等级证书'
    # convert_label(dataset_folder)

    sample_train_label(dataset_folder, sample_num=5)


