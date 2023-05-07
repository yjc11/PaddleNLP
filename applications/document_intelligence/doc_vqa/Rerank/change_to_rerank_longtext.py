import json
import random
import math

from tqdm import tqdm

from pathlib import Path
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split

random.seed(14)

label_map = [
    "签订日期",
    "不含税总价",
    "合同生效或失效条款",
    "税率",
    "买方",
    "项目名称",
    "收款方账号",
    "收款方银行",
    "合同号",
    "合同有效期条款",
    "合同总价",
    "卖方",
]


def create_ds(ocr_res_path, proc_label_path, output_path, dataset_name):
    page2text = get_page2text(Path(ocr_res_path))
    page2questions = get_page2questions(Path(proc_label_path))
    print("total ori texts:", len(page2text))

    with open(Path(output_path) / f'{dataset_name}.tsv', "w", encoding="utf8") as f:
        for page, questions in page2questions.items():
            flag = False
            content_1 = page2text[page]
            for ques in questions:
                if content_1 != "":
                    f.write(f"{ques}\t\t{content_1}\t1\n")
                    flag = True
            if flag:
                redundants = list(set(label_map) ^ set(questions))
                idxs = random.sample(range(len(redundants)), k=min(4, len(redundants)))
                for idx in idxs:
                    n_ques = redundants[idx]
                    f.write(f"{n_ques}\t\t{content_1}\t0\n")

        # 无gt页面构造负样本
        for page, content_2 in page2text.items():
            if page not in page2questions and content_2 != "":
                idx = random.sample(range(7), k=1)[0]
                ques = label_map[idx]
                f.write(f"{ques}\t\t{content_2}\t0\n")

    print(
        'total samples:',
        len(open(Path(output_path) / f'{dataset_name}.tsv', "r").readlines()),
    )


def split_datasets(ds_path, dst, seed=14):
    with open(ds_path, "r") as f:
        data = f.read().split("\n")

    train_data, test_data = train_test_split(
        data, shuffle=True, train_size=0.9, test_size=0.1, random_state=seed
    )
    print(len(train_data))
    dic_label = dict()
    dic_one = dict()
    for i in train_data:
        dic_label[i.split("\t\t")[0]] = dic_label.get(i.split("\t\t")[0], 0) + 1
        dic_one[i.split("\t")[-1]] = dic_one.get(i.split("\t")[-1], 0) + 1

    print(dic_label)
    print(dic_one)

    dic_label = dict()
    dic_one = dict()
    for i in test_data:
        dic_label[i.split("\t\t")[0]] = dic_label.get(i.split("\t\t")[0], 0) + 1
        dic_one[i.split("\t")[-1]] = dic_one.get(i.split("\t")[-1], 0) + 1

    print(dic_label)
    print(dic_one)

    with open(Path(dst) / 'train.tsv', "w") as f:
        f.write("\n".join(train_data))

    with open(Path(dst) / 'test.tsv', "w") as f:
        f.write("\n".join(test_data))


def get_page2text(data_path) -> dict:
    """
    适用dataelem ocr result格式
    args：
        data_path：OCR识别结果文件夹的路径
    """
    ocr_res_path = Path(data_path)

    page2content = {}
    for file_ocr_res in list(ocr_res_path.glob('[!.]*')):
        with file_ocr_res.open('r') as f:
            ocr_res = json.load(f)
        page2content[file_ocr_res.stem] = ''.join(ocr_res['texts'])
    return page2content


def get_page2questions(proc_label_path) -> dict:
    """
    适应yjc处理后的label文件格式
    """
    proc_label_path = Path(proc_label_path)
    with proc_label_path.open('r') as f:
        tasks = json.load(f)

    page2questions = defaultdict(list)
    for i in range(len(tasks)):
        for label in tasks[i]['annotations']:
            page2questions[label['page_name']].append(label['label'][0])

    return page2questions


def get_text2page(data_path) -> dict:
    p2t = get_page2text(data_path)
    return {y: k for k, y in p2t.items()}


def create_test_ds(test_data, output_dir):
    """每个字段单独生成一个测试文件"""
    with open(test_data, "r") as f:
        data = f.readlines()

    data_sorted = sorted(data, key=lambda x: x.split("\t\t")[0])

    current_letter = ""
    current_file = None
    for line in data_sorted:
        # print(line)
        letter = line.split("\t\t")[0]
        if letter != current_letter:
            # 关闭上一个输出文件（如果有的话）
            if current_file:
                current_file.close()
            # 打开新的输出文件
            output_file = Path(output_dir, f"{letter}.tsv")
            if output_file.is_file():
                # raise "file has existed !"
                current_file = output_file.open(mode="a")
            else:
                current_file = output_file.open(mode="w")
            current_letter = letter
        # 将当前行写入当前输出文件
        current_file.write(line)

    # 关闭最后一个输出文件
    if current_file:
        current_file.close()


if __name__ == "__main__":
    # proc_label_path = '/home/youjiachen/PaddleNLP_baidu/workspace/longtext_ie/datasets/contract_v1.1/processed_labels.json'
    # ocr_res_path = '/home/youjiachen/PaddleNLP_baidu/workspace/longtext_ie/datasets/contract/dataelem_ocr_res_rotateupright_true'

    # output_path = Path(
    #     '/home/youjiachen/PaddleNLP_baidu/workspace/longtext_ie/datasets/contract_v1.1'
    # )
    # dataset_name = 'exp1'
    # create_ds(ocr_res_path, proc_label_path, output_path, dataset_name)
    # split_datasets(output_path / f'{dataset_name}.tsv', output_path)

    # 按类别划分测试集（一个类别一个测试集）
    # test_data = (
    #     'PaddleNLP_baidu/workspace/longtext_ie/datasets/contract_v1.1/exp1_test.tsv'
    # )
    # output_dir = 'PaddleNLP_baidu/workspace/longtext_ie/datasets/contract_v1.1/test'
    # create_test_ds(test_data, output_dir)
    pass