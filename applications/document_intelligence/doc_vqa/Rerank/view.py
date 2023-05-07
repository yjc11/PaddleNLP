import shutil
import numpy as np
import pandas as pd

from pathlib import Path
from change_to_rerank_longtext import get_text2page


def calculate_metrics(predictions, gt):
    true_positives = np.logical_and(predictions == 1, gt == 1)
    false_negatives = np.logical_and(predictions == 0, gt == 1)
    false_positives = np.logical_and(predictions == 1, gt == 0)
    true_negatives = np.logical_and(predictions == 0, gt == 0)

    tp_indices = [i for i, x in enumerate(true_positives) if x]
    fn_indices = [i for i, x in enumerate(false_negatives) if x]
    fp_indices = [i for i, x in enumerate(false_positives) if x]
    tn_indices = [i for i, x in enumerate(true_negatives) if x]

    return tp_indices, fn_indices, fp_indices, tn_indices


import time


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to run.")
        return result

    return wrapper


@timer
def copy_predicted_images(
    score_file_path: str,
    gt_file_path: str,
    ocr_res_files_path: str,
    page_files_path: str,
    output_dir_path: str,
    threshold=0.9,
    vis_badcase=False,
) -> list:
    """根据分数文件复制预测为正类的图片到输出目录中"""

    def copy_pages_to_output_path(pred_pages, oup_path) -> None:
        for img_path in pred_pages:
            shutil.copy(img_path, oup_path)

    score_file = Path(score_file_path)
    label = score_file.stem
    output_path = Path(output_dir_path) / label

    with score_file.open('r') as f:
        score = f.readlines()
        y_pred = np.array([float(s.strip()) for s in score]) > threshold

    # 挑选出预测为正类的文本及其对应页码。（注意quoting=3，否则文本中的双引号会被处理掉）
    results_df = pd.read_csv(gt_file_path, sep='\t', header=None, quoting=3)
    pred_results_df = results_df.iloc[y_pred]
    texts = pred_results_df[2].tolist()
    # todo:将ocr结果保存在一个json文件中，以加快速度
    text2page = get_text2page(ocr_res_files_path)
    pred_pages = [text2page[text] for text in texts]

    # 找到预测为正类的图片，复制到输出目录中
    page_files_path = Path(page_files_path)
    pred_page_path = [p for p in page_files_path.glob('[!.]*') if p.stem in pred_pages]
    # output_path.mkdir(exist_ok=True, parents=True)
    # copy_pages_to_output_path(pred_page_path, output_path)

    # 混淆矩阵分析
    if vis_badcase:
        gt_data = pd.read_csv(
            gt_file_path,
            sep='\t',
            header=None,
            names=['img', '-', 'text', 'label'],
            quoting=3,
        )
        gt = gt_data['label'].values
        tp, fn, fp, tn = calculate_metrics(y_pred, gt)
        tp_data, fn_data, fp_data, tn_data = (
            results_df.iloc[idx] for idx in (tp, fn, fp, tn)
        )
        tp_pages = [text2page[t] for t in tp_data[2]]
        fn_pages = [text2page[t] for t in fn_data[2]]
        fp_pages = [text2page[t] for t in fp_data[2]]
        tn_pages = [text2page[t] for t in tn_data[2]]
        tp_pages_path = [p for p in page_files_path.glob('[!.]*') if p.stem in tp_pages]
        fn_pages_path = [p for p in page_files_path.glob('[!.]*') if p.stem in fn_pages]
        fp_pages_path = [p for p in page_files_path.glob('[!.]*') if p.stem in fp_pages]
        tn_pages_path = [p for p in page_files_path.glob('[!.]*') if p.stem in tn_pages]
        tp_oup = output_path / 'tp'
        tp_oup.mkdir(exist_ok=True, parents=True)
        fn_oup = output_path / 'fn'
        fn_oup.mkdir(exist_ok=True, parents=True)
        fp_oup = output_path / 'fp'
        fp_oup.mkdir(exist_ok=True, parents=True)
        tn_oup = output_path / 'tn'
        tn_oup.mkdir(exist_ok=True, parents=True)
        copy_pages_to_output_path(tp_pages_path, tp_oup)
        copy_pages_to_output_path(fn_pages_path, fn_oup)
        copy_pages_to_output_path(fp_pages_path, fp_oup)
        copy_pages_to_output_path(tn_pages_path, tn_oup)

        return tp_pages, fn_pages, fp_pages, tn_pages

    else:
        return pred_pages


if __name__ == "__main__":
    res = copy_predicted_images(
        '/home/youjiachen/PaddleNLP_baidu/applications/document_intelligence/doc_vqa/Rerank/data/dataelem_base/contract_longtext_test/签订日期.score',
        '/home/youjiachen/PaddleNLP_baidu/applications/document_intelligence/doc_vqa/Rerank/data/dataelem_base/contract_longtext_test/签订日期.tsv',
        '/home/youjiachen/PaddleNLP_baidu/workspace/datasets/contract_longtext/dataelem_ocr_res_rotateupright_true',
        '/home/youjiachen/PaddleNLP_baidu/workspace/datasets/contract_longtext/Images',
        '/home/youjiachen/PaddleNLP_baidu/applications/document_intelligence/doc_vqa/Rerank/data/pred_img',
        threshold=0.9,
        vis_badcase=True,
    )
