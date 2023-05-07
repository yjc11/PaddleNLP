import subprocess
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


label_list = [
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
DATA_DIR = Path(__file__).parent
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{DATA_DIR}/score.log"),
        logging.StreamHandler(),  # 输出到控制台
    ],
)


def binary_metric(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    return acc, precision, recall, f1


def evaluate_question(data_path, question, thr=0.9):
    data_path = Path(data_path)
    test_file_path = data_path / f"{question}.tsv"
    score_file_path = data_path / f"{question}.score"
    with test_file_path.open(mode="r") as test_file, score_file_path.open(
        mode="r"
    ) as score_file:
        y_pred = np.array(score_file.read().strip().split("\n"), dtype=np.float32)
        y_pred[y_pred >= thr] = 1
        y_pred[y_pred < thr] = 0
        y_true = np.array(
            list(map(lambda x: int(x.split("\t")[-1].strip()), test_file.readlines()))
        )
    acc, precision, recall, f1 = binary_metric(y_true, y_pred)
    return {
        "Positive": Counter(y_true)[1],
        "Negative": Counter(y_true)[0],
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    }


if __name__ == "__main__":
    data_path = Path('/home/youjiachen/PaddleNLP_baidu/applications/document_intelligence/doc_vqa/Rerank/data/dataelem_base/contract_longtext_test')
    # 各实体分别评分
    for question in label_list:
        subprocess.run(
            ["bash", "run_test_longtext.sh", question], stdout=subprocess.PIPE
        )

    thr = 0.9

    metrics_dict = {q: evaluate_question(data_path, q, thr) for q in label_list}
    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient="index")
    metrics_df.to_excel(data_path / "metrics.xlsx")

    logging.info("*********** Metrics Summary ***********")
    logging.info(metrics_df.to_string())