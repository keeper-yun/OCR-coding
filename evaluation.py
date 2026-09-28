# -*- coding: utf-8 -*-

"""
OpenOCR Unified Evaluation

功能：
    1. 全部短文本 Benchmark
    2. LTB 长文本 Benchmark

使用：
    cd /data/huawei/wzc/OpenOCR
    python evaluation.py
"""

import copy
import csv
import gc
import os
import subprocess
import sys



# ------------------------------------------------------------
# 1. 选择 YAML 配置文件
# ------------------------------------------------------------
CONFIG_PATH = (
    "/data/huawei/wzc/OpenOCR/configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_infer_try.yml"
)

# 2. 选择评估类型
# "short" = 全部短文本
# "long"  = LTB 长文本
EVAL_TYPE = "short"
LONG_MAX_RATIO = 20


# ============================================================
#                    固定配置
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    CONFIG_PATH
)

# ============================================================
#                    数据集配置
# ============================================================
SHORT_DATASETS = [
    # -------------------------
    # Common
    # -------------------------
    "../Dataset/evaluation_En/IIIT5k",
    "../Dataset/evaluation_En/SVT",
    "../Dataset/evaluation_En/IC13_857",
    "../Dataset/evaluation_En/IC15_1811",
    "../Dataset/evaluation_En/SVTP",
    "../Dataset/evaluation_En/CUTE80",
    "../Dataset/evaluation_En/IC13_857",
    "../Dataset/evaluation_En/IC15_1811",
]

# ============================================================
#                    基础工具函数
# ============================================================

def resolve_path(path):
    """
    将相对于 OpenOCR 根目录的路径转换为绝对路径。
    """
    return os.path.abspath(
        os.path.join(PROJECT_ROOT, path)
    )


def check_config():
    """
    检查 YAML 配置文件是否存在。
    """
    if not os.path.isfile(CONFIG_PATH):
        raise FileNotFoundError(
            f"\nYAML 配置文件不存在：\n"
            f"{CONFIG_PATH}\n"
        )


def check_lmdb(path):
    """
    检查 LMDB 数据集是否存在。
    """

    abs_path = resolve_path(path)

    if not os.path.isdir(abs_path):
        print(
            f"[SKIP] 数据集目录不存在：{abs_path}"
        )
        return False

    data_mdb = os.path.join(
        abs_path,
        "data.mdb"
    )

    if not os.path.isfile(data_mdb):
        print(
            f"[SKIP] data.mdb 不存在：{abs_path}"
        )
        return False

    return True


def close_lmdb_dataset(dataset):
    """
    关闭 RatioDataSet / RatioDataSetTVResize
    打开的 LMDB 环境。

    OpenOCR 的 lmdb_sets 结构：

        {
            dataset_id: {
                "env": ...,
                "txn": ...,
                ...
            }
        }
    """

    if dataset is None:
        return

    lmdb_sets = getattr(
        dataset,
        "lmdb_sets",
        None
    )

    if not isinstance(lmdb_sets, dict):
        return

    for dataset_info in lmdb_sets.values():

        if not isinstance(dataset_info, dict):
            continue

        # -------------------------
        # 先结束 transaction
        # -------------------------
        txn = dataset_info.get("txn")

        if txn is not None:
            try:
                txn.abort()
            except Exception:
                pass

        # -------------------------
        # 再关闭 LMDB environment
        # -------------------------
        env = dataset_info.get("env")

        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def close_dataloader(dataloader):
    """
    关闭 DataLoader 对应的数据集。
    """

    if dataloader is None:
        return

    dataset = getattr(
        dataloader,
        "dataset",
        None
    )

    close_lmdb_dataset(dataset)

    gc.collect()


# ============================================================
#                    短文本配置
# ============================================================

def prepare_short_config():
    """
    准备短文本 Evaluation 配置。

    主要处理：

        1. 关闭 AMP
        2. 使用 RatioDataSetTVResizeTest
        3. 开启 ratio metric
        4. 设置 max_len
        5. 设置 max_ratio
        6. 保留 real_ratio
    """

    from tools.engine.config import Config

    cfg = Config(CONFIG_PATH)

    config = copy.deepcopy(cfg.cfg)

    # --------------------------------------------------------
    # 1. Evaluation 禁用 AMP
    #
    # 防止：
    #
    # TypeError:
    # Got unsupported ScalarType BFloat16
    #
    # --------------------------------------------------------
    config["Global"]["use_amp"] = False

    # --------------------------------------------------------
    # 2. RatioDataSetTVResize
    #    ->
    #    RatioDataSetTVResizeTest
    # --------------------------------------------------------
    dataset_name = config[
        "Eval"
    ][
        "dataset"
    ][
        "name"
    ]

    if (
        "RatioDataSetTVResize" in dataset_name
        and not dataset_name.endswith("Test")
    ):
        config[
            "Eval"
        ][
            "dataset"
        ][
            "name"
        ] = dataset_name + "Test"

    # --------------------------------------------------------
    # 3. ratio metric
    # --------------------------------------------------------
    config[
        "PostProcess"
    ][
        "with_ratio"
    ] = True

    config[
        "Metric"
    ][
        "with_ratio"
    ] = True

    # --------------------------------------------------------
    # 4. 英文短文本最大长度
    # --------------------------------------------------------
    config[
        "Metric"
    ][
        "max_len"
    ] = 25

    # --------------------------------------------------------
    # 5. 最大宽高比
    # --------------------------------------------------------
    config[
        "Metric"
    ][
        "max_ratio"
    ] = 12

    # --------------------------------------------------------
    # 6. 保存 real_ratio
    # --------------------------------------------------------
    transforms = config[
        "Eval"
    ][
        "dataset"
    ].get(
        "transforms",
        []
    )

    for transform in transforms:

        if "KeepKeys" not in transform:
            continue

        keep_keys = transform[
            "KeepKeys"
        ].get(
            "keep_keys",
            []
        )

        if "real_ratio" not in keep_keys:
            keep_keys.append(
                "real_ratio"
            )

        transform[
            "KeepKeys"
        ][
            "keep_keys"
        ] = keep_keys

    return config


# ============================================================
#                    单个短文本数据集
# ============================================================

def evaluate_short_dataset(
    trainer,
    base_config,
    dataset_path,
    index,
    total
):
    """
    评估一个短文本 LMDB 数据集。

    流程：

        创建 DataLoader
             ↓
        Trainer.eval()
             ↓
        关闭 LMDB
             ↓
        下一个数据集
    """

    from tools.data import build_dataloader

    abs_path = resolve_path(
        dataset_path
    )

    print()
    print("=" * 70)
    print(
        f"[{index}/{total}] "
        f"{os.path.basename(abs_path)}"
    )
    print("=" * 70)
    print(
        f"Path: {abs_path}"
    )

    # --------------------------------------------------------
    # 检查 LMDB
    # --------------------------------------------------------
    if not check_lmdb(dataset_path):
        return None

    # --------------------------------------------------------
    # 每个数据集使用独立 config
    # --------------------------------------------------------
    config = copy.deepcopy(
        base_config
    )

    config[
        "Eval"
    ][
        "dataset"
    ][
        "data_dir_list"
    ] = [
        dataset_path
    ]

    config[
        "Eval"
    ][
        "dataset"
    ][
        "data_dir"
    ] = dataset_path

    # --------------------------------------------------------
    # 创建 DataLoader
    # --------------------------------------------------------
    print(
        "\nCreating DataLoader..."
    )

    dataloader = build_dataloader(
        config,
        "Eval",
        trainer.logger
    )

    print(
        f"DataLoader iters: "
        f"{len(dataloader)}"
    )

    # --------------------------------------------------------
    # 暂存 Trainer 原来的 dataloader
    # --------------------------------------------------------
    old_dataloader = (
        trainer.valid_dataloader
    )

    trainer.valid_dataloader = (
        dataloader
    )

    try:

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------
        print(
            "\nRunning evaluation..."
        )

        metric = trainer.eval()

        print(
            "\nResult:"
        )

        if isinstance(metric, dict):

            for key, value in metric.items():

                try:
                    print(
                        f"  {key:<15}: "
                        f"{float(value):.6f}"
                    )
                except (
                    TypeError,
                    ValueError
                ):
                    print(
                        f"  {key:<15}: "
                        f"{value}"
                    )

        else:
            print(metric)

        return metric

    finally:

        # ----------------------------------------------------
        # 恢复 Trainer
        # ----------------------------------------------------
        trainer.valid_dataloader = (
            old_dataloader
        )

        # ----------------------------------------------------
        # 关闭 LMDB
        # ----------------------------------------------------
        print(
            "\nClosing LMDB..."
        )

        close_dataloader(
            dataloader
        )

        del dataloader

        gc.collect()

        print(
            "LMDB closed."
        )


# ============================================================
#                    保存短文本结果
# ============================================================

def save_short_results(results):
    """
    保存所有短文本 benchmark 的结果。
    """

    output_dir = os.path.join(
        PROJECT_ROOT,
        "output",
        "evaluation"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    result_file = os.path.join(
        output_dir,
        "short_text_results.csv"
    )

    with open(
        result_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "dataset",
            "metric",
            "value"
        ])

        for item in results:

            dataset = item[
                "dataset"
            ]

            metric = item[
                "metric"
            ]

            if isinstance(
                metric,
                dict
            ):

                for key, value in metric.items():

                    writer.writerow([
                        dataset,
                        key,
                        value
                    ])

            else:

                writer.writerow([
                    dataset,
                    "metric",
                    metric
                ])

    print()
    print(
        f"Results saved to:\n"
        f"{result_file}"
    )


# ============================================================
#                    全部短文本
# ============================================================

def run_short_evaluation():
    """
    运行全部短文本 Benchmark。
    """

    from tools.engine.config import Config
    from tools.engine.trainer import Trainer

    print()
    print("=" * 70)
    print("SHORT TEXT EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 准备配置
    # --------------------------------------------------------
    config = prepare_short_config()

    print(
        f"\nDataset count: "
        f"{len(SHORT_DATASETS)}"
    )

    print(
        f"Dataset class: "
        f"{config['Eval']['dataset']['name']}"
    )

    print(
        f"AMP: "
        f"{config['Global'].get('use_amp')}"
    )

    print(
        f"max_len: "
        f"{config['Metric'].get('max_len')}"
    )

    print(
        f"max_ratio: "
        f"{config['Metric'].get('max_ratio')}"
    )

    # --------------------------------------------------------
    # 显示数据集列表
    # --------------------------------------------------------
    print(
        "\nDatasets:"
    )

    for index, path in enumerate(
        SHORT_DATASETS,
        start=1
    ):

        print(
            f"  {index:02d}. "
            f"{path}"
        )

    # --------------------------------------------------------
    # 创建 Config
    # --------------------------------------------------------
    cfg = Config(
        CONFIG_PATH
    )

    cfg.cfg = config

    # --------------------------------------------------------
    # 创建 Trainer
    #
    # Trainer(mode="eval")
    # 会自动创建一次 valid_dataloader。
    #
    # 因此这里创建后立即关闭。
    # --------------------------------------------------------
    print(
        "\nCreating Trainer..."
    )

    trainer = Trainer(
        cfg,
        mode="eval"
    )

    print(
        "Trainer created."
    )

    # --------------------------------------------------------
    # 关闭 Trainer 自动创建的 DataLoader
    # --------------------------------------------------------
    if trainer.valid_dataloader is not None:

        print(
            "\nClosing initial "
            "valid_dataloader..."
        )

        close_dataloader(
            trainer.valid_dataloader
        )

        trainer.valid_dataloader = (
            None
        )

        gc.collect()

        print(
            "Initial DataLoader closed."
        )

    # --------------------------------------------------------
    # 开始逐个评估
    # --------------------------------------------------------
    results = []

    total = len(
        SHORT_DATASETS
    )

    for index, dataset_path in enumerate(
        SHORT_DATASETS,
        start=1
    ):

        metric = evaluate_short_dataset(
            trainer=trainer,
            base_config=config,
            dataset_path=dataset_path,
            index=index,
            total=total
        )

        if metric is not None:

            results.append({
                "dataset": dataset_path,
                "metric": metric
            })

    # --------------------------------------------------------
    # 保存结果
    # --------------------------------------------------------
    save_short_results(
        results
    )

    # --------------------------------------------------------
    # 清理
    # --------------------------------------------------------
    if trainer.valid_dataloader is not None:

        close_dataloader(
            trainer.valid_dataloader
        )

        trainer.valid_dataloader = (
            None
        )

    del trainer

    gc.collect()

    print()
    print("=" * 70)
    print("SHORT TEXT EVALUATION FINISHED")
    print("=" * 70)

    print(
        f"Completed: "
        f"{len(results)} / {total}"
    )


# ============================================================
#                    LTB 长文本
# ============================================================

def run_long_evaluation():
    """
    调用 OpenOCR 原有的 eval_rec_all_long.py。

    对应原命令：

        python tools/eval_rec_all_long.py \
            --c configs/rec/svtrv2/
               svtrv2_smtr_gtc_rctc_infer.yml \
            --o Eval.loader.max_ratio=20
    """

    script_path = os.path.join(
        PROJECT_ROOT,
        "tools",
        "eval_rec_all_long.py"
    )

    if not os.path.isfile(
        script_path
    ):
        raise FileNotFoundError(
            f"\n找不到长文本评估脚本：\n"
            f"{script_path}\n"
        )

    print()
    print("=" * 70)
    print("LONG TEXT EVALUATION")
    print("=" * 70)

    command = [
        sys.executable,
        script_path,
        "--c",
        CONFIG_PATH,
        "--o",
        f"Eval.loader.max_ratio={LONG_MAX_RATIO}",
    ]

    print(
        "\nRunning command:"
    )

    print(
        " ".join(
            f'"{arg}"'
            if " " in arg
            else arg
            for arg in command
        )
    )

    print()

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:

        raise RuntimeError(
            "\nLTB Evaluation failed.\n"
            f"Return code: "
            f"{result.returncode}"
        )

    print()
    print("=" * 70)
    print("LONG TEXT EVALUATION FINISHED")
    print("=" * 70)


# ============================================================
#                    参数检查
# ============================================================

def validate_config():
    """
    检查用户配置是否合法。
    """

    valid_types = {
        "short",
        "long"
    }

    if EVAL_TYPE not in valid_types:

        raise ValueError(
            "\nEVAL_TYPE 错误。\n"
            f"当前：{EVAL_TYPE}\n\n"
            "可选：\n"
            '    "short"\n'
            '    "long"\n'
        )

    if LONG_MAX_RATIO <= 0:

        raise ValueError(
            "\nLONG_MAX_RATIO 必须大于 0。\n"
        )


# ============================================================
#                    主函数
# ============================================================

def main():

    print("=" * 70)
    print("OpenOCR Unified Evaluation")
    print("=" * 70)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Config       : {CONFIG_PATH}"
    )

    print(
        f"Eval type    : {EVAL_TYPE}"
    )

    if EVAL_TYPE == "long":

        print(
            f"LTB max_ratio: "
            f"{LONG_MAX_RATIO}"
        )

    print("=" * 70)

    # --------------------------------------------------------
    # 检查配置
    # --------------------------------------------------------
    validate_config()
    check_config()

    # --------------------------------------------------------
    # 根据 EVAL_TYPE 选择任务
    # --------------------------------------------------------
    if EVAL_TYPE == "short":

        run_short_evaluation()

    elif EVAL_TYPE == "long":

        run_long_evaluation()

if __name__ == "__main__":
    main()

