# -*- coding: utf-8 -*-

import csv
import copy
import gc
import os
import sys
import traceback


# ============================================================
# 0. 项目根目录
# ============================================================

__dir__ = os.path.dirname(
    os.path.abspath(__file__)
)

if __dir__ not in sys.path:
    sys.path.insert(0, __dir__)


# ============================================================
# 1. 固定 YAML 配置文件
# ============================================================
#
# 注意：
# 这里直接写 YAML 路径
#
# 所以运行时不需要：
#
# python evaluation.py --config xxx.yml
#
# 直接：
#
# python evaluation.py
#
# ============================================================

CONFIG_PATH = os.path.join(
    __dir__,
    "configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_infer_try.yml"
)


# ============================================================
# 2. 导入 OpenOCR
# ============================================================

from tools.data import build_dataloader
from tools.engine.config import Config
from tools.engine.trainer import Trainer


# ============================================================
# 3. 关闭 LMDB
# ============================================================

def close_lmdb_dataset(dataloader):
    """
    关闭 RatioDataSetTVResizeTest 中打开的 LMDB。

    你的 ratio_dataset_tvresize_test.py 中：

        lmdb_sets[dataset_idx] = {
            'dirpath': dirpath,
            'env': env,
            'txn': txn,
            'num_samples': num_samples,
            'ratio_num_samples': ...
        }

    所以：

        lmdb_sets

    是一个 dict：

        {
            0: {...},
            1: {...},
            2: {...}
        }

    必须使用：

        lmdb_sets.values()

    而不能：

        for env in lmdb_sets
    """

    if dataloader is None:
        return

    try:

        # ----------------------------------------------------
        # 获取 Dataset
        # ----------------------------------------------------

        dataset = getattr(
            dataloader,
            "dataset",
            None
        )

        if dataset is None:
            return

        # ----------------------------------------------------
        # 获取 lmdb_sets
        # ----------------------------------------------------

        lmdb_sets = getattr(
            dataset,
            "lmdb_sets",
            None
        )

        if lmdb_sets is None:
            return

        # ----------------------------------------------------
        # 逐个关闭 LMDB
        # ----------------------------------------------------

        for dataset_info in lmdb_sets.values():

            if dataset_info is None:
                continue

            # =================================================
            # 先关闭 transaction
            # =================================================

            txn = dataset_info.get(
                "txn",
                None
            )

            if txn is not None:

                try:
                    txn.abort()
                except Exception:
                    pass

            # =================================================
            # 再关闭 environment
            # =================================================

            env = dataset_info.get(
                "env",
                None
            )

            if env is not None:

                try:
                    env.close()
                except Exception:
                    pass

        print(
            "LMDB environments closed."
        )

    except Exception as e:

        print(
            f"Warning when closing LMDB: {e}"
        )


# ============================================================
# 4. 释放 DataLoader
# ============================================================

def release_dataloader(dataloader):

    if dataloader is None:
        return

    # --------------------------------------------------------
    # 关闭 LMDB
    # --------------------------------------------------------

    close_lmdb_dataset(
        dataloader
    )

    # --------------------------------------------------------
    # 尝试关闭 DataLoader worker
    # --------------------------------------------------------

    try:

        iterator = getattr(
            dataloader,
            "_iterator",
            None
        )

        if iterator is not None:

            shutdown_workers = getattr(
                iterator,
                "_shutdown_workers",
                None
            )

            if shutdown_workers is not None:

                try:
                    shutdown_workers()
                except Exception:
                    pass

    except Exception:
        pass

    # --------------------------------------------------------
    # 删除引用
    # --------------------------------------------------------

    try:
        del dataloader
    except Exception:
        pass

    # --------------------------------------------------------
    # Python 垃圾回收
    # --------------------------------------------------------

    gc.collect()


# ============================================================
# 5. 检查 YAML
# ============================================================

def check_config():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Checking configuration"
    )

    print(
        "=" * 70
    )

    print(
        f"Config path:\n{CONFIG_PATH}"
    )

    if not os.path.isfile(
        CONFIG_PATH
    ):

        raise FileNotFoundError(
            f"\nYAML file not found:\n{CONFIG_PATH}"
        )

    print(
        "Config file exists."
    )


# ============================================================
# 6. 准备配置
# ============================================================

def prepare_config(cfg):

    # ========================================================
    # 判断 RatioDataSet
    # ========================================================

    dataset_name = cfg.cfg[
        "Eval"
    ][
        "dataset"
    ][
        "name"
    ]

    msr = (
        "RatioDataSet"
        in dataset_name
    )

    print(
        f"\nEval dataset class:"
    )

    print(
        dataset_name
    )

    print(
        f"Use RatioDataSet:"
        f" {msr}"
    )

    # ========================================================
    # output_dir
    # ========================================================

    output_dir = cfg.cfg[
        "Global"
    ].get(
        "output_dir",
        None
    )

    if output_dir is not None:

        if output_dir.endswith(
            "/"
        ):

            output_dir = output_dir[:-1]

            cfg.cfg[
                "Global"
            ][
                "output_dir"
            ] = output_dir

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    # ========================================================
    # checkpoint
    # ========================================================

    pretrained_model = cfg.cfg[
        "Global"
    ].get(
        "pretrained_model",
        None
    )

    if pretrained_model is None:

        if output_dir is None:

            raise RuntimeError(
                "Global.output_dir is None "
                "and pretrained_model is also None."
            )

        cfg.cfg[
            "Global"
        ][
            "pretrained_model"
        ] = (
            output_dir
            + "/best.pth"
        )

    print(
        "\nCheckpoint:"
    )

    print(
        cfg.cfg[
            "Global"
        ][
            "pretrained_model"
        ]
    )

    # ========================================================
    # 关闭 AMP
    #
    # 防止：
    #
    # TypeError:
    # Got unsupported ScalarType BFloat16
    # ========================================================

    cfg.cfg[
        "Global"
    ][
        "use_amp"
    ] = False

    print(
        "\nGlobal.use_amp = False"
    )

    # ========================================================
    # RatioDataSet 相关设置
    # ========================================================

    if msr:

        # ----------------------------------------------------
        # PostProcess
        # ----------------------------------------------------

        if "PostProcess" in cfg.cfg:

            cfg.cfg[
                "PostProcess"
            ][
                "with_ratio"
            ] = True

        # ----------------------------------------------------
        # Metric
        # ----------------------------------------------------

        if "Metric" in cfg.cfg:

            cfg.cfg[
                "Metric"
            ][
                "with_ratio"
            ] = True

            cfg.cfg[
                "Metric"
            ][
                "max_len"
            ] = 25

            cfg.cfg[
                "Metric"
            ][
                "max_ratio"
            ] = 12

        # ----------------------------------------------------
        # transforms
        # ----------------------------------------------------

        transforms = cfg.cfg[
            "Eval"
        ][
            "dataset"
        ].get(
            "transforms",
            []
        )

        found_keep_keys = False

        for transform in transforms:

            if "KeepKeys" not in transform:
                continue

            found_keep_keys = True

            keep_keys = transform[
                "KeepKeys"
            ].setdefault(
                "keep_keys",
                []
            )

            if "real_ratio" not in keep_keys:

                keep_keys.append(
                    "real_ratio"
                )

                print(
                    "Added real_ratio to KeepKeys."
                )

            break

        if not found_keep_keys:

            print(
                "Warning: KeepKeys transform "
                "was not found."
            )

    return cfg


# ============================================================
# 7. 获取评估数据集
# ============================================================

def get_data_dirs(cfg):

    dataset_cfg = cfg.cfg[
        "Eval"
    ][
        "dataset"
    ]

    data_dir_list = dataset_cfg.get(
        "data_dir_list",
        None
    )

    if data_dir_list is None:

        raise RuntimeError(
            "Eval.dataset.data_dir_list "
            "is not configured in YAML."
        )

    if len(data_dir_list) == 0:

        raise RuntimeError(
            "Eval.dataset.data_dir_list is empty."
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Evaluation datasets"
    )

    print(
        "=" * 70
    )

    for index, path in enumerate(
        data_dir_list,
        start=1
    ):

        print(
            f"{index}. {path}"
        )

    return data_dir_list


# ============================================================
# 8. 检查数据集路径
# ============================================================

def check_data_dirs(
    data_dirs
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Checking dataset paths"
    )

    print(
        "=" * 70
    )

    all_exist = True

    for data_dir in data_dirs:

        # ----------------------------------------------------
        # 相对路径按照 OpenOCR 根目录解析
        # ----------------------------------------------------

        if os.path.isabs(
            data_dir
        ):

            real_path = data_dir

        else:

            real_path = os.path.abspath(
                os.path.join(
                    __dir__,
                    data_dir
                )
            )

        if os.path.exists(
            real_path
        ):

            print(
                f"[OK]   {data_dir}"
            )

        else:

            print(
                f"[FAIL] {data_dir}"
            )

            print(
                f"       -> {real_path}"
            )

            all_exist = False

    print(
        "=" * 70
    )

    return all_exist


# ============================================================
# 9. 评估一个数据集
# ============================================================

def evaluate_one_dataset(
    trainer,
    base_cfg,
    datadir,
    index
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"[{index}] Evaluating dataset"
    )

    print(
        f"Path: {datadir}"
    )

    print(
        "=" * 70
    )

    valid_dataloader = None

    try:

        # ====================================================
        # 深拷贝配置
        # ====================================================

        config_each = copy.deepcopy(
            base_cfg
        )

        # ====================================================
        # 每次只加载一个 LMDB
        #
        # 例如第一次：
        #
        # data_dir_list:
        #   - CUTE80
        #
        # 第二次：
        #
        # data_dir_list:
        #   - IC13_857
        #
        # ====================================================

        config_each[
            "Eval"
        ][
            "dataset"
        ][
            "data_dir_list"
        ] = [
            datadir
        ]

        # ====================================================
        # 如果不是 RatioDataSet
        # 设置 data_dir
        # ====================================================

        dataset_name = config_each[
            "Eval"
        ][
            "dataset"
        ][
            "name"
        ]

        if "RatioDataSet" not in dataset_name:

            config_each[
                "Eval"
            ][
                "dataset"
            ][
                "data_dir"
            ] = datadir

        # ====================================================
        # 创建 DataLoader
        # ====================================================

        print(
            "\nBuilding dataloader..."
        )

        valid_dataloader = build_dataloader(
            config_each,
            "Eval",
            trainer.logger
        )

        print(
            "Dataloader built successfully."
        )

        # ====================================================
        # 设置 Trainer 的验证 DataLoader
        # ====================================================

        trainer.valid_dataloader = (
            valid_dataloader
        )

        # ====================================================
        # 开始评估
        # ====================================================

        print(
            "\nRunning evaluation..."
        )

        metric = trainer.eval()

        print(
            "\nEvaluation finished."
        )

        print(
            "\nMetric:"
        )

        print(
            metric
        )

        return metric

    except Exception as e:

        print(
            "\n"
            + "!" * 70
        )

        print(
            "Evaluation failed"
        )

        print(
            f"Dataset: {datadir}"
        )

        print(
            f"Error: {e}"
        )

        print(
            "!" * 70
        )

        traceback.print_exc()

        return None

    finally:

        # ====================================================
        # 非常重要：
        #
        # 每个数据集评估完以后立即关闭 LMDB
        # ====================================================

        print(
            "\nClosing LMDB environment..."
        )

        if valid_dataloader is not None:

            close_lmdb_dataset(
                valid_dataloader
            )

        # ====================================================
        # Trainer 不再持有 DataLoader
        # ====================================================

        trainer.valid_dataloader = None

        # ====================================================
        # 删除引用
        # ====================================================

        valid_dataloader = None

        gc.collect()

        print(
            "LMDB environment released."
        )


# ============================================================
# 10. metric 转 dict
# ============================================================

def metric_to_dict(
    metric
):

    if metric is None:
        return {}

    if isinstance(
        metric,
        dict
    ):

        return metric

    try:

        if hasattr(
            metric,
            "__dict__"
        ):

            return dict(
                metric.__dict__
            )

    except Exception:
        pass

    return {
        "metric": str(metric)
    }


# ============================================================
# 11. 保存详细结果
# ============================================================

def save_results_csv(
    results,
    output_dir
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_csv = os.path.join(
        output_dir,
        "evaluation_results.csv"
    )

    # ========================================================
    # 获取所有 metric 字段
    # ========================================================

    metric_keys = []

    for result in results:

        metric = result.get(
            "metric",
            {}
        )

        if not isinstance(
            metric,
            dict
        ):
            continue

        for key in metric.keys():

            if key not in metric_keys:

                metric_keys.append(
                    key
                )

    fieldnames = [
        "dataset"
    ] + metric_keys

    # ========================================================
    # 写 CSV
    # ========================================================

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:

            row = {
                "dataset": result[
                    "dataset"
                ]
            }

            metric = result.get(
                "metric",
                {}
            )

            if isinstance(
                metric,
                dict
            ):

                for key in metric_keys:

                    row[key] = metric.get(
                        key,
                        ""
                    )

            writer.writerow(
                row
            )

    print(
        "\nDetailed results saved:"
    )

    print(
        output_csv
    )


# ============================================================
# 12. 计算平均结果
# ============================================================

def calculate_average(
    results
):

    metric_values = {}

    for result in results:

        metric = result.get(
            "metric",
            {}
        )

        if not isinstance(
            metric,
            dict
        ):
            continue

        for key, value in metric.items():

            if isinstance(
                value,
                bool
            ):
                continue

            try:

                number = float(
                    value
                )

            except (
                TypeError,
                ValueError
            ):

                continue

            metric_values.setdefault(
                key,
                []
            ).append(
                number
            )

    averages = {}

    for key, values in metric_values.items():

        if len(values) == 0:
            continue

        averages[key] = (
            sum(values)
            / len(values)
        )

    return averages


# ============================================================
# 13. 保存平均结果
# ============================================================

def save_average_csv(
    averages,
    output_dir
):

    output_csv = os.path.join(
        output_dir,
        "evaluation_average.csv"
    )

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "metric",
                "average"
            ]
        )

        for key, value in averages.items():

            writer.writerow(
                [
                    key,
                    value
                ]
            )

    print(
        "\nAverage results saved:"
    )

    print(
        output_csv
    )


# ============================================================
# 14. main
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "OpenOCR SVTRv2 English Short Text Evaluation"
    )

    print(
        "=" * 70
    )

    print(
        f"\nProject root:"
    )

    print(
        __dir__
    )

    # ========================================================
    # 14.1 检查 YAML
    # ========================================================

    check_config()

    # ========================================================
    # 14.2 读取 YAML
    #
    # 这里直接使用 CONFIG_PATH
    #
    # 不再使用：
    #
    # parse_args()
    #
    # 所以不会再要求：
    #
    # --config
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Loading YAML"
    )

    print(
        "=" * 70
    )

    cfg = Config(
        CONFIG_PATH
    )

    # ========================================================
    # 14.3 准备配置
    # ========================================================

    cfg = prepare_config(
        cfg
    )

    # ========================================================
    # 14.4 RatioDataSet：
    #
    # RatioDataSetTVResize
    #          ↓
    # RatioDataSetTVResizeTest
    #
    # 因为 evaluation 使用的是 Test Dataset。
    # ========================================================

    dataset_name = cfg.cfg[
        "Eval"
    ][
        "dataset"
    ][
        "name"
    ]

    if (
        "RatioDataSetTVResize"
        in dataset_name
        and
        not dataset_name.endswith(
            "Test"
        )
    ):

        cfg.cfg[
            "Eval"
        ][
            "dataset"
        ][
            "name"
        ] = (
            dataset_name
            + "Test"
        )

        print(
            "\nDataset class changed:"
        )

        print(
            f"    {dataset_name}"
        )

        print(
            "       ↓"
        )

        print(
            f"    {cfg.cfg['Eval']['dataset']['name']}"
        )

    # ========================================================
    # 14.5 获取测试集
    # ========================================================

    data_dirs = get_data_dirs(
        cfg
    )

    # ========================================================
    # 14.6 检查测试集
    # ========================================================

    if not check_data_dirs(
        data_dirs
    ):

        print(
            "\nSome datasets do not exist."
        )

        print(
            "Please check your YAML:"
        )

        print(
            CONFIG_PATH
        )

        return

    # ========================================================
    # 14.7 创建 Trainer
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Creating Trainer"
    )

    print(
        "=" * 70
    )

    trainer = Trainer(
        cfg,
        mode="eval"
    )

    print(
        "\nTrainer created."
    )

    # ========================================================
    # ！！！非常重要！！！
    #
    # Trainer(cfg, mode='eval')
    #
    # 在初始化时会自动执行：
    #
    # self.valid_dataloader = build_dataloader(...)
    #
    # 因此它已经打开 YAML 中的全部：
    #
    # CUTE80
    # IC13_857
    # IC15_1811
    # IIIT5k
    # SVT
    # SVTP
    #
    # 我们后面又需要：
    #
    # 一个一个数据集评估
    #
    # 所以必须先关闭 Trainer 自动创建的
    # valid_dataloader。
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Releasing Trainer initial validation dataloader"
    )

    print(
        "=" * 70
    )

    initial_valid_dataloader = getattr(
        trainer,
        "valid_dataloader",
        None
    )

    if initial_valid_dataloader is not None:

        close_lmdb_dataset(
            initial_valid_dataloader
        )

        trainer.valid_dataloader = None

        del initial_valid_dataloader

        gc.collect()

        print(
            "Initial validation dataloader released."
        )

    else:

        print(
            "No initial validation dataloader."
        )

    # ========================================================
    # 14.8 开始逐个数据集评估
    # ========================================================

    results = []

    for index, datadir in enumerate(
        data_dirs,
        start=1
    ):

        metric = evaluate_one_dataset(
            trainer,
            cfg.cfg,
            datadir,
            index
        )

        if metric is not None:

            results.append(
                {
                    "dataset": datadir,
                    "metric": metric_to_dict(
                        metric
                    )
                }
            )

        # ----------------------------------------------------
        # 再次 GC
        # ----------------------------------------------------

        gc.collect()

    # ========================================================
    # 14.9 打印最终结果
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Evaluation Results"
    )

    print(
        "=" * 70
    )

    for result in results:

        print(
            f"\nDataset:"
        )

        print(
            f"  {result['dataset']}"
        )

        print(
            "Metric:"
        )

        print(
            f"  {result['metric']}"
        )

    # ========================================================
    # 14.10 保存结果
    # ========================================================

    output_dir = cfg.cfg[
        "Global"
    ][
        "output_dir"
    ]

    if results:

        # ----------------------------------------------------
        # 每个 benchmark
        # ----------------------------------------------------

        save_results_csv(
            results,
            output_dir
        )

        # ----------------------------------------------------
        # 平均
        # ----------------------------------------------------

        averages = calculate_average(
            results
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "Average Results"
        )

        print(
            "=" * 70
        )

        for key, value in averages.items():

            print(
                f"{key}: {value:.6f}"
            )

        save_average_csv(
            averages,
            output_dir
        )

    else:

        print(
            "\nNo successful evaluation results."
        )

    # ========================================================
    # 14.11 最终释放 Trainer
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Releasing Trainer"
    )

    print(
        "=" * 70
    )

    final_dataloader = getattr(
        trainer,
        "valid_dataloader",
        None
    )

    if final_dataloader is not None:

        close_lmdb_dataset(
            final_dataloader
        )

        trainer.valid_dataloader = None

    del trainer

    gc.collect()

    print(
        "Trainer released."
    )

    # ========================================================
    # 14.12 完成
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Evaluation completed."
    )

    print(
        "=" * 70
    )


# ============================================================
# 15. 程序入口
# ============================================================

if __name__ == "__main__":

    main()
