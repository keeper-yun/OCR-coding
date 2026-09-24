
# 等价于：
# python tools/train_rec.py -c configs/rec/svtrv2/svtrv2_rctc_try.yml

import os
import sys
import runpy

# ============================================================
# SVTRv2 RCTC Mini Training
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = "configs/rec/svtrv2/svtrv2_rctc_try.yml"

# 切换到 OpenOCR 项目根目录
os.chdir(PROJECT_ROOT)

# 保证项目根目录可以导入 tools
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


if __name__ == "__main__":

    print("=" * 70)
    print("SVTRv2 RCTC Training")
    print("=" * 70)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Config       : {CONFIG_PATH}")
    print("Device       : GPU")
    print("Epochs       : 20")
    print("Batch size   : 32")
    print("Dataset      : Union14M-L-LMDB-Filtered")
    print("Pretrained   : None")

    print("=" * 70)
    print("Starting training...")
    print("=" * 70)

    # 等价于：
    #
    # python tools/train_rec.py \
    #     -c configs/rec/svtrv2/svtrv2_rctc_try.yml

    sys.argv = [
        "tools/train_rec.py",
        "-c",
        CONFIG_PATH,
    ]

    runpy.run_path(
        os.path.join(PROJECT_ROOT, "tools", "train_rec.py"),
        run_name="__main__"
    )
