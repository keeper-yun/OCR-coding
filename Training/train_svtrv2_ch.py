# 等价于：
# python tools/train_rec.py -c configs/rec/svtrv2/svtrv2_ch_try.yml

import os
import sys
import runpy

# ============================================================
# SVTRv2 RCTC Training
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = "configs/rec/svtrv2/svtrv2_ch_try.yml"
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
    print("Epochs       : 100")
    print("Batch size   : 128")
    print("Dataset      : BCTR")
    print("Pretrained   : ./openocr_svtrv2_ch.pth")

    print("=" * 70)
    print("Starting training...")
    print("=" * 70)

    sys.argv = [
        "tools/train_rec.py",
        "-c",
        CONFIG_PATH,
    ]

    runpy.run_path(
        os.path.join(PROJECT_ROOT, "tools", "train_rec.py"),
        run_name="__main__"
    )
