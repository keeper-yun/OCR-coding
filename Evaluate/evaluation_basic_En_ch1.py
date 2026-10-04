import subprocess
from pathlib import Path


# =========================
# 1. 配置
# =========================

config_path = "configs/rec/svtrv2/svtrv2_ch_try.yml"

# "/data/huawei/wzc/OpenOCR/"
    # "openocr_svtrv2_ch.pth"
checkpoint_path = (
    "/data/huawei/wzc/OpenOCR/"
    "output/rec/training/BCTR/"
    "svtrv2_ch_try/v1.0_epoch100/"
    "best.pth"
)

result_path = (
    "/data/huawei/wzc/OpenOCR/"
    "output/rec/training/BCTR/"
    "svtrv2_ch_try/try/"
    "v1.0_valid.txt"
)

# =========================
# 2. 创建结果目录
# =========================

Path(result_path).parent.mkdir(parents=True, exist_ok=True)


# =========================
# 3. Validation 命令
# =========================

cmd = [
    "python",
    # test命令
    # "tools/eval_rec_all_ch.py",
    # validation命令
    "tools/eval_rec.py",
    "-c",
    config_path,
    "-o",
    f"Global.checkpoints={checkpoint_path}",
]


print("=" * 60)
print("开始验证")
print("=" * 60)
print(f"配置文件: {config_path}")
print(f"Checkpoint: {checkpoint_path}")
print(f"结果文件: {result_path}")
print("=" * 60)


# =========================
# 4. 执行 Eval
#    同时显示 + 保存日志
# =========================

with open(result_path, "w", encoding="utf-8") as f:

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        print(line, end="")
        f.write(line)
        f.flush()

    process.wait()


# =========================
# 5. 结果
# =========================

if process.returncode == 0:
    print("\n" + "=" * 60)
    print("验证完成")
    print(f"验证结果已保存到:")
    print(result_path)
    print("=" * 60)
else:
    print("\n" + "=" * 60)
    print("验证失败")
    print(f"返回码: {process.returncode}")
    print("=" * 60)
