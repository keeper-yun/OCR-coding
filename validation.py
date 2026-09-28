
import subprocess
from pathlib import Path

# =========================
# 1. 配置
# =========================

OPENOCR_ROOT = Path("/data/huawei/wzc/OpenOCR")

CONFIG_PATH = "configs/rec/svtrv2/svtrv2_ch_try.yml"

CHECKPOINT_PATH = (
    OPENOCR_ROOT
    / "/data/huawei/wzc/OpenOCR/openocr_svtrv2_ch.pth"
)

RESULT_PATH = "/data/huawei/wzc/OpenOCR/output/rec/training/BCTR/svtrv2_ch_try/v1.0_epoch100/v1.0_valid.txt"

# =========================
# 2. Pre-flight checks
# =========================

errors = []
if not CHECKPOINT_PATH.exists():
    errors.append(f"checkpoint not found: {CHECKPOINT_PATH}")
if not (OPENOCR_ROOT / CONFIG_PATH).exists():
    errors.append(f"config not found: {OPENOCR_ROOT / CONFIG_PATH}")
if not (OPENOCR_ROOT / "tools/eval_rec_all_ch.py").exists():
    errors.append("eval_rec_all_ch.py not found under tools/")
if not (OPENOCR_ROOT.parent / "benchmark_bctr").is_dir():
    print(f"[WARNING] benchmark_bctr not found: "
        f"{OPENOCR_ROOT.parent / 'benchmark_bctr'}")

if errors:
    print("[ERROR] Abort before launch:")
    for e in errors:
        print(f"  - {e}")
    raise SystemExit(1)

RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

# =========================
# 3. Validation 命令
# =========================

cmd = [
    "python", "-u",
    "tools/eval_rec_all_ch.py",
    # "tools/eval_rec.py"
    "-c", CONFIG_PATH,
    "-o", f"Global.checkpoints={CHECKPOINT_PATH}",
]

print("=" * 60)
print("开始验证")
print("=" * 60)
print(f"配置文件: {CONFIG_PATH}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"结果文件: {RESULT_PATH}")
print("=" * 60)

# =========================
# 4. 执行 Eval
#    同时显示 + 保存日志
# =========================

with open(RESULT_PATH, "w", encoding="utf-8") as f:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=OPENOCR_ROOT,
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
    print(f"验证结果已保存到: {RESULT_PATH}")
    print(f"详细 CSV 在配置的 output_dir 下: "
        f"<output_dir>/svtrv2_smtr_gtc_rctc_ch_eval_all_ch_length_ratio.csv")
    print("=" * 60)
else:
    print("\n" + "=" * 60)
    print("验证失败")
    print(f"返回码: {process.returncode}")
    print("=" * 60)




# import subprocess
# from pathlib import Path


# # =========================
# # 1. 配置
# # =========================

# config_path = " configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_ch.yml"

# # "/data/huawei/wzc/OpenOCR/"
#     # "openocr_svtrv2_ch.pth"
# checkpoint_path = (
#     "/data/huawei/wzc/OpenOCR/"
#     "output/rec/training/BCTR/"
#     "svtrv2_ch_try/v1.0_epoch100/"
#     "best.pth"
# )

# result_path = (
#     "/data/huawei/wzc/OpenOCR/"
#     "output/rec/training/BCTR/"
#     "svtrv2_ch_try/v1.0_epoch100/"
#     "v1.0_valid.txt"
# )

# # =========================
# # 2. 创建结果目录
# # =========================

# Path(result_path).parent.mkdir(parents=True, exist_ok=True)


# # =========================
# # 3. Validation 命令
# # =========================

# cmd = [
#     "python",
#     " tools/eval_rec_all_ch.py",
#     "-c",
#     config_path,
#     "-o",
#     f"Global.checkpoints={checkpoint_path}",
# ]


# print("=" * 60)
# print("开始验证")
# print("=" * 60)
# print(f"配置文件: {config_path}")
# print(f"Checkpoint: {checkpoint_path}")
# print(f"结果文件: {result_path}")
# print("=" * 60)


# # =========================
# # 4. 执行 Eval
# #    同时显示 + 保存日志
# # =========================

# with open(result_path, "w", encoding="utf-8") as f:

#     process = subprocess.Popen(
#         cmd,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.STDOUT,
#         text=True,
#         bufsize=1,
#     )

#     for line in process.stdout:
#         print(line, end="")
#         f.write(line)
#         f.flush()

#     process.wait()


# # =========================
# # 5. 结果
# # =========================

# if process.returncode == 0:
#     print("\n" + "=" * 60)
#     print("验证完成")
#     print(f"验证结果已保存到:")
#     print(result_path)
#     print("=" * 60)
# else:
#     print("\n" + "=" * 60)
#     print("验证失败")
#     print(f"返回码: {process.returncode}")
#     print("=" * 60)
