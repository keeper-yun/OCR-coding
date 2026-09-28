
import os
import sys
import runpy

# ============================================================
# SVTRv2 (GTC) training on the Chinese BCTR benchmark
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = "configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_try.yml"
pretrained_model = "/data/huawei/wzc/OpenOCR/output/rec/training/u14m_filter/svtrv2_rctc_try/stage1/v5.0_epoch20/best.pth"

# Set to True to train from scratch (no pretrained weights).
ALLOW_SCRATCH = False

# ---------------------------------------------------------------------------
# Pin the GPU from inside the script.
# MUST run before any torch import / CUDA API call, otherwise CUDA ignores
# the setting. A value already set in the shell always wins.
# ---------------------------------------------------------------------------
DEFAULT_CUDA_VISIBLE_DEVICES = "0"
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = DEFAULT_CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES_SET_BY_SCRIPT = True
else:
    CUDA_VISIBLE_DEVICES_SET_BY_SCRIPT = False


def check_environment():
    """Fail fast with a clear message instead of crashing silently later."""
    import torch

    n_gpus = torch.cuda.device_count()
    launched_by_torchrun = "WORLD_SIZE" in os.environ

    # 1) torchrun with N processes needs N visible GPUs. If the script
    #    defaulted CUDA_VISIBLE_DEVICES to a single GPU, stop here.
    if launched_by_torchrun and CUDA_VISIBLE_DEVICES_SET_BY_SCRIPT:
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size > 1:
            sys.exit(
                f"[ERROR] Launched with torchrun ({world_size} processes), "
                f"but CUDA_VISIBLE_DEVICES was not set in the shell, so the "
                f"script defaulted it to a single GPU.\n"
                f"  Set it explicitly for multi-GPU training:\n"
                f"  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 "
                f"{os.path.basename(__file__)}"
            )

    # 2) Explicit multi-GPU without torchrun: init_process_group() in the
    #    trainer would fail without the launcher's env vars.
    if n_gpus > 1 and not launched_by_torchrun:
        sys.exit(
            f"[ERROR] {n_gpus} GPUs detected, but this script was not launched "
            f"with torchrun.\n"
            f"  Single GPU : python {os.path.basename(__file__)}\n"
            f"  Multi GPU  : CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun "
            f"--nproc_per_node=4 {os.path.basename(__file__)}"
        )

    # 3) load_ckpt() falls back to "train from scratch" SILENTLY when the
    #    pretrained file is missing, so check it here and fail loudly.
    if not ALLOW_SCRATCH and not os.path.exists(
            os.path.join(PROJECT_ROOT, pretrained_model)):
        sys.exit(
            f"[ERROR] Pretrained model not found: {pretrained_model}\n"
            f"  Train the English first stage first "
            f"(configs/rec/svtrv2/svtrv2_rctc.yml),\n"
            f"  or set ALLOW_SCRATCH = True to train from scratch."
        )

    # 4) Data directories are resolved relative to the OpenOCR root.
    bctr_dir = os.path.join(PROJECT_ROOT, "..", "benchmark_bctr")
    if not os.path.isdir(bctr_dir):
        print(f"[WARNING] ../benchmark_bctr not found at: {bctr_dir}")
        print("          Check the dataset layout in docs/svtrv2.md.")

    char_dict = os.path.join(PROJECT_ROOT, "tools", "utils",
                            "ppocr_keys_v1.txt")
    if not os.path.exists(char_dict):
        print(f"[WARNING] Character dict not found: {char_dict}")


if __name__ == "__main__":

    # Data paths in the config are relative to the working directory.
    os.chdir(PROJECT_ROOT)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    check_environment()

    # Learning rate: the config value 0.00065 is tuned for a total batch of
    # 1024 (4 GPUs x 256). Scale it linearly with the actual GPU count.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    lr = 0.00065 * world_size / 4

    print("=" * 70)
    print("SVTRv2 (GTC) Training on BCTR")
    print("=" * 70)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Config       : {CONFIG_PATH}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"GPUs         : {world_size}")
    print(f"Batch size   : 256 per GPU (total {256 * world_size})")
    print(f"Epochs       : 100")
    print(f"Dataset      : BCTR (scene/web/document/handwriting)")
    print(f"Pretrained   : "
        f"{pretrained_model if not ALLOW_SCRATCH else 'None (from scratch)'}")
    print(f"Learning rate: {lr}")
    print("Eval         : scene_test (during training, every epoch)")
    print("=" * 70)
    print("Starting training...")
    print("=" * 70)

    opts = [f"Optimizer.lr={lr}"]
    if not ALLOW_SCRATCH:
        opts.insert(0, f"Global.pretrained_model={pretrained_model}")

    sys.argv = [
        "tools/train_rec.py",
        "-c", CONFIG_PATH,
        "-o",
    ] + opts

    runpy.run_path(
        os.path.join(PROJECT_ROOT, "tools", "train_rec.py"),
        run_name="__main__"
    )

