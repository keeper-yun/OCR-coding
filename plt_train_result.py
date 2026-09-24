import re
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # 服务器无显示环境也能保存图片; 想在本地弹窗查看就删掉这行
import matplotlib.pyplot as plt

# ================= 输入/输出路径: 改成你自己的绝对路径 =================
# Linux 示例:   LOG_PATH = "/root/paddle_ocr/output/rec/u14m_filter/svtrv2_rctc_try/train.log"
# Windows 示例: LOG_PATH = r"D:\Claude_code\train.log" 
LOG_PATH = r"/data/huawei/wzc/OpenOCR/output/rec/training/u14m_filter/svtrv2_rctc_try/train.log"
OUT_PNG = r"/data/huawei/wzc/OpenOCR/output/rec/training/u14m_filter/svtrv2_rctc_try/train_curve.png"
# =====================================================================


def parse_log(path):
    """解析日志, 返回 (train_rows, eval_rows)
    train_rows: [(global_step, epoch, acc, loss), ...]
    eval_rows:  [(epoch, acc), ...]  # 验证集评估行(如果有)
    """
    train, evals = [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            # 只有同时带 global_step 和 acc 的才是训练/评估行,
            # 像 "best metric, acc: 0" 这种行会被过滤掉
            if "global_step:" not in line or "acc:" not in line:
                continue
            m_ep = re.search(r"epoch: \[(\d+)/(\d+)\]", line)
            m_st = re.search(r"global_step: (\d+)", line)
            m_ac = re.search(r"acc: ([\d.]+)", line)
            if not (m_ep and m_st and m_ac):
                continue
            epoch = int(m_ep.group(1))
            step = int(m_st.group(1))
            acc = float(m_ac.group(1))
            m_lo = re.search(r"loss: ([\d.]+)", line)
            loss = float(m_lo.group(1)) if m_lo else None
            if "eval" in line:
                evals.append((epoch, acc))
            else:
                train.append((step, epoch, acc, loss))
    return train, evals


def avg_per_epoch(train):
    """每个 epoch 的 acc/loss 求平均, x 轴用该 epoch 的平均 global_step"""
    d = defaultdict(list)
    for step, ep, acc, loss in train:
        d[ep].append((step, acc, loss))
    xs, accs, losses = [], [], []
    for ep in sorted(d):  # epoch 从 1 开始连续编号
        rows = d[ep]
        xs.append(sum(r[0] for r in rows) / len(rows))
        accs.append(sum(r[1] for r in rows) / len(rows))
        ls = [r[2] for r in rows if r[2] is not None]
        losses.append(sum(ls) / len(ls) if ls else None)
    return xs, accs, losses


def main():
    if not os.path.exists(LOG_PATH):
        print(f"找不到文件: {LOG_PATH}")
        return

    train, evals = parse_log(LOG_PATH)
    if not train:
        print(f"没有从 {LOG_PATH} 解析到任何训练行, 请检查日志内容")
        return

    steps = [r[0] for r in train]
    raw_acc = [r[2] for r in train]
    ep_x, ep_acc, ep_loss = avg_per_epoch(train)

    # 打印汇总数字, 方便写进报告
    best_i = max(range(len(ep_acc)), key=lambda i: ep_acc[i])
    print(f"共 {len(ep_acc)} 个 epoch, {len(train)} 条训练记录")
    print(f"最佳 epoch 平均 acc: {ep_acc[best_i]:.4f} (epoch {best_i + 1})")
    print(f"最后 epoch 平均 acc: {ep_acc[-1]:.4f}")
    if ep_loss:
        print(f"最低 epoch 平均 loss: {min(ep_loss):.4f}")
    if evals:
        print("验证集 acc (eval 行):", [f"ep{e}: {a:.4f}" for e, a in evals])
    else:
        print("日志中没有 eval 行(验证集可能没配置), 只有训练 acc 曲线")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # 上图: acc — 浅灰细线是原始每10步记录, 蓝线是每 epoch 平均
    ax1.plot(steps, raw_acc, color="#94a3b8", lw=1, alpha=0.6,
            label="raw (every 10 steps)")
    ax1.plot(ep_x, ep_acc, color="#2563eb", lw=2, marker="o", ms=5,
            label="per-epoch mean")
    if evals:  # 验证集 acc 画成红叉
        eval_x = [ep_x[e - 1] for e, _ in evals if 0 < e <= len(ep_x)]
        eval_a = [a for e, a in evals if 0 < e <= len(ep_x)]
        ax1.scatter(eval_x, eval_a, marker="x", s=80, color="#dc2626",
                    zorder=5, label="eval acc")
    ax1.set_ylabel("acc")
    ax1.set_title("SVTRv2 training accuracy")
    ax1.legend()
    ax1.grid(True, ls="--", lw=0.5, alpha=0.4)

    # 下图: loss
    ax2.plot(ep_x, ep_loss, color="#ea580c", lw=2, marker="o", ms=5,
            label="per-epoch mean loss")
    ax2.set_xlabel("global_step")
    ax2.set_ylabel("loss")
    ax2.set_title("training loss")
    ax2.legend()
    ax2.grid(True, ls="--", lw=0.5, alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"图片已保存: {os.path.abspath(OUT_PNG)}")


if __name__ == "__main__":
    main()
