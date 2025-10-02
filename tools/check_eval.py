import torch
import os
import numpy as np

def parse_eval_pth(eval_dir="eval"):
    # 遍历 eval 文件夹下所有 .pth
    pth_files = [f for f in os.listdir(eval_dir) if f.endswith(".pth")]
    pth_files.sort()

    for fname in pth_files:
        path = os.path.join(eval_dir, fname)
        print("=" * 60)
        print(f"Loading: {path}")
        data = torch.load(path, map_location="cpu")

        # 打印 keys
        print("Keys:", list(data.keys()))

        # precision/recall 是 numpy 数组
        if "precision" in data:
            prec = data["precision"]  # [IoU, recall, cls, area, maxdet]
            print(f"precision shape: {prec.shape}")
            # 计算平均 AP
            ap = np.mean(prec[prec > -1]) if np.any(prec > -1) else float("nan")
            print(f"  -> mean AP (approx): {ap:.4f}")

        if "recall" in data:
            rec = data["recall"]  # [IoU, cls, area, maxdet]
            print(f"recall shape: {rec.shape}")
            ar = np.mean(rec[rec > -1]) if np.any(rec > -1) else float("nan")
            print(f"  -> mean AR (approx): {ar:.4f}")

        # counts 里可能是 [IoU 阈值数, recall 阈值数, 类别数, area 范围, maxDets]
        if "counts" in data:
            print("counts:", data["counts"])

        if "date" in data:
            print("eval date:", data["date"])

        print()

if __name__ == "__main__":
    parse_eval_pth("/root/autodl-tmp/exps/20250930_231059_CGA_GMS_PARALLEL/eval")
