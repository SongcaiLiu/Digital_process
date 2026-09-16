"""Evaluate processed sessions without mixing adjacent frames across subjects."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from rgbd_project_core import FEATURE_NAMES, OneClassModel


def main():
    p = argparse.ArgumentParser(description="评估RGB-D真人确认模型")
    p.add_argument("--model", type=Path, default=Path("models/liveness_oneclass.json"))
    p.add_argument("--features", type=Path, default=Path("processed/real_features.csv"))
    args = p.parse_args()
    model = OneClassModel.load(args.model)
    with args.features.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    x = np.array([[float(r[n]) for n in FEATURE_NAMES] for r in rows])
    scores = model.distances(x); predictions = scores <= model.threshold
    groups = defaultdict(list)
    for row, score, pred in zip(rows, scores, predictions):
        groups[(row["subject"], row["session"])].append((float(score), bool(pred)))
    print(f"模型阈值：{model.threshold:.3f}")
    for (subject, session), items in sorted(groups.items()):
        print(f"{subject}/{session}: 真人通过率 {np.mean([p for _,p in items]):.1%}，距离中位数 {np.median([s for s,_ in items]):.2f}")
    print(f"总计 {len(rows)} 帧，真人通过率 {predictions.mean():.1%}")
    print("当前只有真人数据，因此这里不能计算照片/屏幕攻击拦截率。")


if __name__ == "__main__":
    main()
