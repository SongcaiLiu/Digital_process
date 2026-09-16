"""Train and subject-wise validate a real-only RGB-D liveness model."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from rgbd_project_core import FEATURE_NAMES, fit_one_class


def load_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    x = np.array([[float(r[n]) for n in FEATURE_NAMES] for r in rows], dtype=np.float64)
    return rows, x


def balanced_indices(rows, max_per_session: int):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["subject"], r["session"])].append(i)
    result = []
    for indices in groups.values():
        if len(indices) <= max_per_session:
            result.extend(indices)
        else:
            positions = np.linspace(0, len(indices)-1, max_per_session).round().astype(int)
            result.extend(indices[position] for position in positions)
    return np.array(sorted(set(result)), dtype=int)


def main():
    p = argparse.ArgumentParser(description="只使用真人数据训练三维人脸单类模型")
    p.add_argument("--features", type=Path, default=Path("processed/real_features.csv"))
    p.add_argument("--output", type=Path, default=Path("models/liveness_oneclass.json"))
    p.add_argument("--max-per-session", type=int, default=40)
    p.add_argument("--real-acceptance", type=float, default=.95)
    args = p.parse_args()
    rows, all_x = load_rows(args.features)
    idx = balanced_indices(rows, args.max_per_session)
    rows = [rows[i] for i in idx]; x = all_x[idx]
    subjects = sorted({r["subject"] for r in rows})
    if len(x) < 30:
        raise SystemExit("有效样本少于30帧，暂时无法稳定训练")

    heldout_scores = []
    print(f"训练样本：{len(x)}；人员：{len(subjects)}（按场次限帧，避免相邻帧占比过高）")
    if len(subjects) >= 2:
        for subject in subjects:
            train_mask = np.array([r["subject"] != subject for r in rows])
            test_mask = ~train_mask
            fold = fit_one_class(x[train_mask])
            scores = fold.distances(x[test_mask])
            heldout_scores.extend(scores.tolist())
            print(f"留出人员 {subject}: {test_mask.sum()} 帧，距离中位数 {np.median(scores):.2f}")
    # With only two people each fold contains one identity, so its raw distance
    # scale is too unstable to set a useful deployment threshold.  Keep it as a
    # diagnostic and calibrate the prototype on all real training samples.
    use_heldout_threshold = len(subjects) >= 5
    model = fit_one_class(x, args.real_acceptance)
    if use_heldout_threshold and heldout_scores:
        model.threshold = max(float(np.quantile(heldout_scores, args.real_acceptance)), 1e-6)
    train_accept = float(model.predict(x).mean())
    model.metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "training_source": str(args.features.resolve()),
        "subjects": subjects, "sample_count": len(x),
        "threshold_calibration": "leave-one-subject-out real scores" if use_heldout_threshold else "training scores (fewer than 5 subjects)",
        "target_real_acceptance": args.real_acceptance,
        "training_acceptance": train_accept,
        "warning": "prototype model; attack data has not been used or evaluated",
    }
    model.save(args.output)
    print(f"阈值：{model.threshold:.3f}；训练集真人通过率：{train_accept:.1%}")
    print(f"模型：{args.output.resolve()}")


if __name__ == "__main__":
    main()
