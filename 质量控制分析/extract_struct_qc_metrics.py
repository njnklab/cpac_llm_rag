#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

# ===== 手动在这里设置参数 =====
# CPAC 数据集结果根目录（结构像对应的数据集）
# 你可以按需要修改为别的数据集路径
BASE_OUTPUT_DIR = "/mnt/sda1/zhangyan/cpac_output/output/ds002748(default)"

# 汇总 CSV 输出路径（结构像 QC 指标表）
OUTPUT_CSV_PATH = "/mnt/sda1/zhangyan/cpac_output/质量控制分析/ds002748-default_struct.csv"

# CPAC pipeline 名称、session（如有变化在这里改）
PIPELINE_NAME = "pipeline_cpac-default-pipeline"
SESSION = "ses-1"


def find_subjects(base_output_dir):
    """在 base_output_dir 下自动发现所有 sub-* 目录。"""
    subjects = []
    for name in os.listdir(base_output_dir):
        sub_path = os.path.join(base_output_dir, name)
        if name.startswith("sub-") and os.path.isdir(sub_path):
            subjects.append(name)
    subjects.sort()
    return subjects


def load_nii(path: Path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return data


def efc(img):
    """Entropy Focus Criterion (EFC)，使用 MRIQC 的定义近似实现。"""
    x = np.asarray(img, dtype=np.float64)
    x = np.abs(x)
    x = x[x > 0]
    if x.size == 0:
        return float("nan")
    x_max = np.sqrt(np.sum(x * x))
    p = x / x_max
    E = -np.sum(p * np.log(p))
    N = x.size
    # 归一化项按照 MRIQC 文档的近似写法
    return float((N / np.sqrt(N) * np.log((np.sqrt(N)) ** -1)) * E)


def cjv(img, wm_mask, gm_mask):
    """CJV = (sigma_WM + sigma_GM) / |mu_WM - mu_GM|"""
    wm = img[wm_mask > 0.5]
    gm = img[gm_mask > 0.5]
    if wm.size < 10 or gm.size < 10:
        return float("nan")
    mu_wm, mu_gm = float(wm.mean()), float(gm.mean())
    sd_wm, sd_gm = float(wm.std(ddof=1)), float(gm.std(ddof=1))
    denom = abs(mu_wm - mu_gm)
    return float((sd_wm + sd_gm) / denom) if denom > 0 else float("nan")


def wm2max(img, wm_mask):
    """WM2MAX = mu_WM / P99.95(X)"""
    wm = img[wm_mask > 0.5]
    if wm.size < 10:
        return float("nan")
    mu_wm = float(wm.mean())
    x = img[np.isfinite(img)]
    x = x[x != 0]  # 去掉背景 0，避免分位数被 0 污染
    if x.size == 0:
        return float("nan")
    p9995 = float(np.percentile(x, 99.95))
    return float(mu_wm / p9995) if p9995 > 0 else float("nan")


def pick_one_or_none(directory: Path, pattern: str, sub_id: str, label: str):
    hits = sorted(directory.glob(pattern))
    if not hits:
        print(f"[{sub_id}] 未找到 {label}: 模式 {pattern} 在 {directory}")
        return None
    if len(hits) > 1:
        names = [h.name for h in hits]
        print(f"[{sub_id}] 找到多个 {label}, 使用 {hits[0].name}: {names}")
    return hits[0]


def extract_struct_metrics_for_subject(base_output_dir, sub_id, pipeline_name, session):
    """从单个被试的 anat 目录中提取 CJV, EFC, WM2MAX 等结构像指标。"""
    anat_dir = (
        Path(base_output_dir)
        / sub_id
        / "output"
        / pipeline_name
        / sub_id
        / session
        / "anat"
    )

    if not anat_dir.is_dir():
        print(f"[{sub_id}] 未找到 anat 目录: {anat_dir}")
        return {
            "SubjectID": sub_id,
            "CJV": np.nan,
            "EFC": np.nan,
            "WM2MAX": np.nan,
            "GM_mean": np.nan,
            "WM_mean": np.nan,
        }

    head_t1_path = pick_one_or_none(
        anat_dir,
        "*_desc-head_T1w.nii.gz",
        sub_id,
        "EFC 用的 desc-head_T1w",
    )
    preproc_t1_path = pick_one_or_none(
        anat_dir,
        "*_desc-preproc_T1w.nii.gz",
        sub_id,
        "CJV/WM2MAX 用的 desc-preproc_T1w",
    )
    gm_mask_path = pick_one_or_none(
        anat_dir,
        "*_label-GM_desc-preproc_mask.nii.gz",
        sub_id,
        "GM mask",
    )
    wm_mask_path = pick_one_or_none(
        anat_dir,
        "*_label-WM_desc-preproc_mask.nii.gz",
        sub_id,
        "WM mask",
    )

    # 默认先全部设为 NaN
    CJV = np.nan
    EFC = np.nan
    WM2MAX = np.nan
    gm_mean = np.nan
    wm_mean = np.nan

    try:
        if head_t1_path is not None:
            head_img = load_nii(head_t1_path)
            EFC = efc(head_img)
    except Exception as e:
        print(f"[{sub_id}] 计算 EFC 失败: {e}")

    try:
        if (
            preproc_t1_path is not None
            and gm_mask_path is not None
            and wm_mask_path is not None
        ):
            t1_img = load_nii(preproc_t1_path)
            gm = load_nii(gm_mask_path)
            wm = load_nii(wm_mask_path)

            CJV = cjv(t1_img, wm, gm)
            WM2MAX = wm2max(t1_img, wm)

            gm_vals = t1_img[gm > 0.5]
            wm_vals = t1_img[wm > 0.5]
            if gm_vals.size > 0:
                gm_mean = float(gm_vals.mean())
            if wm_vals.size > 0:
                wm_mean = float(wm_vals.mean())
    except Exception as e:
        print(f"[{sub_id}] 计算 CJV/WM2MAX 或 GM/WM 均值失败: {e}")

    return {
        "SubjectID": sub_id,
        "CJV": CJV,
        "EFC": EFC,
        "WM2MAX": WM2MAX,
        "GM_mean": gm_mean,
        "WM_mean": wm_mean,
    }


def main():
    base_output_dir = BASE_OUTPUT_DIR
    output_csv_path = OUTPUT_CSV_PATH

    print(f"Base directory: {base_output_dir}")
    print(f"Output CSV   : {output_csv_path}")

    subjects = find_subjects(base_output_dir)
    print(f"发现被试数量: {len(subjects)}")
    if not subjects:
        print("未发现任何 sub-* 目录，请检查 BASE_OUTPUT_DIR 是否正确。")
        return

    all_subjects_data = []

    for sub_id in subjects:
        print(f"处理: {sub_id}")
        metrics = extract_struct_metrics_for_subject(
            base_output_dir=base_output_dir,
            sub_id=sub_id,
            pipeline_name=PIPELINE_NAME,
            session=SESSION,
        )
        all_subjects_data.append(metrics)

    results_df = pd.DataFrame(all_subjects_data)
    results_df.to_csv(output_csv_path, index=False)

    print("\n--- 结构像指标提取完成 ---")
    print(results_df)
    print(f"\n结果已保存到: {output_csv_path}")


if __name__ == "__main__":
    main()
