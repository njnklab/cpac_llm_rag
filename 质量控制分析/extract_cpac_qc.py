#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
import glob
from mriqc.qc.anatomical import efc

# ===== 手动在这里设置参数 =====
# CPAC 数据集结果根目录（功能像和结构像都从这里找）
BASE_OUTPUT_DIR = "/mnt/sda1/zhangyan/cpac_output/output/ds002748-gpt-all"

# 提取出的 QC 指标最终汇总表保存路径
OUTPUT_CSV_PATH = "/mnt/sda1/zhangyan/cpac_output/质量控制分析/cpac-llm/ds002748-llm-qc-all.csv"

# CPAC pipeline 名称、session、task（如有变化在这里改）
PIPELINE_NAME = "pipeline_cpac-default-pipeline"
SESSIONS = ["ses-1", "ses-2"] 
TASK = None  # 若不确定 task 名称，保持为 None 即可


# ========= 公共工具函数 =========

def find_subjects(base_output_dir):
    """在 base_output_dir 下自动发现所有 sub-* 目录。"""
    subjects = []
    for name in os.listdir(base_output_dir):
        sub_path = os.path.join(base_output_dir, name)
        if name.startswith("sub-") and os.path.isdir(sub_path):
            subjects.append(name)
    subjects.sort()
    return subjects

def pick_session_for_subject(base_output_dir, sub_id, pipeline_name, sessions):
    """
    在该 sub_id 的 CPAC 输出目录下，按 sessions 列表顺序，返回第一个存在的 session 名称。
    若都不存在，返回 None。
    """
    base = (
        Path(base_output_dir)
        / sub_id
        / "output"
        / pipeline_name
        / sub_id
    )

    for ses in sessions:
        if (base / ses).is_dir():
            return ses

    return None

def compute_dvars_from_bold(bold_path):
    """
    从 BOLD 数据计算 DVARS（简化版）
    与 extract_deepprep_qc.py 的逻辑保持一致
    """
    try:
        bold_data = nib.load(bold_path).get_fdata(dtype=np.float32)
        
        if len(bold_data.shape) < 4:
            return None
        
        # 将 4D 数据 reshape 为 (n_voxels, n_timepoints)
        n_voxels = np.prod(bold_data.shape[:3])
        n_timepoints = bold_data.shape[3]
        data_2d = bold_data.reshape(n_voxels, n_timepoints)
        
        # 计算每个时间点的全局信号（所有体素的均值）
        global_signal = np.mean(data_2d, axis=0)
        
        # 计算相邻时间点的差值
        diff = np.diff(global_signal)
        
        # DVARS（第一个时间点设为 0）
        dvars = np.insert(np.abs(diff), 0, 0)
        
        # 标准化（近似 std_dvars）
        std_dvars = dvars / np.std(dvars) if np.std(dvars) > 0 else dvars
        
        return std_dvars
    except Exception as e:
        print(f"    [DVARS计算错误] 无法从 {bold_path} 计算 DVARS: {e}")
        return None

# ========= 功能像 QC 指标提取 =========

def extract_func_metrics_for_subject(base_output_dir, sub_id, pipeline_name, session, task):
    """从单个被试的 func 目录中提取 MeanFD_Power, MeanDVARS, boldSnr。"""
    func_dir = os.path.join(
        base_output_dir,
        sub_id,
        "output",
        pipeline_name,
        sub_id,
        session,
        "func",
    )

    motion_tsv_path = None

    if os.path.isdir(func_dir):
        # 如果指定了 task，优先按 task 构造文件名
        if task:
            candidate = os.path.join(
                func_dir,
                f"{sub_id}_{session}_task-{task}_desc-summary_motion.tsv",
            )
            if os.path.exists(candidate):
                motion_tsv_path = candidate
            else:
                print(
                    f"[{sub_id}] 按 task='{task}' 未找到 motion 文件, 尝试自动搜索 *_desc-summary_motion.tsv"
                )

        # 若没有指定 task，或按 task 没找到文件，则在 func 目录中自动搜索
        if motion_tsv_path is None:
            try:
                candidates = []
                prefix = f"{sub_id}_{session}"
                for filename in os.listdir(func_dir):
                    if filename.endswith("_desc-summary_motion.tsv") and filename.startswith(
                        prefix
                    ):
                        candidates.append(filename)

                if len(candidates) == 1:
                    motion_tsv_path = os.path.join(func_dir, candidates[0])
                elif len(candidates) > 1:
                    candidates.sort()
                    motion_tsv_path = os.path.join(func_dir, candidates[0])
                    print(
                        f"[{sub_id}] 找到多个 motion 文件, 使用 {candidates[0]}: {candidates}"
                    )
                else:
                    print(
                        f"[{sub_id}] func 目录下未找到 *_desc-summary_motion.tsv: {func_dir}"
                    )
            except Exception as e:
                print(f"[{sub_id}] 搜索 motion 文件时出错: {e}")
    else:
        print(f"[{sub_id}] 未找到 func 目录: {func_dir}")

    # 查找 preproc_bold 用于计算 DVARS (与 DeepPrep 一致)
    preproc_bold_path = None
    if os.path.isdir(func_dir):
        # 优先寻找 reg-default_desc-preproc_bold.nii.gz
        candidates = glob.glob(os.path.join(func_dir, "*_reg-default_desc-preproc_bold.nii.gz"))
        if not candidates:
            # 退而求其次，寻找任何 preproc_bold.nii.gz
            candidates = glob.glob(os.path.join(func_dir, "*_desc-preproc_bold.nii.gz"))
        
        if candidates:
            preproc_bold_path = candidates[0]

    mean_fd = None
    mean_dvars = None
    bold_snr = None

    # 读取 motion.tsv (获取 FD)
    if motion_tsv_path and os.path.exists(motion_tsv_path):
        try:
            motion_df = pd.read_csv(motion_tsv_path, sep="\t")
            if "MeanFD_Power" in motion_df.columns:
                mean_fd = motion_df["MeanFD_Power"].iloc[0]
        except Exception as e:
            print(f"[{sub_id}] 读取 {motion_tsv_path} 失败: {e}")

    # 从 BOLD 数据计算 标准化 DVARS (与 DeepPrep 一致)
    if preproc_bold_path:
        try:
            dvars_values = compute_dvars_from_bold(preproc_bold_path)
            if dvars_values is not None:
                mean_dvars = np.mean(dvars_values)
                print(f"[{sub_id}] 从 BOLD 计算 Standardized DVARS 完成: {mean_dvars:.4f}")
        except Exception as e:
            print(f"[{sub_id}] 计算 DVARS 失败: {e}")
    else:
        print(f"[{sub_id}] 未找到 preproc_bold 文件，无法计算 DVARS")

    # 查找 boldSnr txt
    if os.path.isdir(func_dir):
        try:
            snr_file_found = False
            for filename in os.listdir(func_dir):
                if "boldSnr_quality" in filename and filename.endswith(".txt"):
                    snr_txt_path = os.path.join(func_dir, filename)
                    with open(snr_txt_path, "r") as f:
                        content = f.read().strip()
                    try:
                        bold_snr = float(content)
                    except ValueError:
                        print(
                            f"[{sub_id}] 无法解析 boldSnr 数值: {snr_txt_path} 内容='{content}'"
                        )
                    snr_file_found = True
                    break
            if not snr_file_found:
                print(f"[{sub_id}] func 目录下未找到 boldSnr_quality*.txt: {func_dir}")
        except Exception as e:
            print(f"[{sub_id}] 处理 SNR 文件出错: {e}")
    else:
        print(f"[{sub_id}] 未找到 func 目录: {func_dir}")

    return {
        "SubjectID": sub_id,
        "Session": session,
        "MeanFD_Power": mean_fd,
        "MeanDVARS": mean_dvars,
        "boldSnr": bold_snr,
    }


# ========= 结构像 QC 指标提取 =========

def load_nii(path: Path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return data


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
            "Session": session,
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
        "Session": session,
        "CJV": CJV,
        "EFC": EFC,
        "WM2MAX": WM2MAX,
        "GM_mean": gm_mean,
        "WM_mean": wm_mean,
    }



# ========= 主流程 =========

def main():
    base_output_dir = BASE_OUTPUT_DIR

    print(f"Base directory: {base_output_dir}")
    print(f"Output CSV : {OUTPUT_CSV_PATH}")

    subjects = find_subjects(base_output_dir)
    print(f"发现被试数量: {len(subjects)}")
    if not subjects:
        print("未发现任何 sub-* 目录，请检查 BASE_OUTPUT_DIR 是否正确。")
        return

    func_data = []
    struct_data = []

    for sub_id in subjects:
        ses = pick_session_for_subject(
            base_output_dir=base_output_dir,
            sub_id=sub_id,
            pipeline_name=PIPELINE_NAME,
            sessions=SESSIONS,
        )

        if ses is None:
            print(f"[{sub_id}] 未找到任何 session（尝试了 {SESSIONS}），跳过该被试")
            continue

        print(f"处理: {sub_id}  (session={ses})")

        func_metrics = extract_func_metrics_for_subject(
            base_output_dir=base_output_dir,
            sub_id=sub_id,
            pipeline_name=PIPELINE_NAME,
            session=ses,
            task=TASK,
        )
        struct_metrics = extract_struct_metrics_for_subject(
            base_output_dir=base_output_dir,
            sub_id=sub_id,
            pipeline_name=PIPELINE_NAME,
            session=ses,
        )

        func_data.append(func_metrics)
        struct_data.append(struct_metrics)

    func_df = pd.DataFrame(func_data)
    struct_df = pd.DataFrame(struct_data)

    # 按 SubjectID 合并两个表
    merged_df = pd.merge(func_df, struct_df, on="SubjectID", how="outer")
    
    # 若目录不存在自动创建
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    
    merged_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("\n--- 合并后的总表 ---")
    print(merged_df)
    print(f"\n合并总表 CSV 已保存到: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
