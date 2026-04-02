#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
extract_fmriprep_qc.py
用于从 fMRIPrep 的处理结果中提取 6 项结构和功能质量控制 (QC) 指标。
包含：CJV, EFC, WM2MAX, tSNR_mean, MeanFD, MeanStdDVARS

使用环境：需要安装 pandas, numpy, nibabel。如果安装了 mriqc，将调用官方 EFC，否则使用备用实现。
"""

import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import time

# ===== 参数设置 =====
# fMRIPrep 的输出根目录 (包含 sub-* 文件夹)
FMRIPREP_ROOT = "/mnt/sda1/zhangyan/cpac_output/fmriprep_output/OHSU_output"

# 结果保存表
OUTPUT_CSV_PATH = "/mnt/sda1/zhangyan/cpac_output/质量控制分析/fmriprep/OHSU-fmriprep-qc-all.csv"
# ====================


def get_efc(img_data):
    """
    计算 EFC (Entropy Focus Criterion)。
    如果能调用 mriqc.qc.anatomical.efc 则调用，否则使用 MRIQC 标准的 Shannon 熵退化公式。
    """
    # 消除由于插值可能产生的微小负值，防止 log 计算报错和产生 warning
    img_data_clean = np.clip(img_data, a_min=0, a_max=None)
    
    try:
        from mriqc.qc.anatomical import efc
        # 临时忽略某些 RuntimeWarning
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(efc(img_data_clean))
    except ImportError:
        x = img_data_clean.flatten()
        # 排除完全的 NaN / Inf
        x = x[np.isfinite(x)]
        x_max = np.sqrt(np.sum(x**2))
        if x_max == 0:
            return np.nan
        
        p = x / x_max
        p = p[p > 0]  # 避免 log(0)
        E = -np.sum(p * np.log(p))
        N = x.size
        # 归一化参数 (最大熵)
        E_max = -np.sqrt(N) * np.log(1.0 / np.sqrt(N))
        
        return float(E / E_max) if E_max != 0 else np.nan


def get_wm2max(img_data, wm_mask, brain_mask):
    """
    计算 WM2MAX (WM 均值 / 99.95% 高分位强度)。
    """
    wm = img_data[wm_mask > 0.5]
    if wm.size < 10:
        return float('nan')
    mu_wm = float(np.mean(wm))
    
    # 提取脑内非 0 有效像素作为全脑分布
    x = img_data[brain_mask > 0]
    x = x[np.isfinite(x)]
    x = x[x != 0]
    
    if x.size == 0:
        return float('nan')
        
    p9995 = float(np.percentile(x, 99.95))
    return float(mu_wm / p9995) if p9995 > 0 else float('nan')


def get_cjv(img_data, wm_mask, gm_mask):
    """计算 CJV (GM/WM joint variation)"""
    wm = img_data[wm_mask > 0.5]
    gm = img_data[gm_mask > 0.5]
    if wm.size < 10 or gm.size < 10:
        return float('nan')
    
    mu_wm = float(np.mean(wm))
    mu_gm = float(np.mean(gm))
    sd_wm = float(np.std(wm, ddof=1))
    sd_gm = float(np.std(gm, ddof=1))
    
    denom = abs(mu_wm - mu_gm)
    if denom > 0:
        return float((sd_wm + sd_gm) / denom)
    return float('nan')


def get_tsnr(bold_path, mask_path):
    """
    根据给定的 BOLD 和 mask 计算 tSNR_mean。
    4D 的时间均值 / 时间标准差，最后取 mask 范围内的空间平均。
    """
    bold_data = nib.load(bold_path).get_fdata(dtype=np.float32)
    mask_data = nib.load(mask_path).get_fdata(dtype=np.float32)

    # 在时间轴上求均值和标准差 (通常 BOLD 序列的 shape 是 X, Y, Z, T，所以 axis=3)
    # 若 BOLD 是 3D 的 (极小概率，或者发生了错误降维)，这会抛错
    if len(bold_data.shape) < 4:
        print(f"    [警告] BOLD 图像只有一个时间点或维度异常！跳过 tSNR 计算。")
        return np.nan

    mu = np.mean(bold_data, axis=3)
    sigma = np.std(bold_data, axis=3, ddof=1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        tsnr = np.where(sigma > 0, mu / sigma, 0)
    
    # 取 mask 内部的 tSNR 求均值
    valid_tsnr = tsnr[mask_data > 0]
    
    if valid_tsnr.size == 0:
        return float('nan')
        
    return float(np.mean(valid_tsnr))


def load_nii(path):
    return nib.load(path).get_fdata(dtype=np.float32)

def main():
    start_time = time.time()
    
    if not os.path.exists(FMRIPREP_ROOT):
        print(f"数据根目录 {FMRIPREP_ROOT} 不存在，请检查！")
        return
        
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    
    # 只查 sub- 开头的文件夹
    subjects = [d for d in os.listdir(FMRIPREP_ROOT) if d.startswith("sub-")]
    subjects = [d for d in subjects if os.path.isdir(os.path.join(FMRIPREP_ROOT, d))]
    subjects.sort()
    
    print(f"发现 {len(subjects)} 个被试。准备开始处理...\n")
    
    results = []
    
    for i, sub in enumerate(subjects, 1):
        print(f"[{i}/{len(subjects)}] 处理被试: {sub}")
        sub_dir = os.path.join(FMRIPREP_ROOT, sub)
        
        # 预设填充值
        metrics = {
            "SubjectID": sub,
            "MeanFD": np.nan,
            "MeanStdDVARS": np.nan,
            "tSNR_mean": np.nan,
            "CJV": np.nan,
            "EFC": np.nan,
            "WM2MAX": np.nan
        }
        
        # ================================
        # 1. 计算结构像指标 (Anat)
        # ================================
        t1w_files = glob.glob(os.path.join(sub_dir, "**", "anat", "*_desc-preproc_T1w.nii.gz"), recursive=True)
        mask_files = glob.glob(os.path.join(sub_dir, "**", "anat", "*_desc-brain_mask.nii.gz"), recursive=True)
        gm_files = glob.glob(os.path.join(sub_dir, "**", "anat", "*_label-GM_probseg.nii.gz"), recursive=True)
        wm_files = glob.glob(os.path.join(sub_dir, "**", "anat", "*_label-WM_probseg.nii.gz"), recursive=True)
        
        if t1w_files and mask_files and gm_files and wm_files:
            try:
                t1w_img = load_nii(t1w_files[0])
                brain_mask = load_nii(mask_files[0]) > 0
                gm_prob = load_nii(gm_files[0])
                wm_prob = load_nii(wm_files[0])
                
                # 生成供计算用的二值 mask
                gm_mask = (gm_prob > 0.5) & brain_mask
                wm_mask = (wm_prob > 0.5) & brain_mask
                
                metrics["EFC"] = get_efc(t1w_img)
                metrics["WM2MAX"] = get_wm2max(t1w_img, wm_mask, brain_mask)
                metrics["CJV"] = get_cjv(t1w_img, wm_mask, gm_mask)
            except Exception as e:
                print(f"    [结构像报错] {sub} 计算异常: {e}")
        else:
            print(f"    [结构像缺失] {sub} anat 目录下文件不全，无法计算 CJV/EFC/WM2MAX")
        
        # ================================
        # 2. 计算功能像指标 (Func)
        # ================================
        # 找到空间为 T1w 的 preproc_bold (默认拿碰到的第一个，通常只有一个 run)
        bold_files = glob.glob(os.path.join(sub_dir, "**", "func", "*_space-T1w_desc-preproc_bold.nii.gz"), recursive=True)
        
        if bold_files:
            bold_path = bold_files[0]
            # 利用前缀拼凑其他依赖文件名为
            # 例：sub-01_task-rest_space-T1w_desc-preproc_bold.nii.gz
            # prefix: sub-01_task-rest_space-T1w
            prefix = bold_path.replace("_desc-preproc_bold.nii.gz", "")
            
            mask_path = prefix + "_desc-brain_mask.nii.gz"
            
            # confounds表通常不带 space，如 sub-01_task-rest_desc-confounds_timeseries.tsv
            run_prefix = prefix.split("_space-")[0]
            confounds_path = run_prefix + "_desc-confounds_timeseries.tsv"
            
            # tSNR 计算
            if os.path.exists(mask_path):
                try:
                    metrics["tSNR_mean"] = get_tsnr(bold_path, mask_path)
                except Exception as e:
                    print(f"    [tSNR报错] {sub} 计算异常: {e}")
            else:
                print(f"    [tSNR缺失] 找不到对应的 func mask: {mask_path}")
            
            # FD / DVARS 提取
            if os.path.exists(confounds_path):
                try:
                    df_conf = pd.read_csv(confounds_path, sep="\t")
                    if "framewise_displacement" in df_conf.columns:
                        fd_col = pd.to_numeric(df_conf["framewise_displacement"], errors='coerce').dropna()
                        metrics["MeanFD"] = fd_col.mean()
                    if "std_dvars" in df_conf.columns:
                        dvars_col = pd.to_numeric(df_conf["std_dvars"], errors='coerce').dropna()
                        metrics["MeanStdDVARS"] = dvars_col.mean()
                except Exception as e:
                    print(f"    [参数表报错] {sub} TSV 解析异常: {e}")
            else:
                print(f"    [参数表缺失] 找不到 confounds TSV: {confounds_path}")
        else:
             print(f"    [功能像缺失] {sub} 找不到 space-T1w 的 BOLD 序列。")
                 
        results.append(metrics)
        
    # 保存结果
    df_results = pd.DataFrame(results)
    
    # 强制将列按照我们的预设顺序排列
    cols = ["SubjectID", "MeanFD", "MeanStdDVARS", "tSNR_mean", "CJV", "EFC", "WM2MAX"]
    df_results = df_results[cols]
    
    df_results.to_csv(OUTPUT_CSV_PATH, index=False)
    
    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"处理完成！总共耗时: {elapsed/60:.2f} 分钟。")
    print(f"结果已保存至: {OUTPUT_CSV_PATH}")
    print("-" * 50)

if __name__ == "__main__":
    main()
