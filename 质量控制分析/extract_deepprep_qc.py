#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
extract_deepprep_qc_fixed.py
用于从 DeepPrep 的处理结果中提取 6 项结构和功能质量控制 (QC) 指标。
修复版：从 FSL mcflirt 的 .par 文件计算 FD，而不是读取 confounds TSV
"""

import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import time

# ===== 参数设置 =====
DEEPPREP_BOLD_ROOT = "/mnt/sda1/zhangyan/cpac_output/deepprep_output/OHSU_output/BOLD"
OUTPUT_CSV_PATH = "/mnt/sda1/zhangyan/cpac_output/质量控制分析/deepprep/OHSU-deepprep-qc-all.csv"
# ====================


def get_efc(img_data):
    """计算 EFC (Entropy Focus Criterion)"""
    img_data_clean = np.clip(img_data, a_min=0, a_max=None)
    
    try:
        from mriqc.qc.anatomical import efc
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(efc(img_data_clean))
    except ImportError:
        x = img_data_clean.flatten()
        x = x[np.isfinite(x)]
        x_max = np.sqrt(np.sum(x**2))
        if x_max == 0:
            return np.nan
        
        p = x / x_max
        p = p[p > 0]
        E = -np.sum(p * np.log(p))
        N = x.size
        E_max = -np.sqrt(N) * np.log(1.0 / np.sqrt(N))
        
        return float(E / E_max) if E_max != 0 else np.nan


def get_wm2max(img_data, wm_mask, brain_mask):
    """计算 WM2MAX"""
    wm = img_data[wm_mask > 0.5]
    if wm.size < 10:
        return float('nan')
    mu_wm = float(np.mean(wm))
    
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
    """计算 tSNR_mean"""
    bold_data = nib.load(bold_path).get_fdata(dtype=np.float32)
    mask_data = nib.load(mask_path).get_fdata(dtype=np.float32)

    if len(bold_data.shape) < 4:
        print(f"    [警告] BOLD 图像维度异常！跳过 tSNR 计算。")
        return np.nan

    mu = np.mean(bold_data, axis=3)
    sigma = np.std(bold_data, axis=3, ddof=1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        tsnr = np.where(sigma > 0, mu / sigma, 0)
    
    valid_tsnr = tsnr[mask_data > 0]
    
    if valid_tsnr.size == 0:
        return float('nan')
        
    return float(np.mean(valid_tsnr))


def compute_fd_from_par(par_file):
    """
    从 FSL mcflirt 的 .par 文件计算 Framewise Displacement (FD)
    
    .par 文件格式: 6列，每行一个时间点
    列 1-3: 旋转（弧度）- Rx, Ry, Rz
    列 4-6: 平移（mm）- Tx, Ty, Tz
    
    FD 计算公式（Power et al. 2012）:
    FD = |Δd_x| + |Δd_y| + |Δd_z| + |Δα| + |Δβ| + |Δγ|
    其中旋转需要转换为位移（假设头部半径 50mm）
    """
    try:
        # 读取运动参数
        motion_params = np.loadtxt(par_file)
        
        if motion_params.ndim == 1:
            motion_params = motion_params.reshape(1, -1)
        
        # 计算相邻时间点的差值
        diff = np.diff(motion_params, axis=0)
        
        # 旋转参数（前3列，弧度）转换为位移（假设头部半径 50mm）
        # 弧长 = 角度 × 半径
        head_radius = 50  # mm
        rot_displacement = np.abs(diff[:, :3]) * head_radius
        
        # 平移参数（后3列，mm）
        trans_displacement = np.abs(diff[:, 3:])
        
        # FD = 旋转位移 + 平移位移
        fd = np.sum(rot_displacement, axis=1) + np.sum(trans_displacement, axis=1)
        
        # 第一个时间点的 FD 设为 0（没有前一个时间点可比较）
        fd = np.insert(fd, 0, 0)
        
        return fd
    except Exception as e:
        print(f"    [FD计算错误] 无法从 {par_file} 计算 FD: {e}")
        return None


def compute_dvars_from_bold(bold_path):
    """
    从 BOLD 数据计算 DVARS（简化版）
    
    DVARS 是相邻时间点之间全局信号变化的均方根
    这里返回的是 std_dvars 的近似值
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


def load_nii(path):
    return nib.load(path).get_fdata(dtype=np.float32)

def main():
    start_time = time.time()
    
    if not os.path.exists(DEEPPREP_BOLD_ROOT):
        print(f"数据根目录 {DEEPPREP_BOLD_ROOT} 不存在，请检查！")
        return
        
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    
    subjects = [d for d in os.listdir(DEEPPREP_BOLD_ROOT) if d.startswith("sub-")]
    subjects = [d for d in subjects if os.path.isdir(os.path.join(DEEPPREP_BOLD_ROOT, d))]
    subjects.sort()
    
    print(f"发现 {len(subjects)} 个被试。准备开始处理...\n")
    
    results = []
    
    for i, sub in enumerate(subjects, 1):
        print(f"[{i}/{len(subjects)}] 处理被试: {sub}")
        sub_dir = os.path.join(DEEPPREP_BOLD_ROOT, sub)
        
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
        t1w_files = glob.glob(os.path.join(sub_dir, "anat", "*_desc-preproc_T1w.nii.gz"))
        mask_files = glob.glob(os.path.join(sub_dir, "anat", "*_desc-brain_mask.nii.gz"))
        
        gm_files = [f for f in glob.glob(os.path.join(sub_dir, "anat", "*_probseg.nii.gz")) if 'GM' in f or 'gm' in f.lower() or 'gray' in f.lower()]
        wm_files = [f for f in glob.glob(os.path.join(sub_dir, "anat", "*_probseg.nii.gz")) if 'WM' in f or 'wm' in f.lower() or 'white' in f.lower()]
        
        if t1w_files and mask_files and gm_files and wm_files:
            try:
                t1w_img = load_nii(t1w_files[0])
                brain_mask = load_nii(mask_files[0]) > 0
                gm_prob = load_nii(gm_files[0])
                wm_prob = load_nii(wm_files[0])
                
                gm_mask = (gm_prob > 0.5) & brain_mask
                wm_mask = (wm_prob > 0.5) & brain_mask
                
                metrics["EFC"] = get_efc(t1w_img)
                metrics["WM2MAX"] = get_wm2max(t1w_img, wm_mask, brain_mask)
                metrics["CJV"] = get_cjv(t1w_img, wm_mask, gm_mask)
            except Exception as e:
                print(f"    [结构像报错] {sub} 计算异常: {e}")
        else:
            print(f"    [结构像缺失] {sub} anat 目录下文件不全")
        
        # ================================
        # 2. 计算功能像指标 (Func)
        # ================================
        # 递归搜索所有 func 目录（支持 ses-*/func/ 等多层结构）
        bold_files = glob.glob(os.path.join(sub_dir, "**", "func", "*_space-T1w*_desc-preproc_bold.nii.gz"), recursive=True)
        
        if bold_files:
            bold_path = bold_files[0]
            basename = os.path.basename(bold_path)
            
            run_key = basename.split("_space-")[0]
            func_dir = os.path.dirname(bold_path)
            
            mask_path = os.path.join(func_dir, f"{run_key}_space-T1w_desc-brain_mask.nii.gz")
            
            par_patterns = [
                os.path.join(func_dir, f"{run_key}_bold_mcf.nii.gz.par"),
                os.path.join(func_dir, f"{run_key}_bold_valid_mcf.nii.gz.par")
            ]
            par_file = None
            for pattern in par_patterns:
                if os.path.exists(pattern):
                    par_file = pattern
                    break
            
            # tSNR 计算
            if os.path.exists(mask_path):
                try:
                    metrics["tSNR_mean"] = get_tsnr(bold_path, mask_path)
                except Exception as e:
                    print(f"    [tSNR报错] {sub} 计算异常: {e}")
            else:
                print(f"    [tSNR缺失] 找不到对应的 func mask: {mask_path}")
            
            # FD 计算（从 .par 文件）
            if par_file and os.path.exists(par_file):
                try:
                    fd_values = compute_fd_from_par(par_file)
                    if fd_values is not None:
                        metrics["MeanFD"] = np.mean(fd_values)
                        print(f"    [FD] 从 {os.path.basename(par_file)} 计算完成")
                except Exception as e:
                    print(f"    [FD计算报错] {sub}: {e}")
            else:
                print(f"    [FD缺失] 找不到 motion parameters 文件")
            
            # DVARS 计算（从 BOLD 数据）
            try:
                dvars_values = compute_dvars_from_bold(bold_path)
                if dvars_values is not None:
                    metrics["MeanStdDVARS"] = np.mean(dvars_values)
                    print(f"    [DVARS] 从 BOLD 数据计算完成")
            except Exception as e:
                print(f"    [DVARS计算报错] {sub}: {e}")
        else:
             print(f"    [功能像缺失] {sub} 找不到 space-T1w 的 BOLD 序列。")
                  
        results.append(metrics)
        
    # 保存结果
    df_results = pd.DataFrame(results)
    
    cols = ["SubjectID", "MeanFD", "MeanStdDVARS", "tSNR_mean", "CJV", "EFC", "WM2MAX"]
    df_results = df_results[cols]
    
    df_results.to_csv(OUTPUT_CSV_PATH, index=False)
    
    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"处理完成！总共耗时: {elapsed/60:.2f} 分钟。")
    print(f"结果已保存至: {OUTPUT_CSV_PATH}")
    print("-" * 50)
    
    # 显示统计摘要
    print("\n=== QC 指标统计摘要 ===")
    print(df_results.describe())

if __name__ == "__main__":
    main()
