#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ===== 手动在这里设置参数 =====
# 存放 QC 结果 CSV 的目录
ANALYSIS_DIR = "/mnt/sda1/zhangyan/cpac_output/质量控制分析"

# default / gemini 的整合 QC 表（功能 + 结构）
DEFAULT_QC_CSV = "ds002748-表格/ds002748-default-qc-all.csv"
GEMINI_QC_CSV = "ds002748-表格/ds002748-llm-qc-all.csv"

# 结果输出目录（会自动创建）
OUTPUT_SUBDIR = "ds002748"


def main():
    analysis_dir = ANALYSIS_DIR
    out_dir = os.path.join(analysis_dir, OUTPUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    default_path = os.path.join(analysis_dir, DEFAULT_QC_CSV)
    gemini_path = os.path.join(analysis_dir, GEMINI_QC_CSV)

    print(f"default QC: {default_path}")
    print(f"gemini  QC: {gemini_path}")
    print(f"输出目录: {out_dir}")

    df_def = pd.read_csv(default_path)
    df_gem = pd.read_csv(gemini_path)

    # 选择要对比的指标
    # 这里既包括功能像也包括结构像指标
    metrics = [
        "MeanFD_Power",
        "MeanDVARS",
        "boldSnr",
        "CJV",
        "EFC",
        "WM2MAX",
        "GM_mean",
        "WM_mean",
    ]

    # 1. 计算各指标在 default / gemini 的总体均值（只看均值，不做检验）
    mean_def = df_def[metrics].mean(skipna=True)
    mean_gem = df_gem[metrics].mean(skipna=True)

    rows = []
    for m in metrics:
        rows.append(
            {
                "metric": m,
                "mean_default": mean_def[m],
                "mean_gemini": mean_gem[m],
                "n_default": int(df_def[m].notna().sum()),
                "n_gemini": int(df_gem[m].notna().sum()),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_csv_path = os.path.join(out_dir, "qc_means_default_vs_gemini.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    print("\n=== 各指标均值对比（default vs gemini） ===")
    print(summary_df)
    print(f"均值汇总已保存到: {summary_csv_path}")

    # 2. 按 SubjectID 合并两个表，用于画散点图
    merge_cols = ["SubjectID"] + metrics
    merged_df = df_def[merge_cols].merge(
        df_gem[merge_cols],
        on="SubjectID",
        how="outer",
        suffixes=("_def", "_gem"),
    )

    merged_csv_path = os.path.join(out_dir, "qc_default_vs_gemini_merged_by_subject.csv")
    merged_df.to_csv(merged_csv_path, index=False)
    print(f"按 SubjectID 合并后的表已保存到: {merged_csv_path}")

    # 3. 为结构像指标画 "箱线图 + 散点点云"（同一坐标系）
    struct_metrics = ["CJV", "EFC", "WM2MAX"]
    fig, axes = plt.subplots(1, len(struct_metrics), figsize=(5 * len(struct_metrics), 4))

    # 当只有一个指标时，axes 不是数组，需要统一处理成列表
    if len(struct_metrics) == 1:
        axes = [axes]

    for ax, m in zip(axes, struct_metrics):
        vals_def = df_def[m].dropna().values
        vals_gem = df_gem[m].dropna().values

        if vals_def.size == 0 and vals_gem.size == 0:
            ax.set_visible(False)
            continue

        positions = [0, 1]
        labels = ["default", "LLM-CPAC"]

        # 画箱线图
        ax.boxplot(
            [vals_def, vals_gem],
            positions=positions,
            widths=0.6,
            showmeans=True,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)

        # 叠加散点（对每个配置在 x 上做一点抖动）
        jitter = 0.06
        if vals_def.size > 0:
            x_def = np.random.normal(loc=positions[0], scale=jitter, size=vals_def.size)
            ax.scatter(x_def, vals_def, alpha=0.7, color="black", s=10)
        if vals_gem.size > 0:
            x_gem = np.random.normal(loc=positions[1], scale=jitter, size=vals_gem.size)
            ax.scatter(x_gem, vals_gem, alpha=0.7, color="black", s=10)

        ax.set_title(m)
        ax.set_ylabel(m)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(
        out_dir, "qc_box_scatter_struct_CJV_EFC_WM2MAX_default_vs_gemini.png"
    )
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"结构像 CJV/EFC/WM2MAX 的箱线图+散点图已保存到: {fig_path}")

    # 4. 功能像指标的箱线图+散点图
    # 4.1 MeanFD_Power 和 MeanDVARS 画在一张图的两个子图中
    func_pair_metrics = ["MeanFD_Power", "MeanDVARS"]
    fig, axes = plt.subplots(1, len(func_pair_metrics), figsize=(5 * len(func_pair_metrics), 4))

    if len(func_pair_metrics) == 1:
        axes = [axes]

    for ax, m in zip(axes, func_pair_metrics):
        vals_def = df_def[m].dropna().values
        vals_gem = df_gem[m].dropna().values

        if vals_def.size == 0 and vals_gem.size == 0:
            ax.set_visible(False)
            continue

        positions = [0, 1]
        labels = ["default", "LLM-CPAC"]

        ax.boxplot(
            [vals_def, vals_gem],
            positions=positions,
            widths=0.6,
            showmeans=True,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)

        jitter = 0.06
        if vals_def.size > 0:
            x_def = np.random.normal(loc=positions[0], scale=jitter, size=vals_def.size)
            ax.scatter(x_def, vals_def, alpha=0.7, color="black", s=10)
        if vals_gem.size > 0:
            x_gem = np.random.normal(loc=positions[1], scale=jitter, size=vals_gem.size)
            ax.scatter(x_gem, vals_gem, alpha=0.7, color="black", s=10)

        ax.set_title(m)
        ax.set_ylabel(m)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    func_fd_dvars_path = os.path.join(
        out_dir, "qc_box_scatter_func_MeanFD_DVARS_default_vs_gemini.png"
    )
    fig.savefig(func_fd_dvars_path, dpi=150)
    plt.close(fig)
    print(f"功能像 MeanFD/DVARS 的箱线图+散点图已保存到: {func_fd_dvars_path}")

    # 4.2 boldSnr 单独一张图
    m = "boldSnr"
    vals_def = df_def[m].dropna().values
    vals_gem = df_gem[m].dropna().values

    fig, ax = plt.subplots(figsize=(5, 4))
    if vals_def.size == 0 and vals_gem.size == 0:
        ax.set_visible(False)
    else:
        # 使用 IQR 规则对极端大的 outlier 做裁剪，只用于画图，不修改原始数据
        combined = np.concatenate([vals_def, vals_gem])
        if combined.size > 0:
            q1, q3 = np.percentile(combined, [25, 75])
            iqr = q3 - q1
            upper = q3 + 1.5 * iqr
            vals_def_plot = vals_def[vals_def <= upper]
            vals_gem_plot = vals_gem[vals_gem <= upper]
        else:
            vals_def_plot = vals_def
            vals_gem_plot = vals_gem

        positions = [0, 1]
        labels = ["default", "LLM-CPAC"]

        ax.boxplot(
            [vals_def_plot, vals_gem_plot],
            positions=positions,
            widths=0.6,
            showmeans=True,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)

        jitter = 0.06
        if vals_def_plot.size > 0:
            x_def = np.random.normal(loc=positions[0], scale=jitter, size=vals_def_plot.size)
            ax.scatter(x_def, vals_def_plot, alpha=0.7, color="black", s=10)
        if vals_gem_plot.size > 0:
            x_gem = np.random.normal(loc=positions[1], scale=jitter, size=vals_gem_plot.size)
            ax.scatter(x_gem, vals_gem_plot, alpha=0.7, color="black", s=10)

        ax.set_title(m)
        ax.set_ylabel(m)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    func_bold_path = os.path.join(
        out_dir, "qc_box_scatter_func_boldSnr_default_vs_gemini.png"
    )
    fig.savefig(func_bold_path, dpi=150)
    plt.close(fig)
    print(f"功能像 boldSnr 的箱线图+散点图已保存到: {func_bold_path}")


if __name__ == "__main__":
    main()
