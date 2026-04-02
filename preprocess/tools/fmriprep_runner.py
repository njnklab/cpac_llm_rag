#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR.parent / "configs" / "config.yaml"

def load_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

cfg = load_config()
fm = cfg['fmriprep']

# Default configuration: edit here if you want to run without passing CLI arguments.
# 说明：你只运行 `python fmriprep_runner.py` 时，会使用下面这些默认值。

# BIDS 根目录（原始数据所在目录，里面应有 sub-*）
DEFAULT_BIDS_DIR = fm['bids_dir']

# fMRIPrep 输出目录（derivatives 根；脚本会在里面生成 logs/）
DEFAULT_OUT_DIR = fm['output_dir']

# 工作目录根（中间文件很大；每个被试会单独用 work_root/sub-XX）
DEFAULT_WORK_ROOT = fm['work_root']

# PyBIDS 数据库目录（加速索引；建议长期保留，不要频繁删）
DEFAULT_BIDS_DB_DIR = fm['bids_db_dir']

# FreeSurfer license 文件路径（必须存在且可读）
DEFAULT_FS_LICENSE = fm['fs_license']

# 被试列表文件（每行一个被试号：01 或 sub-01 都可以；空/不存在则自动扫描 BIDS 目录）
DEFAULT_SUBJECTS_TXT = str(SCRIPT_DIR / fm['subjects_txt'])

# 已完成被试记录文件（脚本会把成功的被试 append 进去；下次会自动跳过）
DEFAULT_DONE_TXT = str(SCRIPT_DIR / fm['done_txt'])

# 默认容器镜像版本：
# - "AUTO"：自动从本机 `docker images nipreps/fmriprep` 里挑一个版本号最高的 tag
# - 或者手动写死："nipreps/fmriprep:25.2.3" / "nipreps/fmriprep:23.2.3" 等
DEFAULT_IMAGE = fm['image']

# 并发：同时跑几个被试（每被试一个独立 fMRIPrep 进程）
DEFAULT_MAX_JOBS = fm['max_jobs']

# 单被试资源：CPU 线程数 / 内存(MB) / OpenMP 线程数
# 你的服务器 32C/128GB，按 2×(12CPU/50GB) 比较稳。
DEFAULT_NPROCS = fm['nprocs']
DEFAULT_MEM_MB = fm['mem_mb']
DEFAULT_OMP_NTHREADS = fm['omp_nthreads']

# 输出空间：固定输出 T1w + 1 个 MNI（这里是默认 MNI 模板名）
DEFAULT_MNI_SPACE = fm['mni_space']

# 输出级别：full 会有报告/更多输出；不要用 minimal（会影响 confounds/QC）
DEFAULT_LEVEL = fm['level']

# 是否跳过 BIDS 校验（bids-validator）。
# 你的数据集根目录里有一些非 BIDS 文件（例如 annex-uuid / dataset_description.txt），会导致 validator 直接报错退出。
# 为了保证批处理能继续跑，这里默认跳过；如果你修好 BIDS 再把它改成 False。
DEFAULT_SKIP_BIDS_VALIDATION = fm['skip_bids_validation']

# QC：FD 统计时用的阈值（计算 fd_gt_<threshold>_ratio）
DEFAULT_FD_THRESHOLD = fm['fd_threshold']

# QC：是否计算 tSNR（需要 python 环境里有 numpy + nibabel；失败不会影响主流程）
DEFAULT_COMPUTE_TSNR = fm['compute_tsnr']

# 是否加 --notrack（禁用 Nipreps 的使用统计；建议保持 True）
DEFAULT_NOTRACK = fm['notrack']


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_txt_list(path: Path) -> List[str]:
    items: List[str] = []
    if not path.exists():
        return items
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("sub-"):
            s = s[4:]
        items.append(s)
    return items


def _append_done(path: Path, subject: str) -> None:
    _mkdir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{subject}\n")


def _scan_subjects(bids_dir: Path) -> List[str]:
    subs = []
    for p in bids_dir.glob("sub-*"):
        if p.is_dir():
            subs.append(p.name.replace("sub-", ""))
    subs = sorted(set(subs))
    return subs


def _run_check(cmd: Sequence[str], label: str) -> None:
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    except FileNotFoundError as e:
        raise RuntimeError(f"Missing executable for {label}: {cmd[0]}") from e
    if r.returncode != 0:
        raise RuntimeError(f"Precheck failed: {label}: return_code={r.returncode}\n{r.stdout}")


def _get_help_text_docker_image(image: str) -> str:
    r = subprocess.run(
        ["docker", "run", "--rm", image, "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(
            "Failed to run fMRIPrep help inside docker image. "
            f"image={image} rc={r.returncode}\n{r.stdout}"
        )
    return r.stdout


def _docker_image_exists(image: str) -> bool:
    r = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return r.returncode == 0


def _select_local_fmriprep_image(repo: str = "nipreps/fmriprep") -> str:
    r = subprocess.run(
        ["docker", "images", repo, "--format", "{{.Repository}}:{{.Tag}}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Failed to query local docker images for {repo}.\n{r.stdout}")

    images: List[str] = []
    for line in r.stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.endswith(":<none>") or s.endswith(":none"):
            continue
        images.append(s)

    if not images:
        raise RuntimeError(
            f"No local docker image found for {repo}. Please pull one first, e.g. 'docker pull {repo}:<tag>'."
        )

    def parse_ver(img: str) -> Optional[Tuple[int, int, int]]:
        tag = img.split(":", 1)[-1]
        m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", tag)
        if not m:
            return None
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    semver = [(parse_ver(i), i) for i in images]
    semver_only = [(v, i) for (v, i) in semver if v is not None]
    if semver_only:
        semver_only.sort(key=lambda x: x[0])
        return semver_only[-1][1]

    return images[0]


def _has_option(help_text: str, opt: str) -> bool:
    pat = re.compile(r"(^|\n)\s*" + re.escape(opt) + r"(\s|,|$)")
    return bool(pat.search(help_text))


def _pick_option(help_text: str, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if _has_option(help_text, c):
            return c
    if required:
        raise RuntimeError(f"Wrapper does not support any of: {candidates}")
    return None


def _parse_bids_entities(path: Path) -> Dict[str, str]:
    name = path.name
    ent: Dict[str, str] = {}
    m = re.search(r"sub-([a-zA-Z0-9]+)", name)
    if m:
        ent["subject"] = m.group(1)
    m = re.search(r"ses-([a-zA-Z0-9]+)", name)
    if m:
        ent["session"] = m.group(1)
    m = re.search(r"task-([a-zA-Z0-9]+)", name)
    if m:
        ent["task"] = m.group(1)
    m = re.search(r"run-([a-zA-Z0-9]+)", name)
    if m:
        ent["run"] = m.group(1)
    m = re.search(r"space-([a-zA-Z0-9]+)", name)
    if m:
        ent["space"] = m.group(1)
    return ent


def _key_from_entities(ent: Dict[str, str], include_space: bool) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    return (
        ent.get("subject"),
        ent.get("session"),
        ent.get("task"),
        ent.get("run"),
        ent.get("space") if include_space else None,
    )


def _safe_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    t = str(s).strip()
    if not t or t.lower() in {"n/a", "na", "nan"}:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _summarize_series(xs: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not xs:
        return None, None, None
    try:
        return float(mean(xs)), float(median(xs)), float(max(xs))
    except Exception:
        return None, None, None


def _confounds_metrics(confounds_tsv: Path, fd_threshold: float) -> Dict[str, Optional[float]]:
    with confounds_tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fd_vals: List[float] = []
        dvars_vals: List[float] = []
        dvars_col: Optional[str] = None
        for row in reader:
            if dvars_col is None:
                if "std_dvars" in row:
                    dvars_col = "std_dvars"
                elif "dvars" in row:
                    dvars_col = "dvars"
            fd = _safe_float(row.get("framewise_displacement"))
            if fd is not None:
                fd_vals.append(fd)
            if dvars_col is not None:
                dv = _safe_float(row.get(dvars_col))
                if dv is not None:
                    dvars_vals.append(dv)

    fd_mean, fd_median, fd_max = _summarize_series(fd_vals)
    dvars_mean, dvars_median, dvars_max = _summarize_series(dvars_vals)

    fd_gt_ratio: Optional[float]
    if fd_vals:
        fd_gt_ratio = float(sum(1 for v in fd_vals if v > fd_threshold) / len(fd_vals))
    else:
        fd_gt_ratio = None

    return {
        "fd_mean": fd_mean,
        "fd_median": fd_median,
        "fd_max": fd_max,
        f"fd_gt_{fd_threshold}_ratio": fd_gt_ratio,
        "dvars_col": dvars_col,
        "dvars_mean": dvars_mean,
        "dvars_median": dvars_median,
        "dvars_max": dvars_max,
    }


def _compute_tsnr(bold_nii: Path, mask_nii: Path) -> Tuple[Optional[float], Optional[float]]:
    try:
        import numpy as np
        import nibabel as nb
    except Exception as e:
        raise RuntimeError("Missing dependency for tSNR: need numpy and nibabel") from e

    bold_img = nb.load(str(bold_nii))
    mask_img = nb.load(str(mask_nii))

    bold = bold_img.get_fdata(dtype=np.float32)
    mask = mask_img.get_fdata(dtype=np.float32)

    if bold.ndim != 4:
        return None, None

    mask_bool = mask > 0.5
    if mask_bool.ndim != 3 or not mask_bool.any():
        return None, None

    mu = np.mean(bold, axis=-1)
    sd = np.std(bold, axis=-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        tsnr = mu / sd
        tsnr[~np.isfinite(tsnr)] = np.nan

    vals = tsnr[mask_bool]
    if vals.size == 0:
        return None, None

    tsnr_median = float(np.nanmedian(vals)) if np.isfinite(np.nanmedian(vals)) else None
    tsnr_mean = float(np.nanmean(vals)) if np.isfinite(np.nanmean(vals)) else None
    return tsnr_median, tsnr_mean


def _find_report(out_dir: Path, subject: str) -> Optional[Path]:
    candidates = list(out_dir.rglob(f"sub-{subject}.html"))
    if candidates:
        return sorted(candidates)[0]
    candidates = list(out_dir.rglob(f"*sub-{subject}*.html"))
    if candidates:
        return sorted(candidates)[0]
    return None


def _gather_qc(
    out_dir: Path,
    logs_dir: Path,
    fd_threshold: float,
    compute_tsnr: bool,
    prefer_space: str = "T1w",
) -> Path:
    confounds = sorted(out_dir.rglob("*desc-confounds_timeseries.tsv"))
    bolds = sorted(out_dir.rglob("*desc-preproc_bold.nii.gz"))
    masks = sorted(out_dir.rglob("*desc-brain_mask.nii.gz"))

    bold_by_key: Dict[Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]], Path] = {}
    mask_by_key: Dict[Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]], Path] = {}

    def insert_best(index: Dict[Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]], Path], p: Path) -> None:
        ent = _parse_bids_entities(p)
        key_space = _key_from_entities(ent, include_space=True)
        key_nospace = _key_from_entities(ent, include_space=False)
        current = index.get(key_space)
        if current is None:
            index[key_space] = p
        if index.get(key_nospace) is None:
            index[key_nospace] = p

    for p in bolds:
        insert_best(bold_by_key, p)
    for p in masks:
        insert_best(mask_by_key, p)

    rows: List[Dict[str, object]] = []

    for c in confounds:
        ent = _parse_bids_entities(c)
        subject = ent.get("subject")
        session = ent.get("session")
        task = ent.get("task")
        run = ent.get("run")

        m = _confounds_metrics(c, fd_threshold=fd_threshold)

        bold_path: Optional[Path] = None
        mask_path: Optional[Path] = None
        tsnr_median: Optional[float] = None
        tsnr_mean: Optional[float] = None

        if compute_tsnr:
            key_space = (subject, session, task, run, prefer_space)
            key_nospace = (subject, session, task, run, None)

            bold_path = bold_by_key.get(key_space) or bold_by_key.get(key_nospace)
            mask_path = mask_by_key.get(key_space) or mask_by_key.get(key_nospace)

            if bold_path is not None and mask_path is not None:
                try:
                    tsnr_median, tsnr_mean = _compute_tsnr(bold_path, mask_path)
                except Exception as e:
                    _eprint(f"tSNR failed for {c}: {e}")

        report_path = subject and _find_report(out_dir, subject)

        row: Dict[str, object] = {
            "subject": subject,
            "session": session,
            "task": task,
            "run": run,
            "fd_mean": m.get("fd_mean"),
            "fd_median": m.get("fd_median"),
            "fd_max": m.get("fd_max"),
            f"fd_gt_{fd_threshold}_ratio": m.get(f"fd_gt_{fd_threshold}_ratio"),
            "dvars_col": m.get("dvars_col"),
            "dvars_mean": m.get("dvars_mean"),
            "dvars_median": m.get("dvars_median"),
            "dvars_max": m.get("dvars_max"),
            "tsnr_median": tsnr_median,
            "tsnr_mean": tsnr_mean,
            "confounds_path": str(c),
            "bold_path": str(bold_path) if bold_path else "",
            "mask_path": str(mask_path) if mask_path else "",
            "report_path": str(report_path) if report_path else "",
        }
        rows.append(row)

    _mkdir(logs_dir)
    qc_csv = logs_dir / "qc_metrics.csv"

    fieldnames = [
        "subject",
        "session",
        "task",
        "run",
        "fd_mean",
        "fd_median",
        "fd_max",
        f"fd_gt_{fd_threshold}_ratio",
        "dvars_col",
        "dvars_mean",
        "dvars_median",
        "dvars_max",
        "tsnr_median",
        "tsnr_mean",
        "confounds_path",
        "bold_path",
        "mask_path",
        "report_path",
    ]

    with qc_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    return qc_csv


@dataclass
class SubjectResult:
    subject: str
    status: str
    return_code: Optional[int]
    start_time: str
    end_time: str
    duration_sec: float
    log_path: str
    work_dir: str
    command: str


def _run_subject(
    subject: str,
    cmd: List[str],
    log_path: Path,
    work_dir: Path,
    dry_run: bool,
) -> SubjectResult:
    _mkdir(log_path.parent)
    _mkdir(work_dir)

    start = time.time()
    start_iso = _now_iso()

    cmd_str = shlex.join(cmd)

    if dry_run:
        end = time.time()
        end_iso = _now_iso()
        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"[{start_iso}] DRY RUN\n")
            f.write(cmd_str + "\n")
        return SubjectResult(
            subject=subject,
            status="dry-run",
            return_code=None,
            start_time=start_iso,
            end_time=end_iso,
            duration_sec=end - start,
            log_path=str(log_path),
            work_dir=str(work_dir),
            command=cmd_str,
        )

    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"[{start_iso}] START\n")
        f.write(cmd_str + "\n")
        f.flush()
        try:
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
            rc = p.wait()
        except Exception as e:
            f.write(f"\nException: {e}\n")
            rc = 1

        end = time.time()
        end_iso = _now_iso()
        f.write(f"\n[{end_iso}] END return_code={rc} duration_sec={end - start:.1f}\n")

    status = "success" if rc == 0 else "fail"
    return SubjectResult(
        subject=subject,
        status=status,
        return_code=rc,
        start_time=start_iso,
        end_time=end_iso,
        duration_sec=end - start,
        log_path=str(log_path),
        work_dir=str(work_dir),
        command=cmd_str,
    )


def _build_docker_run_command(
    help_text: str,
    image: str,
    bids_dir: Path,
    out_dir: Path,
    work_dir: Path,
    bids_db_dir: Path,
    fs_license: Path,
    subject: str,
    nprocs: int,
    omp_nthreads: int,
    mem_mb: int,
    mni_space: str,
    level: str,
    skip_bids_validation: bool,
    session_label: Optional[str],
    task_id: Optional[str],
    bids_filter_file: Optional[Path],
    add_notrack: bool,
) -> List[str]:
    # Container mount points
    c_bids = "/bids"
    c_out = "/out"
    c_work = "/work"
    c_db = "/bids_db"
    c_license = "/license.txt"

    cmd: List[str] = [
        "docker",
        "run",
        "--rm",
        "-u",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{str(bids_dir)}:{c_bids}:ro",
        "-v",
        f"{str(out_dir)}:{c_out}",
        "-v",
        f"{str(work_dir)}:{c_work}",
        "-v",
        f"{str(bids_db_dir)}:{c_db}",
        "-v",
        f"{str(fs_license)}:{c_license}:ro",
        image,
    ]

    fs_opt = _pick_option(help_text, ["--fs-license-file", "--fs_license_file", "--fs-license"], required=True)
    work_opt = _pick_option(help_text, ["-w", "--work-dir", "--work_dir"], required=True)
    bids_db_opt = _pick_option(help_text, ["--bids-database-dir", "--bids_database_dir"], required=False)

    cmd += [fs_opt, c_license]
    cmd += [work_opt, c_work]
    if bids_db_opt is not None:
        cmd += [bids_db_opt, c_db]

    cmd += [c_bids, c_out, "participant"]

    part_opt = _pick_option(help_text, ["--participant-label", "--participant_label"], required=True)
    cmd += [part_opt, subject]

    nprocs_opt = _pick_option(help_text, ["--nprocs", "--nthreads", "--n_cpus"], required=True)
    cmd += [nprocs_opt, str(nprocs)]

    omp_opt = _pick_option(help_text, ["--omp-nthreads", "--omp_nthreads"], required=True)
    cmd += [omp_opt, str(omp_nthreads)]

    mem_opt = _pick_option(help_text, ["--mem", "--mem_mb", "--mem-mb"], required=True)
    cmd += [mem_opt, str(mem_mb)]

    outspaces_opt = _pick_option(help_text, ["--output-spaces", "--output_spaces"], required=True)
    cmd += [outspaces_opt, "T1w", mni_space]

    if add_notrack and _has_option(help_text, "--notrack"):
        cmd += ["--notrack"]

    level_opt = _pick_option(help_text, ["--level"], required=False)
    if level_opt is not None and level:
        cmd += [level_opt, level]

    if skip_bids_validation:
        cmd += ["--skip-bids-validation"]

    if session_label:
        sess_opt = _pick_option(help_text, ["--session-label", "--session_label"], required=False)
        if sess_opt is not None:
            cmd += [sess_opt, session_label]

    if task_id:
        task_opt = _pick_option(help_text, ["--task-id", "--task_id"], required=False)
        if task_opt is not None:
            cmd += [task_opt, task_id]

    if bids_filter_file:
        bf_opt = _pick_option(help_text, ["--bids-filter-file", "--bids_filter_file"], required=False)
        if bf_opt is not None:
            cmd += [bf_opt, str(bids_filter_file)]

    return cmd


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="fmriprep_runner.py")

    ap.add_argument("--bids-dir", default=DEFAULT_BIDS_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--work-root", default=DEFAULT_WORK_ROOT)
    ap.add_argument("--bids-db-dir", default=DEFAULT_BIDS_DB_DIR)
    ap.add_argument("--fs-license", default=DEFAULT_FS_LICENSE)

    ap.add_argument("--max-jobs", type=int, default=DEFAULT_MAX_JOBS)
    ap.add_argument("--nprocs", type=int, default=DEFAULT_NPROCS)
    ap.add_argument("--mem-mb", type=int, default=DEFAULT_MEM_MB)
    ap.add_argument("--omp-nthreads", type=int, default=DEFAULT_OMP_NTHREADS)

    ap.add_argument("--mni-space", default=DEFAULT_MNI_SPACE)

    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--subjects-txt", type=str, default=DEFAULT_SUBJECTS_TXT)
    ap.add_argument("--done-txt", type=str, default=DEFAULT_DONE_TXT)

    ap.add_argument("--fd-threshold", type=float, default=DEFAULT_FD_THRESHOLD)
    ap.add_argument("--compute-tsnr", action="store_true", default=DEFAULT_COMPUTE_TSNR)
    ap.add_argument("--no-compute-tsnr", dest="compute_tsnr", action="store_false")

    ap.add_argument(
        "--skip-bids-validation",
        dest="skip_bids_validation",
        action="store_true",
        default=DEFAULT_SKIP_BIDS_VALIDATION,
    )
    ap.add_argument("--no-skip-bids-validation", dest="skip_bids_validation", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--image", default=DEFAULT_IMAGE)

    ap.add_argument("--rerun-failed", type=str, default=None)

    ap.add_argument("--level", default=DEFAULT_LEVEL, choices=["full", "resampling", "minimal"])

    ap.add_argument("--session-label", default=None)
    ap.add_argument("--task-id", default=None)
    ap.add_argument("--bids-filter-file", default=None)
    ap.add_argument("--notrack", dest="notrack", action="store_true", default=DEFAULT_NOTRACK)
    ap.add_argument("--no-notrack", dest="notrack", action="store_false")

    args = ap.parse_args(argv)

    bids_dir = Path(args.bids_dir)
    out_dir = Path(args.out_dir)
    work_root = Path(args.work_root)
    bids_db_dir = Path(args.bids_db_dir)
    fs_license = Path(args.fs_license)
    bids_filter_file = Path(args.bids_filter_file) if args.bids_filter_file else None

    logs_dir = out_dir / "logs"
    _mkdir(logs_dir)

    if not bids_dir.exists():
        raise RuntimeError(f"BIDS dir not found: {bids_dir}")
    if not fs_license.exists():
        raise RuntimeError(f"FS license not found: {fs_license}")

    _mkdir(out_dir)
    _mkdir(work_root)
    _mkdir(bids_db_dir)

    _run_check(["docker", "info"], label="docker info")

    image = args.image
    if image == "AUTO":
        image = _select_local_fmriprep_image("nipreps/fmriprep")

    if not _docker_image_exists(image):
        raise RuntimeError(
            "Docker image not found locally: "
            f"{image}. Please pull it first (docker pull ...) or change image in config."
        )

    help_text = _get_help_text_docker_image(image)
    wrapper_mode = "docker-run"

    subjects: List[str] = []

    if args.rerun_failed:
        sjson = Path(args.rerun_failed)
        payload = json.loads(sjson.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "subjects" not in payload:
            raise RuntimeError("rerun-failed expects a summary.json produced by this script (top-level key 'subjects')")
        subs = payload["subjects"]
        if not isinstance(subs, list):
            raise RuntimeError("summary.json 'subjects' should be a list")
        for item in subs:
            if isinstance(item, dict) and item.get("status") == "fail" and item.get("subject"):
                subjects.append(str(item["subject"]))
    elif args.subjects is not None and len(args.subjects) > 0:
        for s in args.subjects:
            if s.startswith("sub-"):
                s = s[4:]
            subjects.append(s)
    else:
        subjects_txt = Path(args.subjects_txt) if args.subjects_txt else None
        subjects_from_txt: List[str] = _read_txt_list(subjects_txt) if subjects_txt else []
        subjects = subjects_from_txt if subjects_from_txt else _scan_subjects(bids_dir)

    subjects = sorted(set(subjects))
    if not subjects:
        raise RuntimeError("No subjects found.")

    done_txt = Path(args.done_txt) if args.done_txt else None
    done_set = set(_read_txt_list(done_txt)) if done_txt else set()

    run_subjects = [s for s in subjects if s not in done_set]

    if not run_subjects:
        _eprint("Nothing to do: all selected subjects are marked done.")
        return 0

    _eprint(f"Selected subjects: {len(subjects)}")
    _eprint(f"Will run (excluding done): {len(run_subjects)}")

    results: List[SubjectResult] = []

    with ThreadPoolExecutor(max_workers=args.max_jobs) as ex:
        futs = {}
        for sub in run_subjects:
            sub_work = work_root / f"sub-{sub}"
            sub_log = logs_dir / f"fmriprep_sub-{sub}.log"

            cmd = _build_docker_run_command(
                help_text=help_text,
                image=image,
                bids_dir=bids_dir,
                out_dir=out_dir,
                work_dir=sub_work,
                bids_db_dir=bids_db_dir,
                fs_license=fs_license,
                subject=sub,
                nprocs=args.nprocs,
                omp_nthreads=args.omp_nthreads,
                mem_mb=args.mem_mb,
                mni_space=args.mni_space,
                level=args.level,
                skip_bids_validation=args.skip_bids_validation,
                session_label=args.session_label,
                task_id=args.task_id,
                bids_filter_file=bids_filter_file,
                add_notrack=bool(args.notrack),
            )

            _eprint(f"START sub-{sub}")
            fut = ex.submit(_run_subject, sub, cmd, sub_log, sub_work, args.dry_run)
            futs[fut] = sub

        for fut in as_completed(futs):
            sub = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                _eprint(f"FAIL sub-{sub}: {e}")
                r = SubjectResult(
                    subject=sub,
                    status="fail",
                    return_code=1,
                    start_time=_now_iso(),
                    end_time=_now_iso(),
                    duration_sec=0.0,
                    log_path=str(logs_dir / f"fmriprep_sub-{sub}.log"),
                    work_dir=str(work_root / f"sub-{sub}"),
                    command="",
                )
            results.append(r)
            _eprint(f"END sub-{sub}: {r.status} rc={r.return_code}")

            if done_txt and r.status == "success":
                _append_done(done_txt, sub)

    results = sorted(results, key=lambda x: x.subject)

    summary = {
        "created_at": _now_iso(),
        "wrapper": wrapper_mode,
        "bids_dir": str(bids_dir),
        "out_dir": str(out_dir),
        "work_root": str(work_root),
        "bids_db_dir": str(bids_db_dir),
        "fs_license": str(fs_license),
        "image": image,
        "max_jobs": args.max_jobs,
        "nprocs": args.nprocs,
        "mem_mb": args.mem_mb,
        "omp_nthreads": args.omp_nthreads,
        "mni_space": args.mni_space,
        "level": args.level,
        "skip_bids_validation": bool(args.skip_bids_validation),
        "dry_run": bool(args.dry_run),
        "notes": {
            "cjv_efc_wm2max": "not generated; recommended to run MRIQC if needed",
        },
        "subjects": [asdict(r) for r in results],
    }

    summary_json = logs_dir / "summary.json"
    summary_csv = logs_dir / "summary.csv"

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(asdict(results[0]).keys()) if results else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    qc_csv: Optional[Path] = None
    try:
        qc_csv = _gather_qc(
            out_dir=out_dir,
            logs_dir=logs_dir,
            fd_threshold=args.fd_threshold,
            compute_tsnr=bool(args.compute_tsnr),
            prefer_space="T1w",
        )
    except Exception as e:
        _eprint(f"Warning: QC aggregation failed (ignored): {e}")

    _eprint(f"Wrote: {summary_json}")
    _eprint(f"Wrote: {summary_csv}")
    if qc_csv is not None:
        _eprint(f"Wrote: {qc_csv}")

    n_fail = sum(1 for r in results if r.status == "fail")
    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        _eprint(f"ERROR: {e}")
        raise
