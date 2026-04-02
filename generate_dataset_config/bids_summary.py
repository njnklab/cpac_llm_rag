# A comprehensive script to get detailed BIDS dataset information
import pandas as pd
from bids import BIDSLayout
import json
import os

def safe_get(meta, keys, default="MISSING"):
    """
    Safely get a value from metadata with support for key synonyms.
    Returns the first present, non-empty value among the provided keys.
    """
    try:
        if isinstance(keys, (list, tuple)):
            for k in keys:
                if k in meta and meta[k] not in [None, ""]:
                    return meta[k]
            return default
        return meta.get(keys, default)
    except Exception:
        return default


def _to_float(val):
    try:
        if isinstance(val, (list, tuple)):
            return float(val[0])
        return float(val)
    except Exception:
        return None


def _to_int(val):
    try:
        if isinstance(val, (list, tuple)):
            return int(round(float(val[0])))
        return int(round(float(val)))
    except Exception:
        return None


def get_pe_steps(meta):
    """Return phase-encoding steps if available (AcquisitionMatrixPE/PhaseEncodingSteps/ReconMatrixPE)."""
    pe = safe_get(meta, ["AcquisitionMatrixPE", "PhaseEncodingSteps", "ReconMatrixPE"], "MISSING")
    return _to_int(pe)


def derive_readout_params(meta):
    """
    Derive EffectiveEchoSpacing (EES) and TotalReadoutTime (ToRT) when possible.
    Preference: use JSON values; if missing, derive via (PE_steps-1) relation.
    Returns (ees, tort, pe_steps), where each may be None if unavailable.
    """
    ees = safe_get(meta, ["EffectiveEchoSpacing", "EchoSpacing"], "MISSING")
    tort = safe_get(meta, ["TotalReadoutTime"], "MISSING")
    ees_val = _to_float(ees) if ees != "MISSING" else None
    tort_val = _to_float(tort) if tort != "MISSING" else None
    pe_steps = get_pe_steps(meta)

    # Try derivations
    if tort_val is None and ees_val is not None and pe_steps and pe_steps > 1:
        tort_val = ees_val * (pe_steps - 1)
    if ees_val is None and tort_val is not None and pe_steps and pe_steps > 1:
        ees_val = tort_val / (pe_steps - 1)

    # Fallback: try BandwidthPerPixelPhaseEncode and ReconMatrixPE if available
    if ees_val is None:
        bpppe = _to_float(safe_get(meta, ["BandwidthPerPixelPhaseEncode"], "MISSING"))
        recon_pe = _to_int(safe_get(meta, ["ReconMatrixPE"], "MISSING"))
        if bpppe and recon_pe and bpppe > 0 and recon_pe > 0:
            try:
                ees_val = 1.0 / (bpppe * recon_pe)
            except Exception:
                pass

    return ees_val, tort_val, pe_steps


def analyze_slice_timing(slice_timing, preview_n=10):
    """
    Analyze SliceTiming array: count, min, max, span, pattern (ascending/descending/interleaved), preview.
    Returns a dict with keys: count, min, max, span, pattern, preview_str
    """
    if not isinstance(slice_timing, list):
        return {
            "count": 0,
            "min": None,
            "max": None,
            "span": None,
            "pattern": "MISSING",
            "preview_str": ""
        }
    vals = []
    for v in slice_timing:
        fv = _to_float(v)
        if fv is not None:
            vals.append(fv)
    if not vals:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "span": None,
            "pattern": "MISSING",
            "preview_str": ""
        }
    inc = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    dec = all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    if inc:
        pattern = "ascending"
    elif dec:
        pattern = "descending"
    else:
        pattern = "interleaved/irregular"
    prev = ", ".join(f"{x:.5f}" for x in vals[:preview_n])
    return {
        "count": len(vals),
        "min": min(vals),
        "max": max(vals),
        "span": max(vals) - min(vals),
        "pattern": pattern,
        "preview_str": prev
    }


def _fmt_num(val, ndigits=6):
    if val is None:
        return "MISSING"
    try:
        return f"{float(val):.{ndigits}g}"
    except Exception:
        return str(val)


def summarize_value_counts(series, topn=3):
    """Return a compact summary string of value counts for a pandas Series."""
    if series is None or len(series) == 0:
        return "No runs found"
    # Format values for stable grouping
    def _fmt(v):
        if pd.isna(v):
            return "MISSING"
        if isinstance(v, (int, float)):
            return _fmt_num(v)
        return str(v)
    s = series.map(_fmt)
    unique_count = s.nunique(dropna=False)
    vc = s.value_counts(dropna=False).head(topn)
    if unique_count == 1:
        val = vc.index[0]
        return f"unique=1 -> {val}"
    items = ", ".join(f"{idx}: {cnt}" for idx, cnt in zip(vc.index, vc.values))
    return f"unique={unique_count} -> {{{items}}}"


def extract_bold_run_info(layout, bold_file, fmap_files=None, slicetiming_preview_n=10):
    """
    Extract a comprehensive set of run-level parameters for a BOLD file.
    Returns a dict of values; all missing values are set to 'MISSING'.
    """
    fmap_files = fmap_files or []
    info = {}
    try:
        meta = bold_file.get_metadata()
    except Exception:
        meta = {}
    try:
        header = bold_file.get_image().header
        dims = header.get_data_shape()
        zooms = [round(float(z), 3) for z in header.get_zooms()]
    except Exception:
        dims = ()
        zooms = []

    entities = getattr(bold_file, "entities", {}) or {}
    subject = entities.get("subject", "MISSING")
    session = entities.get("session", "MISSING")
    run = entities.get("run", "MISSING")
    task = entities.get("task", "MISSING")

    # Basics
    info["Subject"] = subject
    info["Session"] = session
    info["Run"] = run
    info["Task"] = task
    info["Filename"] = os.path.basename(getattr(bold_file, "path", "MISSING"))

    # Numeric parameters
    tr = _to_float(safe_get(meta, ["RepetitionTime"], "MISSING"))
    te = _to_float(safe_get(meta, ["EchoTime"], "MISSING"))
    info["RepetitionTime"] = tr if tr is not None else None
    info["EchoTime"] = te if te is not None else None
    flip = _to_float(safe_get(meta, ["FlipAngle"], "MISSING"))
    info["FlipAngle"] = flip if flip is not None else None
    bfs = _to_float(safe_get(meta, ["MagneticFieldStrength"], "MISSING"))
    info["MagneticFieldStrength"] = bfs if bfs is not None else None

    # Geometry & time
    try:
        info["DimX"], info["DimY"], info["DimZ"] = (dims + (None, None, None))[:3]
        info["DimT"] = dims[3] if len(dims) == 4 else None
    except Exception:
        info["DimX"], info["DimY"], info["DimZ"], info["DimT"] = None, None, None, None
    try:
        info["VoxelSizeX"], info["VoxelSizeY"], info["VoxelSizeZ"] = (zooms + [None, None, None])[:3]
    except Exception:
        info["VoxelSizeX"], info["VoxelSizeY"], info["VoxelSizeZ"] = None, None, None
    info["VoxelSizeStr"] = (
        f"{_fmt_num(info['VoxelSizeX'])}x{_fmt_num(info['VoxelSizeY'])}x{_fmt_num(info['VoxelSizeZ'])}"
    )

    # EPI distortion relevant
    info["PhaseEncodingDirection"] = safe_get(meta, ["PhaseEncodingDirection"], "MISSING")
    ees, tort, pe_steps = derive_readout_params(meta)
    info["EffectiveEchoSpacing"] = ees
    info["TotalReadoutTime"] = tort
    info["PE_Steps"] = pe_steps

    # Multiband / parallel factors
    info["MultibandAccelerationFactor"] = _to_float(
        safe_get(meta, ["MultibandAccelerationFactor", "MultibandFactor"], "MISSING")
    )
    prfi = safe_get(meta, ["ParallelReductionFactorInPlane", "ParallelReductionFactors", "AccelFactPE"], "MISSING")
    try:
        info["ParallelReductionFactorInPlane"] = _to_float(prfi) if prfi != "MISSING" else None
    except Exception:
        info["ParallelReductionFactorInPlane"] = None

    # Scanner / coil details
    info["Manufacturer"] = safe_get(meta, ["Manufacturer"], "MISSING")
    info["ManufacturersModelName"] = safe_get(meta, ["ManufacturersModelName", "ManufacturerModelName"], "MISSING")
    info["ReceiveCoilName"] = safe_get(meta, ["ReceiveCoilName"], "MISSING")
    info["CoilString"] = safe_get(meta, ["CoilString"], "MISSING")
    info["DeviceSerialNumber"] = safe_get(meta, ["DeviceSerialNumber"], "MISSING")
    info["SoftwareVersions"] = safe_get(meta, ["SoftwareVersions"], "MISSING")

    # Slice timing analysis
    st = safe_get(meta, ["SliceTiming"], "MISSING")
    st_analysis = analyze_slice_timing(st, slicetiming_preview_n)
    info["SliceTimingCount"] = st_analysis["count"]
    info["SliceTimingPattern"] = st_analysis["pattern"]
    info["SliceTimingPreview"] = st_analysis["preview_str"]

    # Fieldmap association (if provided)
    info["FieldmapAvailable"] = False
    info["FieldmapType"] = "MISSING"
    try:
        if fmap_files:
            intended_target = os.path.relpath(getattr(bold_file, "path", ""), layout.root)
            for fmap in fmap_files:
                try:
                    fmeta = fmap.get_metadata()
                except Exception:
                    fmeta = {}
                intended = fmeta.get("IntendedFor", [])
                if isinstance(intended, str):
                    intended = [intended]
                if isinstance(intended, list) and intended_target in intended:
                    info["FieldmapAvailable"] = True
                    info["FieldmapType"] = getattr(fmap, "entities", {}).get("suffix", "unknown")
                    break
    except Exception:
        pass

    return info


def get_bids_summary(bids_dir):
    """
    Generates a high-level summary of a BIDS dataset, focusing on pipeline-relevant info.
    """
    layout = BIDSLayout(bids_dir, validate=False)
    summary = ["[DATASET_LEVEL_SUMMARY]"]

    # --- 1. Dataset Description ---
    try:
        desc = layout.get_dataset_description()
        summary.append("\n--- Dataset Description ---")
        summary.append(f"  - Name: {desc.get('Name', 'MISSING')}")
        summary.append(f"  - BIDS Version: {desc.get('BIDSVersion', 'MISSING')}")
    except Exception as e:
        summary.append(f"\n- Could not read dataset_description.json: {e}")

    # --- 2. Cohort Size ---
    summary.append("\n--- Cohort Size ---")
    summary.append(f"  - Number of Subjects: {len(layout.get_subjects())}")
    sessions = layout.get_sessions()
    if sessions:
        summary.append(f"  - Number of Sessions: {len(sessions)}")

    # --- 3. Available Data ---
    summary.append("\n--- Available Data ---")
    summary.append(f"  - Modalities: {', '.join(layout.get_datatypes())}")
    summary.append(f"  - Tasks: {', '.join(layout.get_tasks()) if layout.get_tasks() else 'No tasks found'}")

    # --- 4. Representative Scan Parameters (from first BOLD file) ---
    summary.append("\n--- Representative Scan Parameters (assumed consistent) ---")
    bold_files = layout.get(suffix='bold', extension=['nii.gz', 'nii'])
    if bold_files:
        first_bold = bold_files[0]
        meta = first_bold.get_metadata()
        header = first_bold.get_image().header
        dims = header.get_data_shape()
        zooms = [round(z, 2) for z in header.get_zooms()]

        summary.append(f"  - Repetition Time (TR): {meta.get('RepetitionTime', 'MISSING')}")
        summary.append(f"  - Echo Time (TE): {meta.get('EchoTime', 'MISSING')}")
        summary.append(f"  - Voxel Size (mm): {'x'.join(map(str, zooms[:3]))}")
        summary.append(f"  - Dimensions: {'x'.join(map(str, dims))}")
        summary.append(f"  - Phase Encoding Direction: {meta.get('PhaseEncodingDirection', 'MISSING')}")
        
        slice_timing = meta.get('SliceTiming', 'MISSING')
        if isinstance(slice_timing, list):
            summary.append(f"  - Slice Timing: Available ({len(slice_timing)} slices)")
        else:
            summary.append(f"  - Slice Timing: {slice_timing}")
    else:
        summary.append("  - No BOLD files found to extract parameters.")

    # --- 5. Fieldmap Availability ---
    summary.append("\n--- Fieldmap Availability ---")
    fmap_files = layout.get(datatype='fmap')
    if fmap_files:
        fmap_types = sorted(list(set(f.entities['suffix'] for f in fmap_files)))
        summary.append(f"  - Fieldmaps found. Types: {', '.join(fmap_types)}")
    else:
        summary.append("  - No fieldmaps found in the dataset.")

    # --- 6. Dataset-wide Parameter Consistency (aggregate over all BOLD runs) ---
    try:
        if bold_files:
            run_infos = []
            for bf in bold_files:
                try:
                    info = extract_bold_run_info(layout, bf, fmap_files=None, slicetiming_preview_n=10)
                    run_infos.append(info)
                except Exception:
                    continue
            if run_infos:
                df = pd.DataFrame(run_infos)
                summary.append("\n--- Dataset-wide Parameter Consistency ---")
                summary.append(f"  - TR (s): {summarize_value_counts(df['RepetitionTime'])}")
                summary.append(f"  - TE (s): {summarize_value_counts(df['EchoTime'])}")
                summary.append(f"  - FlipAngle (deg): {summarize_value_counts(df['FlipAngle'])}")
                summary.append(f"  - MagneticFieldStrength (T): {summarize_value_counts(df['MagneticFieldStrength'])}")
                summary.append(f"  - PhaseEncodingDirection: {summarize_value_counts(df['PhaseEncodingDirection'])}")
                summary.append(f"  - EffectiveEchoSpacing (s): {summarize_value_counts(df['EffectiveEchoSpacing'])}")
                summary.append(f"  - TotalReadoutTime (s): {summarize_value_counts(df['TotalReadoutTime'])}")
                summary.append(f"  - MultibandAccelerationFactor: {summarize_value_counts(df['MultibandAccelerationFactor'])}")
                if 'VoxelSizeStr' in df.columns:
                    summary.append(f"  - VoxelSize (mm): {summarize_value_counts(df['VoxelSizeStr'])}")
    except Exception as e:
        summary.append(f"\n- Could not compute dataset-wide parameter summary: {e}")

    return "\n".join(summary)


def get_subject_summary(bids_dir, subject_id):
    """
    Generates a structured summary for a specific subject, optimized for LLM consumption.
    """
    layout = BIDSLayout(bids_dir, validate=False)

    # --- 1. Validate and Format Subject ID ---
    clean_subject_id = subject_id[4:] if subject_id.startswith('sub-') else subject_id
    if clean_subject_id not in layout.get_subjects():
        return f"Error: Subject 'sub-{clean_subject_id}' not found in the dataset."
    # Use the full 'sub-XX' format for consistency
    full_subject_id = f"sub-{clean_subject_id}"

    # --- 2. Initialize Summary Sections ---
    preprocessing_summary = ["[PREPROCESSING_DATA]"]
    phenotype_summary = [
        "[PHENOTYPE_FOR_MODELING]",
        "# This data is for modeling reference only and must not be used for preprocessing decisions."
    ]

    # --- 3. Populate Phenotype Section ---
    try:
        participants_files = layout.get(suffix='participants', extension='tsv')
        if participants_files:
            participants_df = participants_files[0].get_df()
            subject_info = participants_df[participants_df['participant_id'] == full_subject_id]
            if not subject_info.empty:
                for col, val in subject_info.iloc[0].items():
                    phenotype_summary.append(f"  - {col}: {val}")
            else:
                phenotype_summary.append("  - No demographic information found for this subject.")
        else:
            phenotype_summary.append("  - participants.tsv file not found.")
    except Exception as e:
        phenotype_summary.append(f"  - Could not process participants.tsv: {e}")

    # --- 4. Populate Preprocessing Section (Run-by-Run) ---
    bold_files = layout.get(subject=clean_subject_id, suffix='bold', extension=['nii.gz', 'nii'])

    if not bold_files:
        preprocessing_summary.append("  - No BOLD scans found for this subject.")
    else:
        # Find all fieldmaps for the subject once
        fmap_files = layout.get(subject=clean_subject_id, datatype='fmap')

        run_infos = []
        for i, bold_file in enumerate(bold_files):
            info = extract_bold_run_info(layout, bold_file, fmap_files=fmap_files, slicetiming_preview_n=10)
            run_infos.append(info)

            preprocessing_summary.append(f"\n  --- Run {i+1} ---")
            preprocessing_summary.append(f"  - File: {info.get('Filename', 'MISSING')}")
            preprocessing_summary.append(f"  - Repetition Time (TR, s): {_fmt_num(info.get('RepetitionTime'))}")
            preprocessing_summary.append(f"  - Echo Time (TE, s): {_fmt_num(info.get('EchoTime'))}")
            preprocessing_summary.append(f"  - FlipAngle (deg): {_fmt_num(info.get('FlipAngle'))}")
            preprocessing_summary.append(f"  - Number of Volumes: {info.get('DimT', 'MISSING')}")

            preprocessing_summary.append(f"  - Voxel Size (mm): {info.get('VoxelSizeStr', 'MISSING')}")
            dims_str = 'x'.join(str(x) for x in [info.get('DimX'), info.get('DimY'), info.get('DimZ'), info.get('DimT')] if x is not None)
            preprocessing_summary.append(f"  - Dimensions: {dims_str if dims_str else 'MISSING'}")

            preprocessing_summary.append(f"  - Phase Encoding Direction: {info.get('PhaseEncodingDirection', 'MISSING')}")
            preprocessing_summary.append(f"  - EffectiveEchoSpacing (s): {_fmt_num(info.get('EffectiveEchoSpacing'))}")
            preprocessing_summary.append(f"  - TotalReadoutTime (s): {_fmt_num(info.get('TotalReadoutTime'))}")
            preprocessing_summary.append(f"  - PE Steps: {info.get('PE_Steps', 'MISSING')}")

            st_count = info.get('SliceTimingCount', 0)
            if st_count and st_count > 0:
                preprocessing_summary.append(
                    f"  - SliceTiming: Available ({st_count} slices), pattern={info.get('SliceTimingPattern')}, preview=[{info.get('SliceTimingPreview')}]"
                )
            else:
                preprocessing_summary.append("  - SliceTiming: MISSING")

            fmap_status = (
                f"Available (type: {info.get('FieldmapType', 'unknown')})" if info.get('FieldmapAvailable') else "Not Available"
            )
            preprocessing_summary.append(f"  - Fieldmap: {fmap_status}")

            # Multiband / Parallel
            preprocessing_summary.append(
                f"  - MultibandAccelerationFactor: {_fmt_num(info.get('MultibandAccelerationFactor'))}"
            )
            prfi_val = info.get('ParallelReductionFactorInPlane')
            preprocessing_summary.append(
                f"  - ParallelReductionFactorInPlane: {_fmt_num(prfi_val)}"
            )

            # Scanner info (concise)
            preprocessing_summary.append(
                f"  - Scanner: {info.get('Manufacturer', 'MISSING')} {info.get('ManufacturersModelName', 'MISSING')}"
            )

            # Intelligent Hints
            dimt = info.get('DimT')
            if isinstance(dimt, int) and dimt < 100:
                preprocessing_summary.append(
                    f"  - [INFO] Low volume count ({dimt}) detected. Consider broader band-pass filter settings."
                )
            vx = (info.get('VoxelSizeX'), info.get('VoxelSizeY'), info.get('VoxelSizeZ'))
            if all(isinstance(v, (int, float)) for v in vx):
                if vx[2] > vx[0] * 1.5 or vx[2] > vx[1] * 1.5:
                    preprocessing_summary.append(
                        f"  - [INFO] Anisotropic voxels ({info.get('VoxelSizeStr')} mm) detected. Avoid large isotropic smoothing kernels."
                    )

    # --- 5. Per-Subject Parameter Consistency ---
    try:
        if bold_files:
            df_sub = pd.DataFrame(run_infos)
            preprocessing_summary.append("\n  --- Per-Subject Parameter Consistency ---")
            preprocessing_summary.append(f"  - TR (s): {summarize_value_counts(df_sub['RepetitionTime'])}")
            preprocessing_summary.append(f"  - TE (s): {summarize_value_counts(df_sub['EchoTime'])}")
            preprocessing_summary.append(f"  - FlipAngle (deg): {summarize_value_counts(df_sub['FlipAngle'])}")
            preprocessing_summary.append(f"  - MagneticFieldStrength (T): {summarize_value_counts(df_sub['MagneticFieldStrength'])}")
            preprocessing_summary.append(f"  - PhaseEncodingDirection: {summarize_value_counts(df_sub['PhaseEncodingDirection'])}")
            preprocessing_summary.append(f"  - EffectiveEchoSpacing (s): {summarize_value_counts(df_sub['EffectiveEchoSpacing'])}")
            preprocessing_summary.append(f"  - TotalReadoutTime (s): {summarize_value_counts(df_sub['TotalReadoutTime'])}")
            preprocessing_summary.append(f"  - MultibandAccelerationFactor: {summarize_value_counts(df_sub['MultibandAccelerationFactor'])}")
            if 'VoxelSizeStr' in df_sub.columns:
                preprocessing_summary.append(f"  - VoxelSize (mm): {summarize_value_counts(df_sub['VoxelSizeStr'])}")
    except Exception:
        pass

    # --- 5. Combine and Return --- 
    final_summary = [f"--- Summary for {full_subject_id} ---"] + preprocessing_summary + ["\n"] + phenotype_summary
    return "\n".join(final_summary)


def main():
    """Test function for get_subject_summary and get_bids_summary."""
    # --- Parameters for testing ---
    bids_directory = '/mnt/sda1/zhangyan/openneuro/ds002748'
    subject_to_test = '05'  # Example subject, change if needed
    # -----------------------------

    print(f"--- Testing get_subject_summary ---")
    print(f"BIDS Directory: {bids_directory}")
    print(f"Subject ID: {subject_to_test}")
    print("-" * 30)

    # Call the function and get the subject summary
    subject_summary_output = get_subject_summary(bids_directory, subject_to_test)
    print(subject_summary_output)
    print("\n--- Test for get_subject_summary complete ---")


    # print("\n" + "="*40 + "\n")

    print(f"--- Testing get_bids_summary ---")
    print(f"BIDS Directory: {bids_directory}")
    print("-" * 30)

    # Call the function and get the dataset summary
    dataset_summary_output = get_bids_summary(bids_directory)
    print(dataset_summary_output)
    print("\n--- Test for get_bids_summary complete ---")


if __name__ == "__main__":
    main()


