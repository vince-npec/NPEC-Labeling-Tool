from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ["PetriDish", "FrameIndex", "plant_id"]


def _valid_column(frame: pd.DataFrame) -> str:
    for column in ("ownership_measurement_valid", "ownership_valid"):
        if column in frame.columns:
            return column
    raise RuntimeError("No ownership-valid column was found.")


def summarize(output_dir: Path, baseline_master: Path) -> tuple[Path, Path]:
    output_dir = output_dir.expanduser().resolve()
    baseline_master = baseline_master.expanduser().resolve()
    master_path = output_dir / "npec_per_plant_detail.csv"
    if not master_path.exists():
        master_path = output_dir / "npec_root_measurements_master.csv"
    report_path = output_dir / "npec_corrected_run_report.json"
    timelapse_path = output_dir / "npec_total_root_length_timelapse.csv"
    if not master_path.exists() or not report_path.exists() or not timelapse_path.exists():
        raise RuntimeError(f"Incomplete ownership package: {output_dir}")

    current = pd.read_csv(master_path, low_memory=False)
    baseline = pd.read_csv(baseline_master, low_memory=False)
    plates = set(current["PetriDish"].astype(str))
    baseline = baseline[baseline["PetriDish"].astype(str).isin(plates)].copy()
    current_valid = _valid_column(current)
    baseline_valid = _valid_column(baseline)

    current_rows = current[KEY_COLUMNS + [current_valid]].copy()
    baseline_rows = baseline[KEY_COLUMNS + [baseline_valid]].copy()
    current_rows["_current_valid"] = (
        current_rows[current_valid].fillna(False).astype(bool)
    )
    baseline_rows["_baseline_valid"] = (
        baseline_rows[baseline_valid].fillna(False).astype(bool)
    )
    comparison = baseline_rows[KEY_COLUMNS + ["_baseline_valid"]].merge(
        current_rows[KEY_COLUMNS + ["_current_valid"]],
        on=KEY_COLUMNS,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if set(comparison["_merge"].astype(str)) != {"both"}:
        raise RuntimeError("Baseline/current plant-frame keys are not identical.")

    plate_comparison = (
        comparison.groupby("PetriDish", sort=True)
        .agg(
            baseline_valid_rows=("_baseline_valid", "sum"),
            current_valid_rows=("_current_valid", "sum"),
            rows=("plant_id", "size"),
        )
        .reset_index()
    )
    plate_comparison["valid_row_delta"] = (
        plate_comparison["current_valid_rows"]
        - plate_comparison["baseline_valid_rows"]
    )
    source_column = current.get(
        "ownership_plate_result_source",
        pd.Series("learned_v11", index=current.index),
    )
    source_by_plate = (
        pd.DataFrame(
            {
                "PetriDish": current["PetriDish"].astype(str),
                "final_result_source": source_column.fillna("learned_v11").astype(str),
            }
        )
        .groupby("PetriDish", sort=True)["final_result_source"]
        .first()
    )
    plate_comparison["final_result_source"] = (
        plate_comparison["PetriDish"].map(source_by_plate).fillna("unknown")
    )
    comparison_csv = output_dir / "npec_v11_vs_v8_plate_comparison.csv"
    plate_comparison.to_csv(comparison_csv, index=False)

    run_report = json.loads(report_path.read_text(encoding="utf-8"))
    timelapse = pd.read_csv(timelapse_path, low_memory=False)
    baseline_complete = baseline[baseline_valid].fillna(False).astype(bool).groupby(
        [baseline["PetriDish"], baseline["FrameIndex"]]
    ).all()
    current_complete = current[current_valid].fillna(False).astype(bool).groupby(
        [current["PetriDish"], current["FrameIndex"]]
    ).all()
    plate_results = run_report.get("plate_results", [])
    conservation_area = max(
        (
            float(result.get("conservation", {}).get("max_area_error_px", 0.0))
            for result in plate_results
            if isinstance(result, dict)
        ),
        default=0.0,
    )
    conservation_length = max(
        (
            float(result.get("conservation", {}).get("max_length_error_px", 0.0))
            for result in plate_results
            if isinstance(result, dict)
        ),
        default=0.0,
    )
    rollback_audit = run_report.get("baseline_validity_safeguard", [])
    rollback_audit = rollback_audit if isinstance(rollback_audit, list) else []
    rolled_back = [
        str(item.get("plate", ""))
        for item in rollback_audit
        if isinstance(item, dict) and str(item.get("plate", "")).strip()
    ]
    learned_plates = int(
        (
            plate_comparison["final_result_source"]
            == "learned_v11"
        ).sum()
    )

    markdown_path = output_dir / "VALIDATION_REPORT.md"
    lines = [
        "# NPEC Yang Per-Seedling Reanalysis Validation",
        "",
        f"- Pipeline: `{run_report.get('pipeline_version', '')}`",
        f"- Frames: {int(run_report.get('frames', len(timelapse)))}",
        f"- Plates: {int(run_report.get('plates', current['PetriDish'].nunique()))}",
        f"- Plant-frame rows: {len(current)}",
        f"- Baseline-valid rows retained: {int(comparison['_baseline_valid'].sum())}",
        f"- Final valid rows: {int(comparison['_current_valid'].sum())}",
        f"- Baseline complete frames: {int(baseline_complete.sum())}",
        f"- Final complete frames: {int(current_complete.sum())}",
        f"- Learned v11 plates retained: {learned_plates}",
        f"- Plates rolled back to v8: {len(rolled_back)}",
        f"- Maximum root-area conservation error: {conservation_area:.6f} px",
        f"- Maximum root-length conservation error: {conservation_length:.6f} px",
        "",
        "## Interpretation",
        "",
        "This pass reassigns existing visible root-mask pixels among five seedlings. "
        "It does not rerun semantic root/shoot inference and does not invent roots "
        "hidden by bacterial occlusion. Learned evidence only resolves temporal-graph "
        "ambiguity; clear graph ownership remains unchanged. Any plate that lost a "
        "previously valid plant-frame row was restored in full from the v8 baseline.",
        "",
        "## Rollbacks",
        "",
    ]
    if rolled_back:
        lines.extend(f"- `{plate}`" for plate in rolled_back)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Plate comparison: `{comparison_csv.name}`",
            f"- Run report: `{report_path.name}`",
            f"- Per-plant detail: `{master_path.name}`",
            f"- Timelapse totals: `{timelapse_path.name}`",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return comparison_csv, markdown_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-master", type=Path, required=True)
    args = parser.parse_args()
    comparison_csv, markdown_path = summarize(
        args.output_dir,
        args.baseline_master,
    )
    print(comparison_csv)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
