"""Handle local artifact storage for pipeline outputs.

Replaces AWS S3 with a free local export system.
Pipeline outputs are copied to an 'exports/' folder inside the project,
making it easy to inspect, share, or archive results without any cloud account.
"""

from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EXPORTS_DIR = PROJECT_ROOT / "exports"

PIPELINE_OUTPUT_FILES = [
    DATA_DIR / "validated_annotations.csv",
    DATA_DIR / "validation_errors.csv",
    DATA_DIR / "annotator_stats.csv",
    DATA_DIR / "cohen_kappa_scores.csv",
    DATA_DIR / "agreement_summary.json",
    DATA_DIR / "percent_agreement.csv",
    DATA_DIR / "contested_questions.csv",
    DATA_DIR / "llm_eval_scores.csv",
]



def export_file(source_path: Path, destination_dir: Path) -> bool:
    """Copy a pipeline output file to the exports directory."""

    if not source_path.exists():
        print(f"Skipped (not found): {source_path.name}")
        return False
    destination_dir.mkdir(parents=True, exist_ok=True)
    dest_path = destination_dir / source_path.name
    shutil.copy2(source_path, dest_path)
    print(f"Exported: {source_path.name} → {dest_path}")
    return True



def export_pipeline_outputs() -> dict[str, str]:
    """Copy all available pipeline output files to the exports/ folder."""

    results: dict[str, str] = {}
    for local_path in PIPELINE_OUTPUT_FILES:
        success = export_file(local_path, EXPORTS_DIR)
        results[local_path.name] = "success" if success else "skipped"
    return results



def list_exports() -> list[dict[str, object]]:
    """List all files currently in the exports/ folder."""

    if not EXPORTS_DIR.exists():
        print("No exports folder found. Run the pipeline first.")
        return []

    items = sorted(EXPORTS_DIR.iterdir())
    if not items:
        print("Exports folder is empty.")
        return []

    results = []
    print(f"\n{'File':<40} {'Size (KB)':>10}")
    print("-" * 52)
    for item in items:
        if item.is_file():
            size_kb = round(item.stat().st_size / 1024, 2)
            print(f"{item.name:<40} {size_kb:>10.2f}")
            results.append({"name": item.name, "size_kb": size_kb, "path": str(item)})
    return results



def check_exports_ready() -> bool:
    """Verify that the exports directory exists and contains files."""

    if EXPORTS_DIR.exists() and any(EXPORTS_DIR.iterdir()):
        print(f"Exports ready at: {EXPORTS_DIR}")
        return True
    print("Exports directory is empty or does not exist.")
    return False



def main() -> None:
    """Export all available pipeline outputs to the local exports/ folder."""

    print(f"Exporting pipeline outputs to: {EXPORTS_DIR}")
    results = export_pipeline_outputs()
    success_count = sum(1 for s in results.values() if s == "success")
    skip_count = sum(1 for s in results.values() if s == "skipped")
    print(f"\nExport complete: {success_count} exported, {skip_count} skipped.")
    list_exports()


if __name__ == "__main__":
    main()
