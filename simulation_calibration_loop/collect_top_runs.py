"""Collect the top-N runs across multiple workspace state.json files.

Reads every state.json under <workspaces_root>, ranks all artifacts by
objective_value (ascending — lower MMD is better), takes the top N, and
either prints the prepare_synth_dataset.py command or runs it directly.

Usage:
    # Print the command (dry-run)
    python collect_top_runs.py <workspaces_root> [--top 40] [--output-dir <path>]

    # Run prepare_synth_dataset.py directly
    python collect_top_runs.py <workspaces_root> --top 40 --output-dir ./top40_dataset --run

    # Also call prepare_synth_dataset.py with --merge (append to existing dataset)
    python collect_top_runs.py <workspaces_root> --top 40 --output-dir ./top40_dataset --run --merge
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_artifacts(workspaces_root: Path) -> list[dict]:
    """Return flat list of artifacts across all state.json files found under root."""
    records = []
    for state_path in sorted(workspaces_root.glob("*/state.json")):
        workspace = state_path.parent.name
        data = json.loads(state_path.read_text())
        for iteration in data.get("iterations", []):
            for artifact in iteration.get("artifacts", []):
                if artifact.get("objective_value") is None:
                    continue
                records.append({
                    "workspace": workspace,
                    "run_id": artifact["run_id"],
                    "objective_value": float(artifact["objective_value"]),
                    "output_dir": artifact["output_dir"],
                })
    return records


def resolve_input_dir(output_dir: Path) -> Path | None:
    """Return the directory that contains rgb_*.png files for prepare_synth_dataset.py."""
    # Files are written flat by Isaac BasicWriter
    if list(output_dir.glob("rgb_*.png")):
        return output_dir
    # Fallback: check Camera subfolder
    for candidate in [output_dir / "Camera", output_dir / "Camera" / "rgb"]:
        if candidate.is_dir() and list(candidate.glob("rgb_*.png")):
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect top-N runs for prepare_synth_dataset.py")
    parser.add_argument("workspaces_root", help="Directory containing workspace_*/state.json files")
    parser.add_argument("--top", type=int, default=40, help="Number of top runs to select (default: 40)")
    parser.add_argument("--output-dir", default=None, help="Output dataset directory for prepare_synth_dataset.py")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--run", action="store_true", help="Actually invoke prepare_synth_dataset.py")
    parser.add_argument("--merge", action="store_true", help="Pass --merge to prepare_synth_dataset.py")
    args = parser.parse_args()

    root = Path(args.workspaces_root).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    records = load_artifacts(root)
    if not records:
        sys.exit("No scored artifacts found in any state.json under the given root.")

    records.sort(key=lambda r: r["objective_value"])
    top = records[: args.top]

    print(f"Total scored artifacts: {len(records)}")
    print(f"Selected top {len(top)}  (MMD range: {top[0]['objective_value']:.6f} – {top[-1]['objective_value']:.6f})\n")

    input_dirs: list[Path] = []
    skipped = 0
    for rec in top:
        output_dir = Path(rec["output_dir"])
        resolved = resolve_input_dir(output_dir)
        if resolved is None:
            print(f"  SKIP (no rgb files found): [{rec['workspace']}] {rec['run_id']}  MMD={rec['objective_value']:.6f}  {output_dir}")
            skipped += 1
            continue
        print(f"  {rec['objective_value']:.6f}  [{rec['workspace']}] {rec['run_id']}  {resolved}")
        input_dirs.append(resolved)

    if skipped:
        print(f"\n  {skipped} run(s) skipped (output dirs not found — are they on a remote machine?)")

    if not input_dirs:
        sys.exit("\nNo valid input directories resolved. Nothing to do.")

    repo_root = Path(__file__).resolve().parent.parent
    prepare_script = repo_root / "od_scripts" / "prepare_synth_dataset.py"

    output_dir_arg = args.output_dir or str(root / f"top{args.top}_dataset")

    # Use the loop venv Python (same as run_with_loop_venv.sh) so that numpy/PIL
    # are available. Fall back to sys.executable if the venv is absent.
    import os
    venv_dir = Path(os.environ.get("LOOP_VENV_DIR", repo_root / ".sim_loop_venv"))
    venv_python = venv_dir / "bin" / "python"
    python_exe = str(venv_python) if venv_python.exists() else sys.executable

    cmd = [
        python_exe,
        str(prepare_script),
        "--input-dirs", *[str(d) for d in input_dirs],
        "--output-dir", output_dir_arg,
        "--val-fraction", str(args.val_fraction),
    ]
    if args.merge:
        cmd.append("--merge")

    print(f"\nCommand:\n  {' '.join(cmd)}\n")

    if args.run:
        subprocess.run(cmd, check=True)
    else:
        print("(pass --run to execute)")


if __name__ == "__main__":
    main()
