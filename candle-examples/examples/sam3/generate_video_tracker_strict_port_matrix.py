#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "List, emit, or run the canonical upstream SAM3 video strict-port "
            "reference matrix defined in video_tracker_strict_port_matrix.json."
        )
    )
    parser.add_argument(
        "--matrix",
        default="candle-examples/examples/sam3/video_tracker_strict_port_matrix.json",
        help="Path to the strict-port matrix manifest JSON.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the bundles in the matrix and exit.",
    )
    parser.add_argument(
        "--bundle",
        action="append",
        default=[],
        help="Bundle name to include. Repeat to select multiple bundles. Defaults to all required bundles.",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include bundles marked optional or required_if_reachable.",
    )
    parser.add_argument(
        "--emit-scenarios-dir",
        help="Write each scenario JSON into this directory and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the export commands instead of running them.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to invoke export_reference.py.",
    )
    parser.add_argument("--sam3-repo", help="Path to the upstream SAM3 repository.")
    parser.add_argument("--checkpoint", help="Path to the SAM3 checkpoint.")
    parser.add_argument(
        "--video",
        help="Input video path or extracted-frame directory passed to export_reference.py.",
    )
    parser.add_argument(
        "--output-root",
        default="candle-examples/examples/sam3/strict_port_reference_matrix",
        help="Root directory where generated reference bundles will be written.",
    )
    parser.add_argument(
        "--video-frame-count",
        type=int,
        help="Override the matrix manifest's default video frame count.",
    )
    parser.add_argument(
        "--export-script",
        default="candle-examples/examples/sam3/export_reference.py",
        help="Path to export_reference.py.",
    )
    return parser.parse_args()


def load_matrix(path: Path):
    matrix = json.loads(path.read_text(encoding="utf-8"))
    bundles = matrix.get("bundles", [])
    if not bundles:
        raise ValueError(f"matrix manifest {path} does not contain any bundles")
    return matrix


def selected_bundles(matrix, requested_names, include_optional):
    bundles = matrix["bundles"]
    if requested_names:
        requested = set(requested_names)
        selected = [bundle for bundle in bundles if bundle["name"] in requested]
        missing = sorted(requested - {bundle["name"] for bundle in selected})
        if missing:
            raise ValueError(f"unknown bundle names requested: {', '.join(missing)}")
        return selected
    selected = []
    for bundle in bundles:
        if bundle.get("required", True):
            selected.append(bundle)
        elif include_optional:
            selected.append(bundle)
    return selected


def list_bundles(bundles):
    for bundle in bundles:
        flags = []
        if bundle.get("required", True):
            flags.append("required")
        if bundle.get("required_if_reachable"):
            flags.append("required-if-reachable")
        flag_text = ", ".join(flags) if flags else "optional"
        print(f"{bundle['name']} [{flag_text}]")
        print(f"  {bundle.get('description', '')}")
        covers = bundle.get("covers", [])
        if covers:
            print(f"  covers: {', '.join(covers)}")


def emit_scenarios(bundles, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for bundle in bundles:
        scenario_path = output_dir / f"{bundle['name']}.json"
        scenario_path.write_text(
            json.dumps(bundle["scenario"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(scenario_path)


def build_export_command(args, bundle, scenario_path: Path, frame_count: int):
    output_dir = Path(args.output_root) / bundle["name"]
    cmd = [
        args.python,
        args.export_script,
        "--sam3-repo",
        args.sam3_repo,
        "--checkpoint",
        args.checkpoint,
        "--video",
        args.video,
        "--video-scenario",
        str(scenario_path),
        "--video-debug-bundle",
        "--video-frame-count",
        str(frame_count),
        "--output-dir",
        str(output_dir),
    ]
    return cmd


def run_exports(args, bundles, frame_count: int):
    missing = [
        flag
        for flag in ("sam3_repo", "checkpoint", "video")
        if getattr(args, flag) is None
    ]
    if missing:
        raise ValueError(
            "running the matrix requires "
            + ", ".join(f"--{flag.replace('_', '-')}" for flag in missing)
        )
    with tempfile.TemporaryDirectory(prefix="sam3_strict_port_matrix_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for bundle in bundles:
            scenario_path = tmpdir_path / f"{bundle['name']}.json"
            scenario_path.write_text(
                json.dumps(bundle["scenario"], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            cmd = build_export_command(args, bundle, scenario_path, frame_count)
            if args.dry_run:
                print(shlex.join(cmd))
                continue
            print(f"running {bundle['name']}")
            subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    matrix_path = Path(args.matrix).expanduser().resolve()
    matrix = load_matrix(matrix_path)
    bundles = selected_bundles(matrix, args.bundle, args.include_optional)
    frame_count = args.video_frame_count or int(matrix.get("video_frame_count", 30))

    if args.list:
        list_bundles(bundles)
        return

    if args.emit_scenarios_dir is not None:
        emit_scenarios(bundles, Path(args.emit_scenarios_dir).expanduser().resolve())
        return

    run_exports(args, bundles, frame_count)


if __name__ == "__main__":
    main()
