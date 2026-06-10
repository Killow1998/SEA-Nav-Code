#!/usr/bin/env python3
"""Run exact hard-room eval for multiple checkpoints and aggregate results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOW_LEVEL_POLICY = REPO_ROOT / "training" / "isaac_lab" / "low_level_policies" / "robotlab_go2_flat_20260527" / "policy.pt"
ROOM_RESOLUTION = 0.1
CONTROL_DT = 0.02


def _apply_eval_protocol(args: argparse.Namespace) -> None:
    if args.eval_protocol == "custom":
        return
    if args.eval_protocol == "strict_env":
        args.episode_length_s = 60.0
        args.max_steps = 3000
        args.stay_steps = 150
        args.disable_contact_termination = False
        args.policy_stop_radius = -1.0
    elif args.eval_protocol == "official_play_style":
        args.episode_length_s = 40.0
        args.max_steps = 2000
        args.stay_steps = 500
        args.disable_contact_termination = True
        args.policy_stop_radius = -1.0
    elif args.eval_protocol == "assist_stop":
        args.episode_length_s = 40.0
        args.max_steps = 2000
        args.stay_steps = 500
        args.disable_contact_termination = True
        args.policy_stop_radius = 0.45
        args.policy_stop_mode = "zero"
    else:
        raise ValueError(f"Unsupported eval protocol: {args.eval_protocol}")


@dataclass(frozen=True)
class ModelSpec:
    label: str
    checkpoint: Path
    low_level_policy: Path
    command_clip: float


def _parse_model_spec(value: str) -> ModelSpec:
    parts = value.split(",")
    if len(parts) not in (2, 4):
        raise argparse.ArgumentTypeError(
            "--model must be LABEL,CHECKPOINT or LABEL,CHECKPOINT,LOW_LEVEL_POLICY,COMMAND_CLIP"
        )
    label = parts[0].strip()
    if not label:
        raise argparse.ArgumentTypeError("model label cannot be empty")
    checkpoint = Path(parts[1]).expanduser()
    low_level_policy = DEFAULT_LOW_LEVEL_POLICY
    command_clip = 1.0
    if len(parts) == 4:
        low_level_policy = Path(parts[2]).expanduser()
        try:
            command_clip = float(parts[3])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid command clip: {parts[3]}") from exc
    return ModelSpec(label=label, checkpoint=checkpoint, low_level_policy=low_level_policy, command_clip=command_clip)


def _existing_run_dirs(log_root: Path) -> set[Path]:
    return set(log_root.glob("manual_reward_probe_hard_room_eval/*"))


def _latest_new_run_dir(log_root: Path, before: set[Path]) -> Path:
    candidates = [
        path
        for path in log_root.glob("manual_reward_probe_hard_room_eval/*")
        if path not in before and (path / "summary.json").is_file() and (path / "trace.jsonl").is_file()
    ]
    if not candidates:
        raise RuntimeError(f"no new eval output found under {log_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _effective_speed(row: dict) -> float | None:
    if row.get("done_reason") != "goal_hold":
        return None
    done_step = row.get("done_step")
    start = row.get("start_cell")
    goal = row.get("goal_cell")
    if done_step is None or not start or not goal or done_step <= 0:
        return None
    distance = math.hypot(float(goal[0]) - float(start[0]), float(goal[1]) - float(start[1])) * ROOM_RESOLUTION
    return distance / (float(done_step) * CONTROL_DT)


def _arrival_time(row: dict) -> float | None:
    if row.get("done_reason") != "goal_hold":
        return None
    first_reach_step = row.get("first_reach_step")
    if first_reach_step is None:
        return None
    return float(first_reach_step) * CONTROL_DT


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _summarize_run(args: argparse.Namespace, model: ModelSpec, level: int, run_dir: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = _load_jsonl(run_dir / "trace.jsonl")
    speeds = [value for row in rows if (value := _effective_speed(row)) is not None]
    arrival_times = [value for row in rows if (value := _arrival_time(row)) is not None]
    return {
        "model": model.label,
        "checkpoint": str(model.checkpoint),
        "repro_mode": args.repro_mode,
        "eval_protocol": args.eval_protocol,
        "actuator_mode": args.actuator_mode,
        "robot_asset_source": args.robot_asset_source,
        "low_level_controller": args.low_level_controller,
        "low_level_policy": str(model.low_level_policy) if args.low_level_controller == "robotlab" else None,
        "robotlab_command_clip": model.command_clip if args.low_level_controller == "robotlab" else None,
        "cbf_fov_deg": args.cbf_fov_deg,
        "episode_length_s": args.episode_length_s,
        "stay_steps": args.stay_steps,
        "disable_contact_termination": args.disable_contact_termination,
        "level": level,
        "episodes": summary.get("episodes"),
        "success_count": summary.get("success_count"),
        "success_rate": summary.get("success_rate"),
        "collision_failures": summary.get("collision_failures"),
        "stand_failures": summary.get("stand_failures"),
        "fall_failures": summary.get("fall_failures"),
        "timeout_failures": summary.get("timeout_failures"),
        "max_step_failures": summary.get("max_step_failures"),
        "mean_success_speed_mps": _mean(speeds),
        "mean_arrival_time_s": _mean(arrival_times),
        "mean_min_distance": summary.get("mean_min_distance"),
        "eval_output": str(run_dir),
    }


def _run_eval(args: argparse.Namespace, model: ModelSpec, level: int) -> Path:
    checkpoint = model.checkpoint if model.checkpoint.is_absolute() else REPO_ROOT / model.checkpoint
    low_level_policy = model.low_level_policy if model.low_level_policy.is_absolute() else REPO_ROOT / model.low_level_policy
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if args.low_level_controller == "robotlab" and not low_level_policy.is_file():
        raise FileNotFoundError(f"low-level policy not found: {low_level_policy}")

    log_root = REPO_ROOT / "logs" / "isaac_lab" / args.comparison_name / model.label / f"level_{level}"
    log_root.mkdir(parents=True, exist_ok=True)
    before = _existing_run_dirs(log_root)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "training" / "isaac_lab" / "manual_reward_probe.py"),
        "--headless",
        "--repro-mode",
        args.repro_mode,
        "--scenario",
        "hard_room_eval",
        "--controller-mode",
        "policy",
        "--checkpoint",
        str(checkpoint),
        "--episodes",
        str(args.episodes),
        "--num-envs",
        str(args.num_envs),
        "--seed",
        str(args.seed),
        "--eval-obstacle-level",
        str(level),
        "--actuator-mode",
        args.actuator_mode,
        "--low-level-controller",
        args.low_level_controller,
        "--robot-asset-source",
        args.robot_asset_source,
        "--episode-length-s",
        str(args.episode_length_s),
        "--max-steps",
        str(args.max_steps),
        "--policy-stop-radius",
        str(args.policy_stop_radius),
        "--policy-stop-mode",
        args.policy_stop_mode,
        "--cbf-fov-deg",
        str(args.cbf_fov_deg),
        "--sim-device",
        args.sim_device,
        "--log-root",
        str(log_root),
    ]
    if args.stay_steps >= 0:
        command.extend(["--stay-steps", str(args.stay_steps)])
    if args.disable_contact_termination:
        command.append("--disable-contact-termination")
    if args.low_level_controller == "robotlab":
        command.extend(
            [
                "--robotlab-low-level-policy",
                str(low_level_policy),
                "--robotlab-command-clip",
                str(model.command_clip),
            ]
        )
    print(
        "[eval] "
        f"model={model.label} "
        f"level={level} "
        f"checkpoint={checkpoint} "
        f"repro_mode={args.repro_mode} "
        f"eval_protocol={args.eval_protocol} "
        f"actuator_mode={args.actuator_mode} "
        f"low_level_controller={args.low_level_controller} "
        f"robot_asset_source={args.robot_asset_source} "
        f"cbf_fov_deg={args.cbf_fov_deg}",
        flush=True,
    )
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    return _latest_new_run_dir(log_root, before)


def _write_outputs(output_dir: Path, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    if rows:
        with (output_dir / "comparison_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    markdown = [
        "| Model | Level 3 | Level 6 | Level 9 | Total | Collisions | Stand | Fall | Timeout | Max-step | Mean success speed | Mean arrival time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, model_rows in by_model.items():
        level_map = {int(row["level"]): row for row in model_rows}
        total_success = sum(int(row["success_count"]) for row in model_rows)
        total_episodes = sum(int(row["episodes"]) for row in model_rows)
        speeds = [row["mean_success_speed_mps"] for row in model_rows if row["mean_success_speed_mps"] is not None]
        arrivals = [row["mean_arrival_time_s"] for row in model_rows if row["mean_arrival_time_s"] is not None]
        markdown.append(
            "| {model} | {l3} | {l6} | {l9} | {total} | {collision} | {stand} | {fall} | {timeout} | {max_step} | {speed} | {arrival} |".format(
                model=model,
                l3=_format_level(level_map.get(3)),
                l6=_format_level(level_map.get(6)),
                l9=_format_level(level_map.get(9)),
                total=f"{total_success}/{total_episodes}",
                collision=sum(int(row.get("collision_failures") or 0) for row in model_rows),
                stand=sum(int(row.get("stand_failures") or 0) for row in model_rows),
                fall=sum(int(row.get("fall_failures") or 0) for row in model_rows),
                timeout=sum(int(row.get("timeout_failures") or 0) for row in model_rows),
                max_step=sum(int(row.get("max_step_failures") or 0) for row in model_rows),
                speed=_format_float(_mean(speeds)),
                arrival=_format_float(_mean(arrivals)),
            )
        )
    (output_dir / "comparison_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def _format_level(row: dict | None) -> str:
    if row is None:
        return "-"
    return f"{row['success_count']}/{row['episodes']}"


def _format_float(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-name", required=True)
    parser.add_argument("--model", action="append", type=_parse_model_spec, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[3, 6, 9])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--episode-length-s", type=float, default=60.0)
    parser.add_argument("--stay-steps", type=int, default=-1)
    parser.add_argument("--disable-contact-termination", action="store_true", default=False)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument(
        "--eval-protocol",
        choices=("custom", "strict_env", "official_play_style", "assist_stop"),
        default="custom",
    )
    parser.add_argument("--repro-mode", choices=("none", "gym_equiv"), default="none")
    parser.add_argument("--actuator-mode", choices=("implicit", "ideal_pd", "gym_torque", "robotlab_dc"), default="robotlab_dc")
    parser.add_argument("--low-level-controller", choices=("sea_nav_jit", "robotlab"), default="robotlab")
    parser.add_argument("--robot-asset-source", choices=("converted_urdf", "native_go2"), default="native_go2")
    parser.add_argument("--policy-stop-radius", type=float, default=None)
    parser.add_argument("--policy-stop-mode", choices=("zero", "linear"), default="zero")
    parser.add_argument("--cbf-fov-deg", type=float, default=None)
    args = parser.parse_args()
    if args.repro_mode == "gym_equiv":
        args.robot_asset_source = "converted_urdf"
        args.actuator_mode = "gym_torque"
        args.low_level_controller = "sea_nav_jit"
        if args.policy_stop_radius is None:
            args.policy_stop_radius = -1.0
        if args.cbf_fov_deg is None:
            args.cbf_fov_deg = 180.0
    elif args.cbf_fov_deg is None:
        args.cbf_fov_deg = 240.0
    if args.policy_stop_radius is None:
        args.policy_stop_radius = 0.45
    _apply_eval_protocol(args)

    rows: list[dict] = []
    for model in args.model:
        for level in args.levels:
            run_dir = _run_eval(args, model, level)
            rows.append(_summarize_run(args, model, level, run_dir))

    output_dir = REPO_ROOT / "logs" / "isaac_lab" / args.comparison_name
    _write_outputs(output_dir, rows)
    print(f"[done] wrote {output_dir / 'comparison_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
