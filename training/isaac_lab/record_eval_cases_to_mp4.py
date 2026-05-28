from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "logs"
    / "isaac_lab"
    / "Go2_pos_rough_isaaclab_robotlab_contract_curriculum"
    / "05_27_22-33-03_256env_16000it_adaptive_limited_curriculum_equiv_steps_v2"
    / "model_3500.pt"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "logs" / "isaac_lab" / "eval_case_mp4"
DEFAULT_DIFFICULTY_LEVELS = {"easy": 3, "medium": 6, "hard": 12}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record easy/medium/hard SEA-Nav IsaacLab eval cases to MP4 in one command."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--difficulties", nargs="+", default=["easy", "medium", "hard"], choices=tuple(DEFAULT_DIFFICULTY_LEVELS)
    )
    parser.add_argument("--easy-level", type=int, default=DEFAULT_DIFFICULTY_LEVELS["easy"])
    parser.add_argument("--medium-level", type=int, default=DEFAULT_DIFFICULTY_LEVELS["medium"])
    parser.add_argument("--hard-level", type=int, default=DEFAULT_DIFFICULTY_LEVELS["hard"])
    parser.add_argument("--max-record-duration-s", type=float, default=60.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--video-width", type=int, default=1024)
    parser.add_argument("--video-height", type=int, default=1024)
    parser.add_argument("--camera-height", type=float, default=16.0)
    parser.add_argument("--recording-backend", choices=("live", "ovd"), default="live")
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--episode-length-s", type=float, default=120.0)
    parser.add_argument("--policy-stop-radius", type=float, default=0.45)
    parser.add_argument("--eval-room-profile", choices=("random", "visual_progressive"), default="visual_progressive")
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--timeout-s", type=float, default=360.0)
    parser.add_argument("--sim-device", type=str, default="cuda:0")
    return parser.parse_args()


def _isaaclab_anim_recordings_root() -> Path:
    import isaaclab

    return Path(isaaclab.__file__).resolve().parent / "anim_recordings"


def _isaac_pip_prebundle() -> Path | None:
    site_packages = Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    matches = sorted(
        (site_packages / "isaacsim" / "kit" / "data" / "Kit").glob(
            "Isaac-Sim/*/exts/3/omni.kit.pip_archive-*/pip_prebundle"
        )
    )
    return matches[-1] if matches else None


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    prebundle = _isaac_pip_prebundle()
    if prebundle is not None:
        old_pythonpath = env.get("PYTHONPATH", "")
        paths = [str(prebundle)]
        if old_pythonpath:
            paths.append(old_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("[RUN] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def _find_new_probe_dir(log_root: Path, before: set[Path], after: set[Path]) -> Path:
    created = sorted(after - before, key=lambda path: path.stat().st_mtime)
    for path in reversed(created):
        if (path / "summary.json").is_file() and (path / "trace.jsonl").is_file():
            return path
    candidates = sorted(
        [path for path in log_root.glob("manual_reward_probe_hard_room_eval/*") if (path / "summary.json").is_file()],
        key=lambda path: path.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]
    raise RuntimeError(f"No manual_reward_probe output was produced under {log_root}")


def _find_new_recording(before: set[Path], after: set[Path]) -> Path:
    created = sorted(after - before, key=lambda path: path.stat().st_mtime)
    for path in reversed(created):
        if (path / "baked_animation_recording.usda").is_file():
            return path
    candidates = sorted(
        [path for path in after if (path / "baked_animation_recording.usda").is_file()],
        key=lambda path: path.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]
    raise RuntimeError("No IsaacLab baked animation recording was produced")


def _copy_recording(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("tmp.ovd"))
    tmp_ovd = src / "tmp.ovd"
    if tmp_ovd.exists():
        tmp_ovd.unlink()


def _load_case_result(eval_dir: Path) -> dict[str, object]:
    import numpy as np

    summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    episode_row = None
    with (eval_dir / "trace.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "start_cell" in row and "goal_cell" in row:
                episode_row = row
    if episode_row is None:
        raise RuntimeError(f"No episode row found in {eval_dir / 'trace.jsonl'}")

    room_path = eval_dir / "room.npy"
    room_stats = None
    if room_path.is_file():
        room = np.load(room_path)
        interior = room[1:-1, 1:-1] > 0.1
        visited = np.zeros_like(interior, dtype=bool)
        components = 0
        for i in range(interior.shape[0]):
            for j in range(interior.shape[1]):
                if not interior[i, j] or visited[i, j]:
                    continue
                components += 1
                stack = [(i, j)]
                visited[i, j] = True
                while stack:
                    x, y = stack.pop()
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < interior.shape[0] and 0 <= ny < interior.shape[1]:
                            if interior[nx, ny] and not visited[nx, ny]:
                                visited[nx, ny] = True
                                stack.append((nx, ny))
        room_stats = {
            "room": str(room_path),
            "interior_occupied_cells": int(interior.sum()),
            "interior_obstacle_components": int(components),
        }

    done_step = episode_row.get("done_step")
    done_time_s = None if done_step is None else (int(done_step) + 1) * 0.02
    return {
        "summary": summary,
        "episode": episode_row,
        "room_stats": room_stats,
        "done_time_s": done_time_s,
    }


def _validate_mp4(path: Path) -> dict[str, object]:
    try:
        import cv2
    except ImportError:
        return {"path": str(path), "bytes": path.stat().st_size, "opencv": "unavailable"}

    cap = cv2.VideoCapture(str(path))
    opened = bool(cap.isOpened())
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, frame = cap.read()
    first_frame_mean = None if not ok else float(frame.mean())
    first_frame_std = None if not ok else float(frame.std())
    cap.release()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "opened": opened,
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "first_frame": bool(ok),
        "first_frame_mean": first_frame_mean,
        "first_frame_std": first_frame_std,
    }


def _remove_capture_frames_dir(mp4_path: Path) -> None:
    frames_dir = mp4_path.with_name(f"{mp4_path.stem}_frames")
    if frames_dir.is_dir():
        shutil.rmtree(frames_dir)


def main() -> None:
    args = _parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {args.checkpoint}")

    run_id = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "manifest.json"
    env = _subprocess_env()
    anim_root = _isaaclab_anim_recordings_root()
    anim_root.mkdir(parents=True, exist_ok=True)

    difficulty_levels = {
        "easy": args.easy_level,
        "medium": args.medium_level,
        "hard": args.hard_level,
    }
    results: list[dict[str, object]] = []
    started_at = datetime.now().isoformat(timespec="seconds")
    print(f"[RECORD_MP4] output_dir={output_dir}", flush=True)

    for index, difficulty in enumerate(args.difficulties):
        level = difficulty_levels[difficulty]
        case_dir = output_dir / difficulty
        case_dir.mkdir()
        eval_log_root = case_dir / "eval"
        eval_log_root.mkdir()
        probe_root = eval_log_root / "manual_reward_probe_hard_room_eval"
        probe_root.mkdir(parents=True, exist_ok=True)

        before_probe = {path for path in probe_root.iterdir() if path.is_dir()}
        eval_cmd = [
            sys.executable,
            "-u",
            "training/isaac_lab/manual_reward_probe.py",
            "--scenario",
            "hard_room_eval",
            "--controller-mode",
            "policy",
            "--checkpoint",
            str(args.checkpoint),
            "--episodes",
            "1",
            "--eval-obstacle-level",
            str(level),
            "--eval-room-profile",
            args.eval_room_profile,
            "--actuator-mode",
            "robotlab_dc",
            "--low-level-controller",
            "robotlab",
            "--robot-asset-source",
            "native_go2",
            "--episode-length-s",
            str(args.episode_length_s),
            "--max-steps",
            str(args.max_steps),
            "--policy-stop-radius",
            str(args.policy_stop_radius),
            "--policy-stop-mode",
            "zero",
            "--seed",
            str(args.seed_base + index),
            "--sim-device",
            args.sim_device,
            "--log-root",
            str(eval_log_root),
            "--trace-steps",
            "--headless",
        ]
        started = time.monotonic()
        _run(eval_cmd, env)
        after_probe = {path for path in probe_root.iterdir() if path.is_dir()}
        eval_dir = _find_new_probe_dir(eval_log_root, before_probe, after_probe)
        case_result = _load_case_result(eval_dir)
        done_time_s = case_result["done_time_s"]
        if done_time_s is None:
            record_duration_s = args.max_record_duration_s
        else:
            record_duration_s = min(args.max_record_duration_s, float(done_time_s) + 2.0)
        record_duration_s = max(1.0, record_duration_s)
        record_max_steps = max(1, int(record_duration_s / 0.02) + 50)
        capture_every_steps = max(1, round(50.0 / float(args.fps)))
        actual_record_fps = 50.0 / capture_every_steps

        mp4_path = case_dir / f"{difficulty}_topdown.mp4"
        record_cmd = [
            sys.executable,
            "-u",
            "training/isaac_lab/manual_reward_probe.py",
            "--scenario",
            "hard_room_eval",
            "--controller-mode",
            "policy",
            "--checkpoint",
            str(args.checkpoint),
            "--episodes",
            "1",
            "--case-trace",
            str(eval_dir / "trace.jsonl"),
            "--case-episode",
            "0",
            "--eval-obstacle-level",
            str(level),
            "--eval-room-profile",
            args.eval_room_profile,
            "--actuator-mode",
            "robotlab_dc",
            "--low-level-controller",
            "robotlab",
            "--robot-asset-source",
            "native_go2",
            "--episode-length-s",
            str(args.episode_length_s),
            "--max-steps",
            str(record_max_steps),
            "--policy-stop-radius",
            str(args.policy_stop_radius),
            "--policy-stop-mode",
            "zero",
            "--seed",
            str(args.seed_base + index),
            "--sim-device",
            args.sim_device,
            "--show-start-goal-markers",
        ]
        recording_dir = None
        usd_path = None
        if args.recording_backend == "live":
            record_cmd.extend(
                [
                    "--record-topdown-video",
                    str(mp4_path),
                    "--record-video-fps",
                    str(actual_record_fps),
                    "--record-video-width",
                    str(args.video_width),
                    "--record-video-height",
                    str(args.video_height),
                    "--record-every-n-steps",
                    str(capture_every_steps),
                    "--record-max-frames",
                    str(record_max_steps // capture_every_steps + 2),
                    "--record-camera-height",
                    str(args.camera_height),
                ]
            )
            _run(record_cmd, env)
        else:
            before = {path for path in anim_root.iterdir() if path.is_dir()}
            record_cmd.extend(
                [
                    "--anim_recording_enabled",
                    "--anim_recording_start_time",
                    "0",
                    "--anim_recording_stop_time",
                    str(record_duration_s),
                ]
            )
            _run(record_cmd, env)
            after = {path for path in anim_root.iterdir() if path.is_dir()}
            recording_src = _find_new_recording(before, after)
            ovd_dir = case_dir / "ovd"
            _copy_recording(recording_src, ovd_dir)

            usd_path = ovd_dir / "baked_animation_recording.usda"
            render_cmd = [
                sys.executable,
                "-u",
                "training/isaac_lab/render_usd_animation_to_mp4.py",
                "--usd",
                str(usd_path),
                "--output",
                str(mp4_path),
                "--start-time",
                "0",
                "--end-time",
                str(record_duration_s),
                "--fps",
                str(args.fps),
                "--video-width",
                str(args.video_width),
                "--video-height",
                str(args.video_height),
                "--camera-height",
                str(args.camera_height),
                "--timeout-s",
                str(args.timeout_s),
            ]
            _run(render_cmd, env)
            _remove_capture_frames_dir(mp4_path)
            recording_dir = ovd_dir
        validation = _validate_mp4(mp4_path)
        elapsed_s = time.monotonic() - started
        result = {
            "difficulty": difficulty,
            "obstacle_level": level,
            "eval_room_profile": args.eval_room_profile,
            "seed": args.seed_base + index,
            "record_duration_s": record_duration_s,
            "max_record_duration_s": args.max_record_duration_s,
            "requested_fps": args.fps,
            "actual_record_fps": actual_record_fps,
            "capture_every_steps": capture_every_steps,
            "eval_dir": str(eval_dir),
            "case_result": case_result,
            "episode_complete_in_video": done_time_s is not None and done_time_s <= record_duration_s,
            "recording_backend": args.recording_backend,
            "recording_dir": None if recording_dir is None else str(recording_dir),
            "usd": None if usd_path is None else str(usd_path),
            "mp4": str(mp4_path),
            "validation": validation,
            "elapsed_s": elapsed_s,
        }
        results.append(result)
        manifest_path.write_text(
            json.dumps(
                {
                    "started_at": started_at,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "checkpoint": str(args.checkpoint),
                    "output_dir": str(output_dir),
                    "results": results,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print("[RECORD_MP4] case_done " + json.dumps(result, ensure_ascii=True), flush=True)

    print(f"[RECORD_MP4] done manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
