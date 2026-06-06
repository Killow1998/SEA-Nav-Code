from __future__ import annotations

import argparse
import json
import math
import traceback
from datetime import datetime
from pathlib import Path

from train import AppLauncher, DEFAULT_USD_DIR, _ensure_isaaclab_imports, _make_go2_usd


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOM_CELLS = 100
ROOM_RESOLUTION = 0.1
ROOM_SIZE = ROOM_CELLS * ROOM_RESOLUTION


def _parse_args():
    parser = argparse.ArgumentParser(description="Open-space low-level tracking probe for SEA-Nav")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=80)
    parser.add_argument("--measure-steps", type=int, default=160)
    parser.add_argument("--episode-length-s", type=float, default=60.0)
    parser.add_argument("--profiles", type=str, default="forward,lateral,yaw,forward_turn")
    parser.add_argument(
        "--custom-command",
        action="append",
        default=[],
        help="Additional profile in the form name:vx,vy,wz. Can be repeated.",
    )
    parser.add_argument("--step-log-interval", type=int, default=0)
    parser.add_argument(
        "--actuator-mode", choices=("implicit", "ideal_pd", "gym_torque", "robotlab_dc"), default="ideal_pd"
    )
    parser.add_argument("--low-level-controller", choices=("sea_nav_jit", "robotlab"), default="sea_nav_jit")
    parser.add_argument(
        "--robotlab-low-level-policy",
        type=str,
        default=str(REPO_ROOT / "training" / "isaac_lab" / "low_level_policies" / "robotlab_go2_flat_20260527" / "policy.pt"),
    )
    parser.add_argument("--robotlab-command-clip", type=float, default=1.0)
    parser.add_argument("--robot-asset-source", choices=("converted_urdf", "native_go2"), default="converted_urdf")
    parser.add_argument("--sim-device", type=str, default="cuda:0")
    parser.add_argument("--usd-dir", type=str, default=str(DEFAULT_USD_DIR))
    parser.add_argument("--log-root", type=str, default=str(REPO_ROOT / "logs" / "isaac_lab"))
    parser.add_argument("--nav-action-scale", type=float, nargs=3, metavar=("VX", "VY", "WZ"), default=(1.0, 1.0, 1.0))
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _make_empty_room(np):
    room = np.zeros((ROOM_CELLS, ROOM_CELLS), dtype=float)
    room[0, :] = 1.0
    room[-1, :] = 1.0
    room[:, 0] = 1.0
    room[:, -1] = 1.0
    room[78:90, 78:90] = 1.0
    return room


def _yaw_from_quat(quat):
    return math.atan2(
        2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
        1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3]),
    )


def _tensor_mean(rows, key):
    return sum(row[key] for row in rows) / max(len(rows), 1)


def _tensor_rmse(rows, actual_key, target_key):
    mse = sum((row[actual_key] - row[target_key]) ** 2 for row in rows) / max(len(rows), 1)
    return mse**0.5


def _profile_specs(selected_names: list[str]):
    all_profiles = [
        {"name": "forward", "command": [1.0, 0.0, 0.0]},
        {"name": "lateral", "command": [0.0, 0.5, 0.0]},
        {"name": "yaw", "command": [0.0, 0.0, 0.8]},
        {"name": "yaw_neg", "command": [0.0, 0.0, -0.8]},
        {"name": "crawl_turn", "command": [0.1, 0.0, 0.8]},
        {"name": "crawl_turn_neg", "command": [0.1, 0.0, -0.8]},
        {"name": "slow_turn", "command": [0.2, 0.0, 0.8]},
        {"name": "slow_turn_neg", "command": [0.2, 0.0, -0.8]},
        {"name": "forward_turn", "command": [0.35, 0.0, 1.0]},
        {"name": "forward_turn_neg", "command": [0.35, 0.0, -1.0]},
        {"name": "reverse", "command": [-0.5, 0.0, 0.0]},
        {"name": "reverse_turn", "command": [-0.3, 0.0, 0.8]},
        {"name": "reverse_turn_neg", "command": [-0.3, 0.0, -0.8]},
        {"name": "lateral_turn", "command": [0.0, 0.35, 0.8]},
        {"name": "lateral_turn_neg", "command": [0.0, -0.35, -0.8]},
    ]
    return [profile for profile in all_profiles if profile["name"] in selected_names]


def _parse_custom_profiles(values: list[str]):
    profiles = []
    for value in values:
        name, sep, payload = value.partition(":")
        if not sep:
            raise ValueError(f"Custom command must be name:vx,vy,wz, got: {value}")
        parts = [part.strip() for part in payload.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Custom command must have 3 numeric components, got: {value}")
        profiles.append(
            {
                "name": name.strip(),
                "command": [float(parts[0]), float(parts[1]), float(parts[2])],
            }
        )
    return profiles


def _write_outputs(log_dir: Path, summary: dict, traces: dict[str, list[dict]]):
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    for profile_name, rows in traces.items():
        with (log_dir / f"{profile_name}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main():
    _ensure_isaaclab_imports()
    args = _parse_args()
    simulation_app = AppLauncher(args).app

    import numpy as np
    import torch

    from sea_nav_env import SeaNavIsaacLabEnv, make_sea_nav_env_cfg

    timestamp = datetime.now().strftime("%m_%d_%H-%M-%S")
    log_dir = Path(args.log_root) / "low_level_tracking_probe" / timestamp
    requested_profiles = [name.strip() for name in args.profiles.split(",") if name.strip()]
    custom_profiles = _parse_custom_profiles(args.custom_command)
    selected_profiles = _profile_specs(requested_profiles)
    selected_profile_names = {profile["name"] for profile in selected_profiles}
    for profile in custom_profiles:
        if profile["name"] in selected_profile_names:
            raise ValueError(f"Duplicate profile name: {profile['name']}")
        selected_profiles.append(profile)
        selected_profile_names.add(profile["name"])

    env = None
    try:
        go2_usd_path = _make_go2_usd(args) if args.robot_asset_source == "converted_urdf" else None
        env_cfg = make_sea_nav_env_cfg(
            go2_usd_path=go2_usd_path,
            num_envs=1,
            seed=args.seed,
            actuator_mode=args.actuator_mode,
            robot_asset_source=args.robot_asset_source,
            low_level_controller=args.low_level_controller,
            robotlab_policy_path=args.robotlab_low_level_policy,
            episode_length_s=args.episode_length_s,
            nav_action_scale=tuple(args.nav_action_scale),
            terrain_rows=1,
            terrain_cols=1,
            obstacle_level=0,
            preset_room=_make_empty_room(np),
            randomize_friction=False,
            randomize_base_mass=True,
            added_mass_range=(0.0, 0.0),
        )
        env_cfg.robotlab_command_clip = args.robotlab_command_clip
        env_cfg.sim.device = args.sim_device
        env_cfg.add_noise = False
        env = SeaNavIsaacLabEnv(cfg=env_cfg, render_mode=None)
        print("[TRACK] env constructed")
        env.reset()
        print("[TRACK] env reset returned")

        env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
        start_xy = torch.tensor([[0.0, 0.0]], dtype=torch.float, device=env.device)
        goal_xy = torch.tensor([[4.0, 0.0]], dtype=torch.float, device=env.device)
        zero_action = torch.zeros((1, env.num_nav_actions), device=env.device)

        all_traces: dict[str, list[dict]] = {}
        profile_summaries = []

        for profile in selected_profiles:
            print(f"[TRACK][{profile['name']}] starting")
            env.reset()
            env.set_manual_start_and_goal(env_ids, start_xy, goal_xy, yaw=0.0, root_height=0.42)
            print(f"[TRACK][{profile['name']}] manual start applied")
            settle_invalid = False
            for settle_idx in range(args.settle_steps):
                _, _, terminated, truncated, _ = env.step(zero_action)
                if args.step_log_interval > 0 and ((settle_idx + 1) % args.step_log_interval == 0):
                    print(f"[TRACK][{profile['name']}] settle_step={settle_idx + 1}")
                if bool((terminated | truncated).item()):
                    settle_invalid = True
                    break

            rows = []
            done_step = None
            done_reason = None
            if not settle_invalid:
                action = torch.tensor([profile["command"]], dtype=torch.float, device=env.device)
                total_steps = args.warmup_steps + args.measure_steps
                for step_idx in range(total_steps):
                    _, rewards, terminated, truncated, _ = env.step(action)
                    rows.append(
                        {
                            "step": step_idx,
                            "reward": rewards[0].item(),
                            "target_vx": float(action[0, 0].item()),
                            "target_vy": float(action[0, 1].item()),
                            "target_wz": float(action[0, 2].item()),
                            "filtered_vx": float(env.slr_commands[0, 0].item()),
                            "filtered_vy": float(env.slr_commands[0, 1].item()),
                            "filtered_wz": float(env.slr_commands[0, 2].item()),
                            "actual_vx": float(env._robot.data.root_lin_vel_b[0, 0].item()),
                            "actual_vy": float(env._robot.data.root_lin_vel_b[0, 1].item()),
                            "actual_wz": float(env._robot.data.root_ang_vel_b[0, 2].item()),
                            "x": float(env._robot.data.root_pos_w[0, 0].item()),
                            "y": float(env._robot.data.root_pos_w[0, 1].item()),
                            "yaw": _yaw_from_quat(env._robot.data.root_quat_w[0].tolist()),
                            "terminated": bool(terminated[0].item()),
                            "truncated": bool(truncated[0].item()),
                        }
                    )
                    if args.step_log_interval > 0 and ((step_idx + 1) % args.step_log_interval == 0):
                        print(
                            f"[TRACK][{profile['name']}] step={step_idx + 1} "
                            f"actual_vx={rows[-1]['actual_vx']:.4f} "
                            f"actual_vy={rows[-1]['actual_vy']:.4f} "
                            f"actual_wz={rows[-1]['actual_wz']:.4f}"
                        )
                    if bool((terminated | truncated).item()):
                        done_step = step_idx
                        done_reason = "terminated" if bool(terminated[0].item()) else "truncated"
                        break
            steady_rows = rows[-args.measure_steps :] if len(rows) >= args.measure_steps else rows
            summary = {
                "profile": profile["name"],
                "command": profile["command"],
                "settle_invalid": settle_invalid,
                "samples": len(rows),
                "done_step": done_step,
                "done_reason": done_reason,
                "done_flags": {
                    "contact": bool(env.last_done_contact[0].item()),
                    "goal_hold": bool(env.last_done_goal_hold[0].item()),
                    "stand": bool(env.last_done_stand[0].item()),
                    "fall": bool(env.last_done_fall[0].item()),
                    "timeout": bool(env.last_done_timeout[0].item()),
                    "collision_occurred": bool(env.collision_occurred[0].item()),
                },
                "filtered_command_mean": {
                    "vx": _tensor_mean(steady_rows, "filtered_vx") if steady_rows else None,
                    "vy": _tensor_mean(steady_rows, "filtered_vy") if steady_rows else None,
                    "wz": _tensor_mean(steady_rows, "filtered_wz") if steady_rows else None,
                },
                "actual_mean": {
                    "vx": _tensor_mean(steady_rows, "actual_vx") if steady_rows else None,
                    "vy": _tensor_mean(steady_rows, "actual_vy") if steady_rows else None,
                    "wz": _tensor_mean(steady_rows, "actual_wz") if steady_rows else None,
                },
                "rmse_vs_target": {
                    "vx": _tensor_rmse(steady_rows, "actual_vx", "target_vx") if steady_rows else None,
                    "vy": _tensor_rmse(steady_rows, "actual_vy", "target_vy") if steady_rows else None,
                    "wz": _tensor_rmse(steady_rows, "actual_wz", "target_wz") if steady_rows else None,
                },
                "rmse_vs_filtered": {
                    "vx": _tensor_rmse(steady_rows, "actual_vx", "filtered_vx") if steady_rows else None,
                    "vy": _tensor_rmse(steady_rows, "actual_vy", "filtered_vy") if steady_rows else None,
                    "wz": _tensor_rmse(steady_rows, "actual_wz", "filtered_wz") if steady_rows else None,
                },
                "terminal_state": {
                    "projected_gravity_b_z": float(env._robot.data.projected_gravity_b[0, 2].item()),
                    "root_height": float(env._robot.data.root_pos_w[0, 2].item()),
                },
                "final_pose": rows[-1] if rows else None,
            }
            all_traces[profile["name"]] = rows
            profile_summaries.append(summary)
            print(f"[TRACK][{profile['name']}] {json.dumps(summary, ensure_ascii=True)}")

        output = {
            "seed": args.seed,
            "actuator_mode": args.actuator_mode,
            "robot_asset_source": args.robot_asset_source,
            "low_level_controller": args.low_level_controller,
            "robotlab_low_level_policy": args.robotlab_low_level_policy,
            "robotlab_command_clip": args.robotlab_command_clip,
            "settle_steps": args.settle_steps,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "episode_length_s": args.episode_length_s,
            "nav_action_scale": list(args.nav_action_scale),
            "profiles_requested": requested_profiles,
            "custom_profiles_requested": custom_profiles,
            "profiles": profile_summaries,
        }
        _write_outputs(log_dir, output, all_traces)
        print(f"[TRACK] log_dir={log_dir}")
        print(json.dumps(output, indent=2, ensure_ascii=True))
    except Exception as exc:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
