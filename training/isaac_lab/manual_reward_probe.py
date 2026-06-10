from __future__ import annotations

import argparse
import asyncio
import heapq
import json
import math
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from train import AppLauncher, DEFAULT_USD_DIR, GO2_URDF_PATH, _ensure_isaaclab_imports, _make_go2_usd


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOM_CELLS = 100
ROOM_RESOLUTION = 0.1
ROOM_SIZE = ROOM_CELLS * ROOM_RESOLUTION
PROBE_TUNING = {
    "turn_entry_x_cell": 45.0,
    "pivot_yaw_target": 1.35,
    "pre_turn_vx": 0.8,
    "turn_forward_vx": 0.35,
    "turn_yaw_rate": 1.0,
    "path_lookahead": 8,
}


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Manual reward probe for simple SEA-Nav scenes")
    parser.add_argument("--repro-mode", choices=("none", "gym_equiv"), default="none")
    parser.add_argument(
        "--scenario",
        choices=("straight", "turn", "turn_wide", "turn_open", "hard_room_eval"),
        required=True,
    )
    parser.add_argument(
        "--controller-mode",
        choices=(
            "scripted",
            "policy",
            "goal_heading",
            "safe_heading",
            "goal_vector",
            "safe_vector",
            "path_follow",
            "pivot_turn",
            "pivot_then_heading",
            "pivot_then_safe_heading",
            "pivot_then_safe_vector",
        ),
        default="scripted",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--episode-length-s", type=float, default=60.0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--eval-obstacle-level", type=int, default=9)
    parser.add_argument("--eval-room-profile", choices=("random", "visual_progressive"), default="random")
    parser.add_argument("--max-init-terrain-level", type=int, default=-1)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--stay-steps", type=int, default=-1)
    parser.add_argument("--disable-contact-termination", action="store_true", default=False)
    parser.add_argument("--num-envs", type=int, default=1)
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
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--preset-room-npy", type=str, default="")
    parser.add_argument("--case-trace", type=str, default="")
    parser.add_argument("--case-episode", type=int, default=-1)
    parser.add_argument("--fixed-start-cell", type=str, default="")
    parser.add_argument("--fixed-goal-cell", type=str, default="")
    parser.add_argument("--fixed-start-yaw", type=float, default=0.0)
    parser.add_argument("--trace-steps", action="store_true", default=False)
    parser.add_argument("--policy-stop-radius", type=float, default=-1.0)
    parser.add_argument("--policy-stop-mode", choices=("zero", "linear"), default="zero")
    parser.add_argument("--nav-action-scale", type=float, nargs=3, metavar=("VX", "VY", "WZ"), default=(1.0, 1.0, 1.0))
    parser.add_argument("--cbf-fov-deg", type=float, default=None)
    parser.add_argument("--policy-turn-yaw-threshold", type=float, default=-1.0)
    parser.add_argument("--policy-turn-forward-floor-pos", type=float, default=-1.0)
    parser.add_argument("--policy-turn-forward-floor-neg", type=float, default=-1.0)
    parser.add_argument("--policy-path-blend-weight", type=float, default=-1.0)
    parser.add_argument("--policy-path-blend-min-distance", type=float, default=1.0)
    parser.add_argument("--record-topdown-video", type=str, default="")
    parser.add_argument("--record-frame-dir", type=str, default="")
    parser.add_argument("--record-video-fps", type=float, default=24.0)
    parser.add_argument("--record-video-width", type=int, default=1024)
    parser.add_argument("--record-video-height", type=int, default=1024)
    parser.add_argument("--record-every-n-steps", type=int, default=4)
    parser.add_argument("--record-max-frames", type=int, default=240)
    parser.add_argument("--record-camera-height", type=float, default=16.0)
    parser.add_argument("--show-topdown-camera", action="store_true", default=False)
    parser.add_argument("--show-start-goal-markers", action="store_true", default=False)
    parser.add_argument("--viewer-camera-eye", type=float, nargs=3, metavar=("X", "Y", "Z"), default=())
    parser.add_argument("--viewer-camera-target", type=float, nargs=3, metavar=("X", "Y", "Z"), default=())
    parser.add_argument("--record-start-delay-s", type=float, default=0.0)
    parser.add_argument("--step-sleep-s", type=float, default=0.0)
    parser.add_argument("--turn-entry-x-cell", type=float, default=PROBE_TUNING["turn_entry_x_cell"])
    parser.add_argument("--pivot-yaw-target", type=float, default=PROBE_TUNING["pivot_yaw_target"])
    parser.add_argument("--pre-turn-vx", type=float, default=PROBE_TUNING["pre_turn_vx"])
    parser.add_argument("--turn-forward-vx", type=float, default=PROBE_TUNING["turn_forward_vx"])
    parser.add_argument("--turn-yaw-rate", type=float, default=PROBE_TUNING["turn_yaw_rate"])
    parser.add_argument("--path-lookahead", type=int, default=PROBE_TUNING["path_lookahead"])
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _parse_args(argv=None):
    parser = build_arg_parser()
    return parser.parse_args(argv)


def _apply_repro_mode(args):
    if getattr(args, "repro_mode", "none") == "none":
        if args.cbf_fov_deg is None:
            args.cbf_fov_deg = 240.0
        return
    if args.repro_mode != "gym_equiv":
        raise ValueError(f"Unsupported repro mode: {args.repro_mode}")
    args.robot_asset_source = "converted_urdf"
    args.actuator_mode = "gym_torque"
    args.low_level_controller = "sea_nav_jit"
    if args.cbf_fov_deg is None:
        args.cbf_fov_deg = 180.0


def _configure_probe_tuning(args):
    PROBE_TUNING.update(
        {
            "turn_entry_x_cell": args.turn_entry_x_cell,
            "pivot_yaw_target": args.pivot_yaw_target,
            "pre_turn_vx": args.pre_turn_vx,
            "turn_forward_vx": args.turn_forward_vx,
            "turn_yaw_rate": args.turn_yaw_rate,
            "path_lookahead": args.path_lookahead,
        }
    )


def _preload_probe_runtime_dependencies():
    # GUI Isaac Sim adds its pip_prebundle to import paths during startup. Preload
    # the NumPy/SciPy stack first so trimesh does not mix venv NumPy with Kit NumPy.
    import numpy.random  # noqa: F401
    import scipy.spatial  # noqa: F401
    import trimesh  # noqa: F401


def _run_with_sim_app(args):
    import numpy as np
    import torch

    from sea_nav_env import SeaNavIsaacLabEnv, make_sea_nav_env_cfg

    spec = _scenario_spec(args.scenario, np)
    if args.preset_room_npy:
        spec["room"] = np.load(args.preset_room_npy).astype(float)
    elif args.scenario == "hard_room_eval" and args.eval_room_profile == "visual_progressive":
        spec["room"] = _make_visual_progressive_room(np, args.eval_obstacle_level)
    fixed_start_cell = _parse_cell_arg(args.fixed_start_cell)
    fixed_goal_cell = _parse_cell_arg(args.fixed_goal_cell)
    fixed_start_yaw = args.fixed_start_yaw
    if args.case_trace:
        if fixed_start_cell is not None or fixed_goal_cell is not None:
            raise ValueError("--case-trace cannot be combined with --fixed-start-cell/--fixed-goal-cell")
        case = _load_case_from_trace(args.case_trace, args.case_episode)
        fixed_start_cell = case["start_cell"]
        fixed_goal_cell = case["goal_cell"]
        fixed_start_yaw = float(case["start_yaw"])
        if not args.preset_room_npy:
            room_file = Path(case["room_npy"])
            if room_file.is_file():
                spec["room"] = np.load(room_file).astype(float)
            else:
                raise FileNotFoundError(
                    f"case trace did not have a sibling room.npy: {room_file}. "
                    "Pass --preset-room-npy explicitly to override."
                )
    if args.scenario == "hard_room_eval" and args.controller_mode != "policy":
        if fixed_start_cell is None or fixed_goal_cell is None:
            raise ValueError("scripted hard_room_eval requires --fixed-start-cell and --fixed-goal-cell")
    if args.controller_mode == "scripted":
        command_fn = spec["command"]
    elif args.controller_mode == "policy":
        command_fn = None
    elif args.controller_mode == "goal_heading":
        command_fn = _goal_heading_command
    elif args.controller_mode == "safe_heading":
        command_fn = _safe_heading_command
    elif args.controller_mode == "goal_vector":
        command_fn = _goal_vector_command
    elif args.controller_mode == "safe_vector":
        command_fn = _safe_vector_command
    elif args.controller_mode == "path_follow":
        command_fn = _path_follow_command
    elif args.controller_mode == "pivot_turn":
        command_fn = _pivot_turn_command
    elif args.controller_mode == "pivot_then_safe_heading":
        command_fn = _pivot_then_safe_heading_command
    elif args.controller_mode == "pivot_then_safe_vector":
        command_fn = _pivot_then_safe_vector_command
    else:
        command_fn = _pivot_then_heading_command
    timestamp = datetime.now().strftime("%m_%d_%H-%M-%S")
    log_dir = Path(args.log_root) / f"manual_reward_probe_{args.scenario}" / timestamp

    env = None
    try:
        go2_usd_path = _make_go2_usd(args) if args.robot_asset_source == "converted_urdf" else None
        env_cfg = make_sea_nav_env_cfg(
            go2_usd_path=go2_usd_path,
            num_envs=args.num_envs,
            seed=args.seed,
            actuator_mode=args.actuator_mode,
            robot_asset_source=args.robot_asset_source,
            low_level_controller=args.low_level_controller,
            robotlab_policy_path=args.robotlab_low_level_policy,
            robotlab_command_clip=args.robotlab_command_clip,
            episode_length_s=args.episode_length_s,
            nav_action_scale=tuple(args.nav_action_scale),
            terrain_rows=1,
            terrain_cols=1,
            obstacle_level=args.eval_obstacle_level if args.scenario == "hard_room_eval" else 0,
            terrain_difficulty_range=(1.0, 1.0) if args.scenario == "hard_room_eval" else None,
            preset_room=spec["room"],
            randomize_friction=False,
            randomize_base_mass=True,
            added_mass_range=(0.0, 0.0),
        )
        env_cfg.add_noise = False
        env_cfg.enable_collision_replay = False
        if args.disable_contact_termination:
            env_cfg.enable_contact_termination = False
            env_cfg.termination_body_patterns = ()
        if args.stay_steps >= 0:
            env_cfg.stay_steps = args.stay_steps
        if args.max_init_terrain_level >= 0:
            env_cfg.terrain.max_init_terrain_level = args.max_init_terrain_level
        env_cfg.sim.device = args.sim_device
        env = SeaNavIsaacLabEnv(cfg=env_cfg, render_mode=None)
        env.reset()
        policy_fn = None
        if args.controller_mode == "policy":
            if not args.checkpoint:
                raise ValueError("--checkpoint is required when controller-mode=policy")
            policy_fn = _load_inference_policy(env, args.checkpoint, args.cbf_fov_deg)

        if args.scenario == "hard_room_eval":
            if args.case_trace:
                args.fixed_start_cell = f"{fixed_start_cell[0]},{fixed_start_cell[1]}"
                args.fixed_goal_cell = f"{fixed_goal_cell[0]},{fixed_goal_cell[1]}"
                args.fixed_start_yaw = fixed_start_yaw
            if _can_vectorize_hard_room_eval(args, policy_fn, command_fn):
                summary, trace_rows, room_map = _run_hard_room_eval_vectorized(env, args, policy_fn)
            else:
                summary, trace_rows, room_map = _run_hard_room_eval(env, args, policy_fn, command_fn)
            _write_summary(log_dir, summary, trace_rows, room_map=room_map)
            print(f"[PROBE] scenario={args.scenario} log_dir={log_dir}")
            print(json.dumps(summary, indent=2, ensure_ascii=True))
            return {"log_dir": str(log_dir), "summary": summary, "trace_rows": trace_rows}

        env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
        start_xy = torch.tensor([_grid_to_local_xy(spec["start_cell"])], device=env.device)
        goal_xy = torch.tensor([_grid_to_local_xy(spec["goal_cell"])], device=env.device)
        if args.controller_mode == "path_follow":
            path_cells = _astar_path(spec["room"], spec["start_cell"], spec["goal_cell"])
            path_points_local = torch.tensor(
                [[_grid_to_local_xy(cell)[0], _grid_to_local_xy(cell)[1]] for cell in path_cells],
                dtype=torch.float,
                device=env.device,
            )
            env._probe_path_points_local = path_points_local
            env._probe_path_lookahead = args.path_lookahead
        env.set_manual_start_and_goal(env_ids, start_xy, goal_xy, yaw=0.0, root_height=0.42)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(dt=0.0)
        env._get_observations()

        zero_action = torch.zeros((1, env.num_nav_actions), device=env.device)
        settle_invalid = False
        for _ in range(args.settle_steps):
            _, _, terminated, truncated, _ = env.step(zero_action)
            if bool((terminated | truncated).item()):
                settle_invalid = True
                break

        trace_rows = []
        total_reward = 0.0
        reach_reward_sum = 0.0
        min_distance = float("inf")
        first_reach_step = None
        done_step = None
        done_reason = None
        episode_summary = None

        if not settle_invalid:
            policy_obs = env._get_observations()["policy"] if policy_fn is not None else None
            for step_idx in range(args.max_steps):
                if policy_fn is not None:
                    with torch.inference_mode():
                        command = policy_fn(policy_obs)
                    command = _apply_policy_turn_prior(command, args)
                else:
                    command = torch.tensor([command_fn(step_idx, env)], dtype=torch.float, device=env.device)
                obs_dict, rewards, terminated, truncated, extras = env.step(command)
                if policy_fn is not None:
                    policy_obs = obs_dict["policy"]

                reward_value = rewards[0].item()
                reach_reward_value = env.last_reward_terms["reach_pos_target_tight"][0].item()
                distance_value = env.last_distance[0].item()
                yaw_value = _yaw_from_quat(env.last_root_quat_w[0].tolist())
                front_clearance = env.last_rays[0].min().item()
                local_x = (env.last_root_pos_w[0, 0] - env._terrain.env_origins[0, 0]).item()
                local_y = (env.last_root_pos_w[0, 1] - env._terrain.env_origins[0, 1]).item()
                total_reward += reward_value
                reach_reward_sum += reach_reward_value
                min_distance = min(min_distance, distance_value)
                if first_reach_step is None and reach_reward_value > 0.0:
                    first_reach_step = step_idx

                trace_rows.append(
                    {
                        "step": step_idx,
                        "command": [float(x) for x in command[0].tolist()],
                        "nav_action_scaled": [float(x) for x in env.nav_actions_orig[0].tolist()],
                        "low_level_command": [float(x) for x in env.slr_commands[0].tolist()],
                        "root_lin_vel_b": [float(x) for x in env._robot.data.root_lin_vel_b[0].tolist()],
                        "root_ang_vel_b": [float(x) for x in env._robot.data.root_ang_vel_b[0].tolist()],
                        "stay_timer": int(env.stay_timer[0].item()),
                        "goal_hold_timer": int(env.goal_hold_timer[0].item()),
                        "reward": reward_value,
                        "reach_reward": reach_reward_value,
                        "distance": distance_value,
                        "yaw": yaw_value,
                        "x": local_x,
                        "y": local_y,
                        "goal_local_x": env.last_goal_local_pos[0, 0].item(),
                        "goal_local_y": env.last_goal_local_pos[0, 1].item(),
                        "front_clearance": front_clearance,
                        "done_contact": bool(env.last_done_contact[0].item()),
                        "done_goal_hold": bool(env.last_done_goal_hold[0].item()),
                        "done_stand": bool(env.last_done_stand[0].item()),
                        "done_fall": bool(env.last_done_fall[0].item()),
                        "done_timeout": bool(env.last_done_timeout[0].item()),
                        "terminated": bool(terminated[0].item()),
                        "truncated": bool(truncated[0].item()),
                    }
                )

                if "episode" in extras and extras["episode"]:
                    episode_summary = {key: _tensor_scalar(value) for key, value in extras["episode"].items()}

                if bool((terminated | truncated).item()):
                    done_step = step_idx
                    done_reason = "terminated" if bool(terminated[0].item()) else "truncated"
                    break
        else:
            done_reason = "invalid_settle"

        summary = {
            "scenario": args.scenario,
            "repro_mode": getattr(args, "repro_mode", "none"),
            "controller_mode": args.controller_mode,
            "actuator_mode": args.actuator_mode,
            "robot_asset_source": args.robot_asset_source,
            "low_level_controller": args.low_level_controller,
            "robotlab_low_level_policy": args.robotlab_low_level_policy,
            "robotlab_command_clip": args.robotlab_command_clip,
            "cbf_fov_deg": args.cbf_fov_deg,
            "checkpoint": args.checkpoint if args.checkpoint else None,
            "stay_steps": args.stay_steps,
            "disable_contact_termination": args.disable_contact_termination,
            "room_cells": ROOM_CELLS,
            "room_resolution": ROOM_RESOLUTION,
            "start_cell": list(spec["start_cell"]),
            "goal_cell": list(spec["goal_cell"]),
            "settle_steps": args.settle_steps,
            "episode_length_s": args.episode_length_s,
            "nav_action_scale": list(args.nav_action_scale),
            "turn_entry_x_cell": args.turn_entry_x_cell,
            "pivot_yaw_target": args.pivot_yaw_target,
            "pre_turn_vx": args.pre_turn_vx,
            "turn_forward_vx": args.turn_forward_vx,
            "turn_yaw_rate": args.turn_yaw_rate,
            "path_lookahead": args.path_lookahead,
            "policy_stop_radius": args.policy_stop_radius if args.policy_stop_radius >= 0.0 else None,
            "policy_stop_mode": args.policy_stop_mode if args.policy_stop_radius >= 0.0 else None,
            "policy_turn_yaw_threshold": args.policy_turn_yaw_threshold if args.policy_turn_yaw_threshold >= 0.0 else None,
            "policy_turn_forward_floor_pos": args.policy_turn_forward_floor_pos if args.policy_turn_forward_floor_pos >= 0.0 else None,
            "policy_turn_forward_floor_neg": args.policy_turn_forward_floor_neg if args.policy_turn_forward_floor_neg >= 0.0 else None,
            "policy_path_blend_weight": args.policy_path_blend_weight if args.policy_path_blend_weight >= 0.0 else None,
            "policy_path_blend_min_distance": args.policy_path_blend_min_distance if args.policy_path_blend_weight >= 0.0 else None,
            "settle_invalid": settle_invalid,
            "max_steps": args.max_steps,
            "total_reward_sum": total_reward,
            "reach_reward_sum": reach_reward_sum,
            "min_distance": None if min_distance == float("inf") else min_distance,
            "first_reach_step": first_reach_step,
            "done_step": done_step,
            "done_reason": done_reason,
            "done_flags": None
            if env is None
            else {
                "contact": bool(env.last_done_contact[0].item()),
                "goal_hold": bool(env.last_done_goal_hold[0].item()),
                "stand": bool(env.last_done_stand[0].item()),
                "fall": bool(env.last_done_fall[0].item()),
                "timeout": bool(env.last_done_timeout[0].item()),
            },
            "episode_summary": episode_summary,
        }
        if args.preset_room_npy:
            summary["room_path"] = args.preset_room_npy
        elif args.case_trace:
            summary["room_path"] = str(Path(args.case_trace).with_name("room.npy"))
        else:
            room_path = log_dir / "room.npy"
            summary["room_path"] = str(room_path)

        _write_summary(log_dir, summary, trace_rows, room_map=spec["room"])
        print(f"[PROBE] scenario={args.scenario} log_dir={log_dir}")
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return {"log_dir": str(log_dir), "summary": summary, "trace_rows": trace_rows}
    finally:
        if env is not None:
            env.close()


def _parse_cell_arg(value: str) -> tuple[int, int] | None:
    value = value.strip()
    if not value:
        return None
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError(f"Cell argument must be 'x,y', got: {value}")
    return int(parts[0]), int(parts[1])


def _load_case_from_trace(trace_path: str, episode_idx: int) -> dict[str, object]:
    trace_file = Path(trace_path)
    if not trace_file.is_file():
        raise FileNotFoundError(f"case trace does not exist: {trace_file}")
    if episode_idx < 0:
        raise ValueError("--case-episode must be >= 0 when --case-trace is provided")
    with trace_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("episode", -1)) != episode_idx:
                continue
            start_cell = row.get("start_cell")
            goal_cell = row.get("goal_cell")
            start_yaw = row.get("start_yaw")
            if start_cell is None or goal_cell is None or start_yaw is None:
                raise ValueError(f"trace row for episode {episode_idx} is missing start/goal/yaw")
            return {
                "episode": episode_idx,
                "start_cell": (int(start_cell[0]), int(start_cell[1])),
                "goal_cell": (int(goal_cell[0]), int(goal_cell[1])),
                "start_yaw": float(start_yaw),
                "room_npy": str(trace_file.with_name("room.npy")),
            }
    raise ValueError(f"episode {episode_idx} not found in case trace {trace_file}")


def _make_empty_room(np):
    room = np.zeros((ROOM_CELLS, ROOM_CELLS), dtype=float)
    room[0, :] = 1.0
    room[-1, :] = 1.0
    room[:, 0] = 1.0
    room[:, -1] = 1.0
    return room


def _make_straight_room(np):
    room = _make_empty_room(np)
    room[10:91, 40] = 1.0
    room[10:91, 60] = 1.0
    return room


def _make_turn_room(np):
    room = _make_empty_room(np)
    room[10:56, 30] = 1.0
    room[10:56, 40] = 1.0
    room[50, 35:71] = 1.0
    room[60, 35:71] = 1.0
    return room


def _make_turn_wide_room(np):
    room = _make_empty_room(np)
    room[10:56, 25] = 1.0
    room[10:56, 45] = 1.0
    room[45, 35:76] = 1.0
    room[65, 35:76] = 1.0
    return room


def _make_turn_open_room(np):
    room = _make_empty_room(np)
    room[10:46, 30] = 1.0
    room[10:46, 40] = 1.0
    room[60, 35:86] = 1.0
    room[70, 35:86] = 1.0
    return room


def _make_visual_progressive_room(np, level: int):
    room = _make_empty_room(np)
    presets = [
        (12, 15, 7, 9),
        (67, 12, 7, 7),
        (15, 63, 7, 7),
        (68, 67, 7, 8),
        (48, 42, 9, 6),
        (58, 45, 8, 6),
        (30, 77, 7, 7),
        (78, 32, 7, 7),
        (20, 28, 6, 8),
        (82, 80, 6, 7),
        (38, 18, 8, 6),
        (15, 83, 7, 7),
        (72, 52, 8, 7),
        (43, 70, 6, 8),
        (57, 73, 6, 8),
        (30, 48, 6, 6),
        (84, 17, 6, 6),
        (18, 43, 5, 7),
        (48, 17, 5, 8),
        (63, 22, 6, 6),
        (72, 74, 6, 6),
        (37, 58, 5, 6),
        (86, 57, 5, 6),
        (23, 70, 5, 6),
        (53, 30, 5, 5),
        (66, 36, 5, 5),
        (34, 34, 5, 5),
        (25, 56, 5, 5),
        (76, 43, 5, 5),
        (58, 61, 5, 5),
    ]
    if level <= 3:
        count = 8
    elif level <= 6:
        count = 17
    else:
        count = 30
    for idx, (row, col, height, width) in enumerate(presets[:count]):
        value = 0.55 + 0.015 * min(idx, 20)
        room[row : row + height, col : col + width] = np.maximum(
            room[row : row + height, col : col + width],
            min(value, 0.95),
        )
    if level > 6:
        # Add two partial walls with deliberate gaps. These make hard visibly denser
        # without sealing the room into disconnected regions.
        room[41:45, 12:42] = np.maximum(room[41:45, 12:42], 0.75)
        room[41:45, 52:88] = np.maximum(room[41:45, 52:88], 0.75)
        room[61:65, 18:52] = np.maximum(room[61:65, 18:52], 0.8)
        room[61:65, 62:86] = np.maximum(room[61:65, 62:86], 0.8)
    return room


def _grid_to_local_xy(cell_xy: tuple[int, int]):
    return (
        cell_xy[0] * ROOM_RESOLUTION - ROOM_SIZE * 0.5,
        cell_xy[1] * ROOM_RESOLUTION - ROOM_SIZE * 0.5,
    )


def _local_xy_to_cell(local_xy: tuple[float, float]):
    gx = int(round((local_xy[0] + ROOM_SIZE * 0.5) / ROOM_RESOLUTION))
    gy = int(round((local_xy[1] + ROOM_SIZE * 0.5) / ROOM_RESOLUTION))
    gx = max(0, min(ROOM_CELLS - 1, gx))
    gy = max(0, min(ROOM_CELLS - 1, gy))
    return gx, gy


def _astar_path(room, start_cell: tuple[int, int], goal_cell: tuple[int, int]):
    rows, cols = room.shape

    def neighbors(cell):
        x, y = cell
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = x + dx
            ny = y + dy
            if 0 <= nx < rows and 0 <= ny < cols and room[nx, ny] <= 0.1:
                yield (nx, ny)

    def heuristic(cell):
        return abs(cell[0] - goal_cell[0]) + abs(cell[1] - goal_cell[1])

    frontier = [(heuristic(start_cell), 0, start_cell)]
    came_from = {start_cell: None}
    cost_so_far = {start_cell: 0}
    while frontier:
        _, cost, current = heapq.heappop(frontier)
        if current == goal_cell:
            break
        if cost != cost_so_far[current]:
            continue
        for nxt in neighbors(current):
            new_cost = cost + 1
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + heuristic(nxt)
                heapq.heappush(frontier, (priority, new_cost, nxt))
                came_from[nxt] = current
    if goal_cell not in came_from:
        raise RuntimeError(f"No free-cell path found from {start_cell} to {goal_cell}")
    path = []
    cursor = goal_cell
    while cursor is not None:
        path.append(cursor)
        cursor = came_from[cursor]
    path.reverse()
    return path


def _scenario_spec(scenario: str, np):
    if scenario == "hard_room_eval":
        return {
            "room": None,
            "start_cell": None,
            "goal_cell": None,
            "command": None,
        }
    if scenario == "straight":
        return {
            "room": _make_straight_room(np),
            "start_cell": (20, 50),
            "goal_cell": (80, 50),
            "command": _straight_command,
        }
    if scenario == "turn_wide":
        return {
            "room": _make_turn_wide_room(np),
            "start_cell": (18, 35),
            "goal_cell": (60, 70),
            "command": _turn_command,
        }
    if scenario == "turn_open":
        return {
            "room": _make_turn_open_room(np),
            "start_cell": (18, 35),
            "goal_cell": (75, 65),
            "command": _turn_command,
        }
    return {
        "room": _make_turn_room(np),
        "start_cell": (18, 35),
        "goal_cell": (55, 65),
        "command": _turn_command,
    }


def _yaw_from_quat(quat):
    return math.atan2(
        2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
        1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3]),
    )


def _straight_command(step_idx, env):
    return [1.0, 0.0, 0.0]


def _turn_command(step_idx, env):
    local_x = (env._robot.data.root_pos_w[0, 0] - env._terrain.env_origins[0, 0]).item()
    yaw = _yaw_from_quat(env._robot.data.root_quat_w[0].tolist())
    turn_entry_x = _grid_to_local_xy((PROBE_TUNING["turn_entry_x_cell"], 35))[0]
    if local_x < turn_entry_x:
        return [1.0, 0.0, 0.0]
    if yaw < PROBE_TUNING["pivot_yaw_target"]:
        return [PROBE_TUNING["turn_forward_vx"], 0.0, PROBE_TUNING["turn_yaw_rate"]]
    return [1.0, 0.0, 0.0]


def _goal_heading_command(step_idx, env):
    del step_idx
    goal_x = env.goal_local_pos[0, 0].item()
    goal_y = env.goal_local_pos[0, 1].item()
    heading_error = math.atan2(goal_y, max(goal_x, 1.0e-4))
    front_clearance = env.rays[0].min().item()
    distance = env.distance[0].item()

    yaw_cmd = max(-1.0, min(1.0, 1.8 * heading_error))
    forward_cmd = max(0.0, math.cos(heading_error))
    if abs(heading_error) > 0.7:
        forward_cmd *= 0.25
    elif abs(heading_error) > 0.35:
        forward_cmd *= 0.5
    if front_clearance < 0.8:
        forward_cmd *= max(0.1, min(1.0, front_clearance / 0.8))
    if distance < 1.0:
        forward_cmd *= max(0.15, min(1.0, distance / 1.0))
    forward_cmd = min(0.8, 0.8 * forward_cmd)
    return [forward_cmd, 0.0, yaw_cmd]


def _safe_heading_command(step_idx, env):
    del step_idx
    goal_x = env.goal_local_pos[0, 0].item()
    goal_y = env.goal_local_pos[0, 1].item()
    heading_error = math.atan2(goal_y, max(goal_x, 1.0e-4))
    front_clearance = env.rays[0].min().item()
    distance = env.distance[0].item()

    yaw_cmd = max(-1.0, min(1.0, 1.8 * heading_error))
    if front_clearance < 0.35 and abs(heading_error) > 0.12:
        return [0.0, 0.0, yaw_cmd]

    forward_cmd = max(0.0, math.cos(heading_error))
    if abs(heading_error) > 0.7:
        forward_cmd *= 0.15
    elif abs(heading_error) > 0.35:
        forward_cmd *= 0.4
    if front_clearance < 0.7:
        forward_cmd *= max(0.0, min(1.0, (front_clearance - 0.2) / 0.5))
    if distance < 1.2:
        forward_cmd *= max(0.1, min(1.0, distance / 1.2))
    forward_cmd = min(0.7, 0.7 * forward_cmd)
    return [forward_cmd, 0.0, yaw_cmd]


def _goal_vector_command(step_idx, env):
    del step_idx
    goal_x = env.goal_local_pos[0, 0].item()
    goal_y = env.goal_local_pos[0, 1].item()
    distance = env.distance[0].item()
    heading_error = math.atan2(goal_y, max(goal_x, 1.0e-4))
    front_clearance = env.rays[0].min().item()

    vx_cmd = max(-0.1, min(0.8, 0.45 * goal_x))
    vy_cmd = max(-0.45, min(0.45, 0.55 * goal_y))
    yaw_cmd = max(-1.0, min(1.0, 1.2 * heading_error))

    if abs(heading_error) > 0.9:
        vx_cmd *= 0.15
        vy_cmd *= 0.35
    elif abs(heading_error) > 0.5:
        vx_cmd *= 0.35
        vy_cmd *= 0.6

    if front_clearance < 0.8:
        scale = max(0.15, min(1.0, front_clearance / 0.8))
        vx_cmd *= scale
        vy_cmd *= scale

    if distance < 1.2:
        approach_scale = max(0.2, min(1.0, distance / 1.2))
        vx_cmd *= approach_scale
        vy_cmd *= approach_scale

    return [vx_cmd, vy_cmd, yaw_cmd]


def _safe_vector_command(step_idx, env):
    del step_idx
    goal_x = env.goal_local_pos[0, 0].item()
    goal_y = env.goal_local_pos[0, 1].item()
    distance = env.distance[0].item()
    heading_error = math.atan2(goal_y, max(goal_x, 1.0e-4))
    front_clearance = env.rays[0].min().item()

    yaw_cmd = max(-1.0, min(1.0, 1.4 * heading_error))
    if front_clearance < 0.35 and abs(heading_error) > 0.12:
        return [0.0, 0.0, yaw_cmd]

    vx_cmd = max(0.0, min(0.6, 0.22 * goal_x))
    vy_cmd = max(-0.22, min(0.22, 0.18 * goal_y))
    if abs(heading_error) > 0.75:
        vx_cmd *= 0.15
        vy_cmd *= 0.45
    elif abs(heading_error) > 0.4:
        vx_cmd *= 0.35
        vy_cmd *= 0.7
    if front_clearance < 0.7:
        vx_cmd *= max(0.0, min(1.0, (front_clearance - 0.2) / 0.5))
    if distance < 1.2:
        approach_scale = max(0.15, min(1.0, distance / 1.2))
        vx_cmd *= approach_scale
        vy_cmd *= approach_scale
    return [vx_cmd, vy_cmd, yaw_cmd]


def _path_follow_command(step_idx, env):
    del step_idx
    import torch

    path_points = getattr(env, "_probe_path_points_local", None)
    if path_points is None or path_points.shape[0] == 0:
        raise RuntimeError("Path-follow controller requires env._probe_path_points_local")

    robot_world_xy = env._robot.data.root_pos_w[0, :2]
    env_origin_xy = env._terrain.env_origins[0, :2]
    robot_local_xy = robot_world_xy - env_origin_xy
    distances = torch.norm(path_points - robot_local_xy.unsqueeze(0), dim=-1)
    nearest_idx = int(torch.argmin(distances).item())
    lookahead = int(getattr(env, "_probe_path_lookahead", PROBE_TUNING["path_lookahead"]))
    target_idx = min(nearest_idx + lookahead, path_points.shape[0] - 1)
    target_local_xy = path_points[target_idx]

    yaw = _yaw_from_quat(env._robot.data.root_quat_w[0].tolist())
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    delta_local_world = target_local_xy - robot_local_xy
    goal_x = cos_yaw * delta_local_world[0].item() + sin_yaw * delta_local_world[1].item()
    goal_y = -sin_yaw * delta_local_world[0].item() + cos_yaw * delta_local_world[1].item()

    heading_error = math.atan2(goal_y, max(goal_x, 1.0e-4))
    front_clearance = env.rays[0].min().item()
    distance = float(torch.norm(delta_local_world).item())
    yaw_cmd = max(-1.0, min(1.0, 1.6 * heading_error))
    if front_clearance < 0.35 and abs(heading_error) > 0.12:
        return [0.0, 0.0, yaw_cmd]

    vx_cmd = max(0.0, min(0.55, 0.28 * goal_x))
    vy_cmd = max(-0.18, min(0.18, 0.16 * goal_y))
    if abs(heading_error) > 0.7:
        vx_cmd *= 0.12
        vy_cmd *= 0.45
    elif abs(heading_error) > 0.4:
        vx_cmd *= 0.3
        vy_cmd *= 0.7
    if front_clearance < 0.7:
        vx_cmd *= max(0.0, min(1.0, (front_clearance - 0.2) / 0.5))
    if distance < 0.8:
        scale = max(0.12, min(1.0, distance / 0.8))
        vx_cmd *= scale
        vy_cmd *= scale
    return [vx_cmd, vy_cmd, yaw_cmd]


def _pivot_turn_command(step_idx, env):
    del step_idx
    local_x = (env._robot.data.root_pos_w[0, 0] - env._terrain.env_origins[0, 0]).item()
    yaw = _yaw_from_quat(env._robot.data.root_quat_w[0].tolist())
    distance = env.distance[0].item()
    turn_entry_x = _grid_to_local_xy((PROBE_TUNING["turn_entry_x_cell"], 35))[0]
    if local_x < turn_entry_x:
        return [PROBE_TUNING["pre_turn_vx"], 0.0, 0.0]
    if yaw < PROBE_TUNING["pivot_yaw_target"]:
        return [0.0, 0.0, PROBE_TUNING["turn_yaw_rate"]]
    forward_cmd = PROBE_TUNING["pre_turn_vx"] if distance > 1.0 else max(0.15, min(PROBE_TUNING["pre_turn_vx"], distance))
    return [forward_cmd, 0.0, 0.0]


def _pivot_then_heading_command(step_idx, env):
    del step_idx
    local_x = (env._robot.data.root_pos_w[0, 0] - env._terrain.env_origins[0, 0]).item()
    yaw = _yaw_from_quat(env._robot.data.root_quat_w[0].tolist())
    turn_entry_x = _grid_to_local_xy((PROBE_TUNING["turn_entry_x_cell"], 35))[0]
    if local_x < turn_entry_x:
        return [PROBE_TUNING["pre_turn_vx"], 0.0, 0.0]
    if yaw < PROBE_TUNING["pivot_yaw_target"]:
        return [0.0, 0.0, PROBE_TUNING["turn_yaw_rate"]]
    return _goal_heading_command(0, env)


def _pivot_then_safe_heading_command(step_idx, env):
    del step_idx
    local_x = (env._robot.data.root_pos_w[0, 0] - env._terrain.env_origins[0, 0]).item()
    yaw = _yaw_from_quat(env._robot.data.root_quat_w[0].tolist())
    turn_entry_x = _grid_to_local_xy((PROBE_TUNING["turn_entry_x_cell"], 35))[0]
    if local_x < turn_entry_x:
        return [PROBE_TUNING["pre_turn_vx"], 0.0, 0.0]
    if yaw < PROBE_TUNING["pivot_yaw_target"]:
        return [0.0, 0.0, PROBE_TUNING["turn_yaw_rate"]]
    return _safe_heading_command(0, env)


def _pivot_then_safe_vector_command(step_idx, env):
    del step_idx
    local_x = (env._robot.data.root_pos_w[0, 0] - env._terrain.env_origins[0, 0]).item()
    yaw = _yaw_from_quat(env._robot.data.root_quat_w[0].tolist())
    turn_entry_x = _grid_to_local_xy((PROBE_TUNING["turn_entry_x_cell"], 35))[0]
    if local_x < turn_entry_x:
        return [PROBE_TUNING["pre_turn_vx"], 0.0, 0.0]
    if yaw < PROBE_TUNING["pivot_yaw_target"]:
        return [0.0, 0.0, PROBE_TUNING["turn_yaw_rate"]]
    return _safe_vector_command(0, env)


def _tensor_scalar(value):
    if hasattr(value, "ndim"):
        return float(value.float().mean().item() if value.ndim > 0 else value.float().item())
    return float(value)


def _load_inference_policy(env, checkpoint_path: str, cbf_fov_deg: float):
    import torch

    rsl_rl_root = REPO_ROOT / "training" / "rsl_rl"
    if str(rsl_rl_root) not in sys.path:
        sys.path.insert(0, str(rsl_rl_root))

    from rsl_rl.modules.cbf_actor_critic import DifferentiableSafeActorCritic

    actor_critic = DifferentiableSafeActorCritic(
        num_actions=env.num_nav_actions,
        num_props=env.num_props,
        his_len=env.history_length,
        num_rays=env.rays.shape[1],
        init_noise_std=1.5,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        cbf_fov_deg=cbf_fov_deg,
    ).to(env.device)
    loaded_dict = torch.load(checkpoint_path, map_location=env.device)
    actor_critic.load_state_dict(loaded_dict["model_state_dict"])
    actor_critic.eval()
    print(f"[PLAY] loaded_policy={checkpoint_path} cbf_fov_deg={cbf_fov_deg}")
    return actor_critic.act_inference


def _setup_viewport_camera(env, eye, target, label: str):
    import omni.kit.app

    env.sim.set_camera_view(
        eye=tuple(float(v) for v in eye),
        target=tuple(float(v) for v in target),
    )
    for _ in range(4):
        omni.kit.app.get_app().update()
    print(
        f"[RECORD] {label} "
        f"eye=({float(eye[0]):.3f},{float(eye[1]):.3f},{float(eye[2]):.3f}) "
        f"target=({float(target[0]):.3f},{float(target[1]):.3f},{float(target[2]):.3f})"
    )


def _setup_topdown_viewport_camera(env, start_local_xy, goal_local_xy, camera_height: float):
    origin = env._terrain.env_origins[0]
    origin_x = float(origin[0].item())
    origin_y = float(origin[1].item())
    robot_x = float(env._robot.data.root_pos_w[0, 0].item())
    robot_y = float(env._robot.data.root_pos_w[0, 1].item())
    goal_x = origin_x + float(goal_local_xy[0])
    goal_y = origin_y + float(goal_local_xy[1])

    # A perfectly vertical free-camera can produce an unstable up-vector in Kit.
    # Keep a tiny Y offset while staying visually top-down.
    _setup_viewport_camera(
        env,
        eye=(origin_x, origin_y - 0.01, camera_height),
        target=(origin_x, origin_y, 0.0),
        label="topdown_camera_ready",
    )
    print(
        "[RECORD] topdown_camera_context "
        f"robot=({robot_x:.3f},{robot_y:.3f}) "
        f"goal=({goal_x:.3f},{goal_y:.3f}) "
        f"start_local=({float(start_local_xy[0]):.3f},{float(start_local_xy[1]):.3f}) "
        f"goal_local=({float(goal_local_xy[0]):.3f},{float(goal_local_xy[1]):.3f})"
    )


def _add_start_goal_markers(env, start_local_xy, goal_local_xy):
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    stage = omni.usd.get_context().get_stage()
    origin = env._terrain.env_origins[0]
    origin_x = float(origin[0].item())
    origin_y = float(origin[1].item())
    start_xy = (origin_x + float(start_local_xy[0]), origin_y + float(start_local_xy[1]))
    goal_xy = (origin_x + float(goal_local_xy[0]), origin_y + float(goal_local_xy[1]))
    UsdGeom.Xform.Define(stage, Sdf.Path("/World/RecordMarkers"))

    def material(path: str, color: tuple[float, float, float]):
        mat = UsdShade.Material.Define(stage, Sdf.Path(path))
        shader = UsdShade.Shader.Define(stage, Sdf.Path(path + "/PreviewSurface"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat

    start_mat = material("/World/RecordMarkers/StartMaterial", (0.05, 0.8, 0.15))
    goal_mat = material("/World/RecordMarkers/GoalMaterial", (0.95, 0.1, 0.05))

    def cylinder(path: str, xy: tuple[float, float], mat, radius: float, height: float):
        prim = UsdGeom.Cylinder.Define(stage, Sdf.Path(path))
        prim.CreateRadiusAttr(radius)
        prim.CreateHeightAttr(height)
        xform = UsdGeom.Xformable(prim.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(xy[0], xy[1], height * 0.5 + 0.04))
        UsdShade.MaterialBindingAPI(prim.GetPrim()).Bind(mat)

    cylinder("/World/RecordMarkers/Start", start_xy, start_mat, 0.35, 0.08)
    cylinder("/World/RecordMarkers/Goal", goal_xy, goal_mat, 0.42, 0.35)
    print(
        "[RECORD] start_goal_markers "
        f"start=({start_xy[0]:.3f},{start_xy[1]:.3f}) "
        f"goal=({goal_xy[0]:.3f},{goal_xy[1]:.3f})"
    )


class TopDownViewportRecorder:
    def __init__(
        self,
        output_path: str,
        frame_output_dir: str,
        fps: float,
        every_n_steps: int,
        max_frames: int,
        camera_height: float,
        resolution: tuple[int, int],
    ):
        self.output_path = Path(output_path) if output_path else None
        self.frame_output_dir = Path(frame_output_dir) if frame_output_dir else None
        self.fps = fps
        self.every_n_steps = max(1, every_n_steps)
        self.max_frames = max_frames
        self.camera_height = camera_height
        self.resolution = resolution
        self.frame_count = 0
        self._viewport_api = None
        self._capture_frame_dir = None
        self._cleanup_capture_frame_dir = False
        self._saved_resolution = None

    def setup(self, env):
        import omni.kit.app
        from omni.kit.viewport.utility import get_active_viewport

        del env
        app = omni.kit.app.get_app()
        self._viewport_api = get_active_viewport()
        if self._viewport_api is None:
            raise RuntimeError("Playback recording requires an active Isaac Sim viewport.")

        try:
            self._saved_resolution = tuple(int(v) for v in self._viewport_api.resolution)
            self._viewport_api.resolution = tuple(int(v) for v in self.resolution)
        except Exception as exc:
            raise RuntimeError("Playback recording could not configure the active viewport resolution.") from exc

        if self.frame_output_dir is not None:
            self.frame_output_dir.mkdir(parents=True, exist_ok=True)
            self._capture_frame_dir = self.frame_output_dir
        else:
            self._capture_frame_dir = Path(tempfile.mkdtemp(prefix="sea_nav_play_frames_", dir="/tmp"))
            self._cleanup_capture_frame_dir = True

        for _ in range(4):
            app.update()

    def _capture_viewport_frame(self, frame_path: Path):
        import omni.kit.app
        import omni.renderer_capture
        from omni.kit.viewport.utility import capture_viewport_to_file

        if self._viewport_api is None:
            raise RuntimeError("Playback recording viewport was not initialized.")

        capture_helper = capture_viewport_to_file(self._viewport_api, file_path=str(frame_path))
        app = omni.kit.app.get_app()
        renderer = omni.renderer_capture.acquire_renderer_capture_interface()
        capture_future = asyncio.ensure_future(capture_helper.wait_for_result(0))
        deadline = time.time() + 10.0
        while time.time() < deadline:
            app.update()
            renderer.wait_async_capture()
            if capture_future.done():
                break
        if not capture_future.done():
            capture_future.cancel()
            raise RuntimeError(f"Viewport capture timed out before completion: {frame_path}")
        result = capture_future.result()
        if result and frame_path.is_file() and frame_path.stat().st_size > 0:
            return
        raise RuntimeError(f"Viewport capture did not produce a readable frame: {frame_path}")

    def _encode_video(self):
        import cv2

        if self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        frame_paths = [self._capture_frame_dir / f"frame_{index:04d}.png" for index in range(self.frame_count)]
        first_frame = cv2.imread(str(frame_paths[0]))
        if first_frame is None:
            raise RuntimeError(f"Failed to read first captured frame: {frame_paths[0]}")
        height, width = first_frame.shape[:2]
        writer = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self.output_path}")
        try:
            writer.write(first_frame)
            for frame_path in frame_paths[1:]:
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    raise RuntimeError(f"Failed to read captured frame: {frame_path}")
                writer.write(frame)
        finally:
            writer.release()
        if not self.output_path.is_file() or self.output_path.stat().st_size <= 0:
            raise RuntimeError(f"Viewport capture did not produce a readable video: {self.output_path}")

    def capture_if_due(self, step_idx: int):
        if self._viewport_api is None or step_idx % self.every_n_steps != 0:
            return
        if self.max_frames > 0 and self.frame_count >= self.max_frames:
            return
        frame_path = self._capture_frame_dir / f"frame_{self.frame_count:04d}.png"
        self._capture_viewport_frame(frame_path)
        self.frame_count += 1

    def close(self):
        if self.frame_count == 0:
            raise RuntimeError("No frames were captured for the top-down video")
        self._encode_video()
        if self.output_path is not None:
            print(f"[RECORD] topdown_video={self.output_path} frames={self.frame_count} fps={self.fps}")
        if self.frame_output_dir is not None:
            print(f"[RECORD] frame_dir={self.frame_output_dir} frames={self.frame_count}")
        if self._saved_resolution is not None and self._viewport_api is not None:
            try:
                self._viewport_api.resolution = self._saved_resolution
            except Exception:
                pass
        if self._cleanup_capture_frame_dir and self._capture_frame_dir is not None:
            shutil.rmtree(self._capture_frame_dir, ignore_errors=True)


def _write_summary(log_dir: Path, summary: dict, trace_rows: list[dict], room_map=None):
    log_dir.mkdir(parents=True, exist_ok=True)
    if room_map is not None:
        np.save(log_dir / "room.npy", np.asarray(room_map, dtype=np.float32))
        summary["room_path"] = str(log_dir / "room.npy")
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    with (log_dir / "trace.jsonl").open("w") as f:
        for row in trace_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _apply_policy_turn_prior(command, args):
    if args.policy_turn_yaw_threshold < 0.0:
        return command
    wz = float(command[0, 2].item())
    if abs(wz) < args.policy_turn_yaw_threshold:
        return command
    vx = float(command[0, 0].item())
    if wz >= 0.0 and args.policy_turn_forward_floor_pos >= 0.0 and vx < args.policy_turn_forward_floor_pos:
        command = command.clone()
        command[0, 0] = args.policy_turn_forward_floor_pos
    if wz < 0.0 and args.policy_turn_forward_floor_neg >= 0.0 and vx < args.policy_turn_forward_floor_neg:
        command = command.clone()
        command[0, 0] = args.policy_turn_forward_floor_neg
    return command


def _maybe_prepare_path_follow(env, args, fixed_start_cell, fixed_goal_cell):
    room_map = env.room_maps[0, 0].cpu().numpy()
    if fixed_start_cell is None:
        start_local_xy = (
            env._robot.data.root_pos_w[0, 0].item() - env._terrain.env_origins[0, 0].item(),
            env._robot.data.root_pos_w[0, 1].item() - env._terrain.env_origins[0, 1].item(),
        )
        goal_local_xy = (
            env.position_targets[0, 0].item() - env._terrain.env_origins[0, 0].item(),
            env.position_targets[0, 1].item() - env._terrain.env_origins[0, 1].item(),
        )
        start_cell = _local_xy_to_cell(start_local_xy)
        goal_cell = _local_xy_to_cell(goal_local_xy)
    else:
        start_cell = fixed_start_cell
        goal_cell = fixed_goal_cell
    path_cells = _astar_path(room_map, start_cell, goal_cell)
    import torch

    path_points_local = torch.tensor(
        [[_grid_to_local_xy(cell)[0], _grid_to_local_xy(cell)[1]] for cell in path_cells],
        dtype=torch.float,
        device=env.device,
    )
    env._probe_path_points_local = path_points_local
    env._probe_path_lookahead = args.path_lookahead


def _mean_tail(values, tail=50):
    if not values:
        return None
    clipped = values[-tail:]
    return float(sum(clipped) / len(clipped))


def _mean_vector_tail(values, tail=50):
    if not values:
        return None
    clipped = values[-tail:]
    width = len(clipped[0])
    return [float(sum(row[i] for row in clipped) / len(clipped)) for i in range(width)]


def _classify_stand_diagnostic(diagnostics):
    raw = diagnostics.get("mean_policy_command_abs_tail50")
    low_level = diagnostics.get("mean_low_level_command_abs_tail50")
    speed = diagnostics.get("mean_body_speed_tail50")
    yaw = diagnostics.get("mean_abs_yaw_rate_tail50")
    progress = diagnostics.get("mean_distance_progress_tail50")
    if raw is None or low_level is None or speed is None or yaw is None:
        return "insufficient_data"
    motion = speed + 0.5 * yaw
    if raw < 0.05:
        return "high_level_idle"
    if raw >= 0.10 and low_level < 0.05:
        return "command_suppressed_after_policy"
    if low_level >= 0.10 and motion < 0.12:
        return "low_level_not_tracking_or_blocked"
    if motion >= 0.12 and progress is not None and progress < 0.002:
        return "moving_without_progress"
    return "mixed_or_unknown"


def _can_vectorize_hard_room_eval(args, policy_fn, command_fn):
    return (
        args.scenario == "hard_room_eval"
        and args.num_envs > 1
        and policy_fn is not None
        and command_fn is None
        and not args.case_trace
        and not args.fixed_start_cell
        and not args.fixed_goal_cell
        and not args.trace_steps
        and not args.record_topdown_video
        and not args.record_frame_dir
        and not args.show_topdown_camera
        and not args.show_start_goal_markers
        and args.policy_turn_yaw_threshold < 0.0
        and args.policy_path_blend_weight < 0.0
    )


def _new_vector_episode_state(env, env_id, episode_idx):
    start_local_xy = (
        env._robot.data.root_pos_w[env_id, 0].item() - env._terrain.env_origins[env_id, 0].item(),
        env._robot.data.root_pos_w[env_id, 1].item() - env._terrain.env_origins[env_id, 1].item(),
    )
    goal_local_xy = (
        env.position_targets[env_id, 0].item() - env._terrain.env_origins[env_id, 0].item(),
        env.position_targets[env_id, 1].item() - env._terrain.env_origins[env_id, 1].item(),
    )
    return {
        "episode": episode_idx,
        "start_cell": list(_local_xy_to_cell(start_local_xy)),
        "goal_cell": list(_local_xy_to_cell(goal_local_xy)),
        "start_yaw": _yaw_from_quat(env._robot.data.root_quat_w[env_id].tolist()),
        "reward_sum": 0.0,
        "reach_reward_sum": 0.0,
        "min_distance": float("inf"),
        "first_reach_step": None,
        "step": 0,
        "previous_distance": None,
        "policy_command_abs_hist": [],
        "nav_action_abs_hist": [],
        "low_level_command_abs_hist": [],
        "body_speed_hist": [],
        "abs_yaw_rate_hist": [],
        "distance_hist": [],
        "distance_progress_hist": [],
        "policy_command_vec_hist": [],
        "nav_action_vec_hist": [],
        "low_level_command_vec_hist": [],
        "body_velocity_vec_hist": [],
    }


def _finalize_vector_episode(state, done_step, done_reason, done_flags, episode_summary=None):
    action_diagnostics = {
        "tail_window_steps": 50,
        "mean_policy_command_abs_tail50": _mean_tail(state["policy_command_abs_hist"]),
        "mean_nav_action_abs_tail50": _mean_tail(state["nav_action_abs_hist"]),
        "mean_low_level_command_abs_tail50": _mean_tail(state["low_level_command_abs_hist"]),
        "mean_body_speed_tail50": _mean_tail(state["body_speed_hist"]),
        "mean_abs_yaw_rate_tail50": _mean_tail(state["abs_yaw_rate_hist"]),
        "mean_distance_tail50": _mean_tail(state["distance_hist"]),
        "mean_distance_progress_tail50": _mean_tail(state["distance_progress_hist"]),
        "mean_policy_command_tail50": _mean_vector_tail(state["policy_command_vec_hist"]),
        "mean_nav_action_tail50": _mean_vector_tail(state["nav_action_vec_hist"]),
        "mean_low_level_command_tail50": _mean_vector_tail(state["low_level_command_vec_hist"]),
        "mean_body_velocity_tail50": _mean_vector_tail(state["body_velocity_vec_hist"]),
        "last_policy_command": state["policy_command_vec_hist"][-1] if state["policy_command_vec_hist"] else None,
        "last_nav_action": state["nav_action_vec_hist"][-1] if state["nav_action_vec_hist"] else None,
        "last_low_level_command": state["low_level_command_vec_hist"][-1] if state["low_level_command_vec_hist"] else None,
        "last_body_velocity": state["body_velocity_vec_hist"][-1] if state["body_velocity_vec_hist"] else None,
    }
    if done_reason == "stand":
        action_diagnostics["stand_classification"] = _classify_stand_diagnostic(action_diagnostics)
    return {
        "episode": state["episode"],
        "start_cell": state["start_cell"],
        "goal_cell": state["goal_cell"],
        "start_yaw": state["start_yaw"],
        "reward_sum": state["reward_sum"],
        "reach_reward_sum": state["reach_reward_sum"],
        "min_distance": None if state["min_distance"] == float("inf") else state["min_distance"],
        "first_reach_step": state["first_reach_step"],
        "done_step": done_step,
        "done_reason": done_reason,
        "done_flags": done_flags,
        "episode_summary": episode_summary,
        "action_diagnostics": action_diagnostics,
    }


def _reset_vector_eval_env(env, env_id):
    import torch

    env_ids = torch.tensor([env_id], dtype=torch.long, device=env.device)
    env._reset_idx(env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=0.0)


def _run_hard_room_eval_vectorized(env, args, policy_fn):
    import torch

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if env.num_envs <= 1:
        raise ValueError("vectorized hard_room_eval requires --num-envs > 1")

    room_map = env.room_maps[0, 0].detach().cpu().numpy()
    obs_dict, _ = env.reset()
    policy_obs = obs_dict["policy"]

    episode_rows = []
    states = [None for _ in range(env.num_envs)]
    next_episode_idx = 0
    initial_slots = min(env.num_envs, args.episodes)
    for env_id in range(initial_slots):
        states[env_id] = _new_vector_episode_state(env, env_id, next_episode_idx)
        next_episode_idx += 1

    success_count = 0
    collision_failures = 0
    timeout_failures = 0
    stand_failures = 0
    fall_failures = 0
    truncated_failures = 0
    max_step_failures = 0
    total_reward_sum = 0.0
    total_reach_reward_sum = 0.0
    min_distance_values = []
    stand_diagnostic_counts = {}
    active_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    while len(episode_rows) < args.episodes:
        active_indices = [idx for idx, state in enumerate(states) if state is not None]
        if not active_indices:
            break
        active_mask.zero_()
        active_mask[torch.tensor(active_indices, dtype=torch.long, device=env.device)] = True

        with torch.inference_mode():
            command = policy_fn(policy_obs)
        if args.policy_stop_radius >= 0.0:
            stop_mask = env.distance <= args.policy_stop_radius
            if args.policy_stop_mode == "zero":
                command = torch.where(stop_mask.unsqueeze(1), torch.zeros_like(command), command)
            else:
                scales = (env.distance / args.policy_stop_radius).clamp(0.0, 1.0).unsqueeze(1)
                command = torch.where(stop_mask.unsqueeze(1), command * scales, command)
        command = torch.where(active_mask.unsqueeze(1), command, torch.zeros_like(command))

        raw_command = command.detach().clone()
        body_velocity = env._robot.data.root_lin_vel_b.detach().clone()
        yaw_rate = env._robot.data.root_ang_vel_b[:, 2].detach().clone()
        obs_dict, rewards, terminated, truncated, extras = env.step(command)
        policy_obs = obs_dict["policy"]

        reach_rewards = env.last_reward_terms["reach_pos_target_tight"].detach().clone()
        distances = env.last_distance.detach().clone()
        nav_actions = env.nav_actions_orig.detach().clone()
        low_level_commands = env.slr_commands.detach().clone()
        done_mask = (terminated | truncated).detach().clone()

        for env_id in active_indices:
            state = states[env_id]
            if state is None:
                continue
            reward_value = float(rewards[env_id].item())
            reach_reward_value = float(reach_rewards[env_id].item())
            distance_value = float(distances[env_id].item())
            raw_command_vec = [float(x) for x in raw_command[env_id].tolist()]
            nav_action_vec = [float(x) for x in nav_actions[env_id].tolist()]
            low_level_command_vec = [float(x) for x in low_level_commands[env_id].tolist()]
            body_velocity_vec = [float(x) for x in body_velocity[env_id].tolist()]
            body_speed_value = math.sqrt(body_velocity_vec[0] ** 2 + body_velocity_vec[1] ** 2)
            yaw_rate_value = float(yaw_rate[env_id].item())

            if state["previous_distance"] is not None:
                state["distance_progress_hist"].append(state["previous_distance"] - distance_value)
            state["previous_distance"] = distance_value
            state["policy_command_abs_hist"].append(sum(abs(x) for x in raw_command_vec) / len(raw_command_vec))
            state["nav_action_abs_hist"].append(sum(abs(x) for x in nav_action_vec) / len(nav_action_vec))
            state["low_level_command_abs_hist"].append(
                sum(abs(x) for x in low_level_command_vec) / len(low_level_command_vec)
            )
            state["body_speed_hist"].append(body_speed_value)
            state["abs_yaw_rate_hist"].append(abs(yaw_rate_value))
            state["distance_hist"].append(distance_value)
            state["policy_command_vec_hist"].append(raw_command_vec)
            state["nav_action_vec_hist"].append(nav_action_vec)
            state["low_level_command_vec_hist"].append(low_level_command_vec)
            state["body_velocity_vec_hist"].append(body_velocity_vec)
            state["reward_sum"] += reward_value
            state["reach_reward_sum"] += reach_reward_value
            state["min_distance"] = min(state["min_distance"], distance_value)
            if state["first_reach_step"] is None and reach_reward_value > 0.0:
                state["first_reach_step"] = state["step"]

        done_ids = [idx for idx in active_indices if bool(done_mask[idx].item())]
        for env_id in done_ids:
            state = states[env_id]
            if state is None or len(episode_rows) >= args.episodes:
                states[env_id] = None
                continue
            done_flags = {
                "contact": bool(env.last_done_contact[env_id].item()),
                "goal_hold": bool(env.last_done_goal_hold[env_id].item()),
                "stand": bool(env.last_done_stand[env_id].item()),
                "fall": bool(env.last_done_fall[env_id].item()),
                "timeout": bool(env.last_done_timeout[env_id].item()),
            }
            if done_flags["goal_hold"]:
                done_reason = "goal_hold"
                success_count += 1
            elif done_flags["contact"]:
                done_reason = "contact"
                collision_failures += 1
            elif done_flags["timeout"] or bool(truncated[env_id].item()):
                done_reason = "timeout"
                timeout_failures += 1
                if bool(truncated[env_id].item()):
                    truncated_failures += 1
            elif done_flags["stand"]:
                done_reason = "stand"
                stand_failures += 1
            elif done_flags["fall"]:
                done_reason = "fall"
                fall_failures += 1
            else:
                done_reason = "terminated"

            row = _finalize_vector_episode(state, state["step"], done_reason, done_flags)
            if done_reason == "stand":
                classification = row["action_diagnostics"].get("stand_classification", "insufficient_data")
                stand_diagnostic_counts[classification] = stand_diagnostic_counts.get(classification, 0) + 1
            episode_rows.append(row)
            total_reward_sum += state["reward_sum"]
            total_reach_reward_sum += state["reach_reward_sum"]
            if state["min_distance"] != float("inf"):
                min_distance_values.append(state["min_distance"])

            if next_episode_idx < args.episodes:
                states[env_id] = _new_vector_episode_state(env, env_id, next_episode_idx)
                next_episode_idx += 1
            else:
                states[env_id] = None

        manual_reset_happened = False
        for env_id in active_indices:
            if states[env_id] is not None:
                states[env_id]["step"] += 1
                if states[env_id]["step"] >= args.max_steps:
                    if len(episode_rows) >= args.episodes:
                        states[env_id] = None
                        continue
                    state = states[env_id]
                    row = _finalize_vector_episode(
                        state,
                        args.max_steps - 1,
                        "max_steps",
                        {
                            "contact": False,
                            "goal_hold": False,
                            "stand": False,
                            "fall": False,
                            "timeout": False,
                        },
                    )
                    episode_rows.append(row)
                    max_step_failures += 1
                    total_reward_sum += state["reward_sum"]
                    total_reach_reward_sum += state["reach_reward_sum"]
                    if state["min_distance"] != float("inf"):
                        min_distance_values.append(state["min_distance"])
                    if next_episode_idx < args.episodes:
                        _reset_vector_eval_env(env, env_id)
                        manual_reset_happened = True
                        states[env_id] = _new_vector_episode_state(env, env_id, next_episode_idx)
                        next_episode_idx += 1
                    else:
                        states[env_id] = None
        if manual_reset_happened:
            policy_obs = env._get_observations()["policy"]

    episode_rows = sorted(episode_rows[: args.episodes], key=lambda row: row["episode"])
    episodes = max(1, args.episodes)
    collision_free_success_count = sum(
        1 for row in episode_rows if row["done_reason"] == "goal_hold" and not row["done_flags"]["contact"]
    )
    summary = {
        "scenario": args.scenario,
        "repro_mode": getattr(args, "repro_mode", "none"),
        "controller_mode": args.controller_mode,
        "actuator_mode": args.actuator_mode,
        "robot_asset_source": args.robot_asset_source,
        "low_level_controller": args.low_level_controller,
        "robotlab_low_level_policy": args.robotlab_low_level_policy,
        "robotlab_command_clip": args.robotlab_command_clip,
        "cbf_fov_deg": args.cbf_fov_deg,
        "checkpoint": args.checkpoint if args.checkpoint else None,
        "episodes": args.episodes,
        "num_envs": env.num_envs,
        "eval_parallel": True,
        "eval_obstacle_level": args.eval_obstacle_level,
        "max_steps": args.max_steps,
        "episode_length_s": args.episode_length_s,
        "nav_action_scale": list(args.nav_action_scale),
        "stay_steps": args.stay_steps,
        "disable_contact_termination": args.disable_contact_termination,
        "fixed_start_cell": None,
        "fixed_goal_cell": None,
        "fixed_start_yaw": None,
        "trace_steps": False,
        "policy_stop_radius": args.policy_stop_radius if args.policy_stop_radius >= 0.0 else None,
        "policy_stop_mode": args.policy_stop_mode if args.policy_stop_radius >= 0.0 else None,
        "policy_turn_yaw_threshold": None,
        "policy_turn_forward_floor_pos": None,
        "policy_turn_forward_floor_neg": None,
        "policy_path_blend_weight": None,
        "policy_path_blend_min_distance": None,
        "success_count": success_count,
        "collision_free_success_count": collision_free_success_count,
        "collision_failures": collision_failures,
        "timeout_failures": timeout_failures,
        "stand_failures": stand_failures,
        "fall_failures": fall_failures,
        "truncated_failures": truncated_failures,
        "max_step_failures": max_step_failures,
        "stand_diagnostic_counts": stand_diagnostic_counts,
        "success_rate": success_count / episodes,
        "collision_free_success_rate": collision_free_success_count / episodes,
        "collision_failure_rate": collision_failures / episodes,
        "mean_reward_sum": total_reward_sum / episodes,
        "mean_reach_reward_sum": total_reach_reward_sum / episodes,
        "mean_min_distance": None if not min_distance_values else sum(min_distance_values) / len(min_distance_values),
    }
    return summary, episode_rows, room_map


def _run_hard_room_eval(env, args, policy_fn, command_fn=None):
    import torch

    if env.num_envs != 1:
        raise ValueError(
            "serial hard_room_eval requires --num-envs 1. "
            "Use policy mode without trace/recording/fixed cases to enable vectorized eval."
        )

    room_map = env.room_maps[0, 0].detach().cpu().numpy()
    episode_rows = []
    trace_rows = []
    success_count = 0
    collision_failures = 0
    timeout_failures = 0
    stand_failures = 0
    fall_failures = 0
    truncated_failures = 0
    max_step_failures = 0
    total_reward_sum = 0.0
    total_reach_reward_sum = 0.0
    min_distance_values = []
    stand_diagnostic_counts = {}
    fixed_start_cell = _parse_cell_arg(args.fixed_start_cell)
    fixed_goal_cell = _parse_cell_arg(args.fixed_goal_cell)
    if (fixed_start_cell is None) != (fixed_goal_cell is None):
        raise ValueError("--fixed-start-cell and --fixed-goal-cell must be provided together")
    if args.trace_steps and args.episodes != 1:
        raise ValueError("--trace-steps currently requires --episodes 1")
    if policy_fn is None and command_fn is None:
        raise ValueError("hard_room_eval requires either a policy function or a scripted command function")
    if policy_fn is None and fixed_start_cell is None:
        raise ValueError("scripted hard_room_eval requires --fixed-start-cell and --fixed-goal-cell")
    recorder = None
    custom_viewer_eye = tuple(getattr(args, "viewer_camera_eye", ()) or ())
    custom_viewer_target = tuple(getattr(args, "viewer_camera_target", ()) or ())
    frame_output_dir = getattr(args, "record_frame_dir", "")

    for episode_idx in range(args.episodes):
        obs_dict, _ = env.reset()
        if fixed_start_cell is not None:
            env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
            start_xy = torch.tensor([_grid_to_local_xy(fixed_start_cell)], device=env.device)
            goal_xy = torch.tensor([_grid_to_local_xy(fixed_goal_cell)], device=env.device)
            env.set_manual_start_and_goal(env_ids, start_xy, goal_xy, yaw=args.fixed_start_yaw, root_height=0.42)
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(dt=0.0)
            obs_dict = env._get_observations()
        if args.controller_mode == "path_follow" or args.policy_path_blend_weight >= 0.0:
            _maybe_prepare_path_follow(env, args, fixed_start_cell, fixed_goal_cell)
        policy_obs = obs_dict["policy"]
        start_local_xy = (
            env._robot.data.root_pos_w[0, 0].item() - env._terrain.env_origins[0, 0].item(),
            env._robot.data.root_pos_w[0, 1].item() - env._terrain.env_origins[0, 1].item(),
        )
        goal_local_xy = (
            env.position_targets[0, 0].item() - env._terrain.env_origins[0, 0].item(),
            env.position_targets[0, 1].item() - env._terrain.env_origins[0, 1].item(),
        )
        start_yaw = _yaw_from_quat(env._robot.data.root_quat_w[0].tolist())
        if args.show_start_goal_markers:
            _add_start_goal_markers(env, start_local_xy, goal_local_xy)
        if (args.record_topdown_video or frame_output_dir) and recorder is None:
            recorder = TopDownViewportRecorder(
                output_path=args.record_topdown_video,
                frame_output_dir=frame_output_dir,
                fps=args.record_video_fps,
                every_n_steps=args.record_every_n_steps,
                max_frames=args.record_max_frames,
                camera_height=args.record_camera_height,
                resolution=(args.record_video_width, args.record_video_height),
            )
            if len(custom_viewer_eye) == 3 and len(custom_viewer_target) == 3:
                _setup_viewport_camera(env, custom_viewer_eye, custom_viewer_target, label="play_camera_ready")
            else:
                _setup_topdown_viewport_camera(env, start_local_xy, goal_local_xy, args.record_camera_height)
            recorder.setup(env)
            recorder.capture_if_due(0)
        elif args.show_topdown_camera:
            if len(custom_viewer_eye) == 3 and len(custom_viewer_target) == 3:
                _setup_viewport_camera(env, custom_viewer_eye, custom_viewer_target, label="play_camera_ready")
            else:
                _setup_topdown_viewport_camera(env, start_local_xy, goal_local_xy, args.record_camera_height)
        if (args.record_topdown_video or frame_output_dir or args.show_topdown_camera) and args.record_start_delay_s > 0.0:
            print(f"[RECORD] start_delay_s={args.record_start_delay_s}")
            time.sleep(args.record_start_delay_s)

        episode_reward_sum = 0.0
        episode_reach_reward_sum = 0.0
        min_distance = float("inf")
        first_reach_step = None
        done_step = None
        done_reason = None
        episode_summary = None
        done_flags = {
            "contact": False,
            "goal_hold": False,
            "stand": False,
            "fall": False,
            "timeout": False,
        }
        policy_command_abs_hist = []
        nav_action_abs_hist = []
        low_level_command_abs_hist = []
        body_speed_hist = []
        abs_yaw_rate_hist = []
        distance_hist = []
        distance_progress_hist = []
        policy_command_vec_hist = []
        nav_action_vec_hist = []
        low_level_command_vec_hist = []
        body_velocity_vec_hist = []
        previous_distance_value = None

        for step_idx in range(args.max_steps):
            if policy_fn is not None:
                with torch.inference_mode():
                    command = policy_fn(policy_obs)
                if args.policy_path_blend_weight >= 0.0 and env.distance[0].item() >= args.policy_path_blend_min_distance:
                    path_command = torch.tensor([_path_follow_command(step_idx, env)], dtype=torch.float, device=env.device)
                    weight = max(0.0, min(1.0, args.policy_path_blend_weight))
                    command = weight * command + (1.0 - weight) * path_command
                command = _apply_policy_turn_prior(command, args)
            else:
                command = torch.tensor([command_fn(step_idx, env)], dtype=torch.float, device=env.device)
            if args.policy_stop_radius >= 0.0 and env.distance[0].item() <= args.policy_stop_radius:
                if args.policy_stop_mode == "zero":
                    command = torch.zeros_like(command)
                else:
                    scale = max(0.0, min(1.0, env.distance[0].item() / args.policy_stop_radius))
                    command = command * scale
            raw_command_vec = [float(x) for x in command[0].tolist()]
            obs_dict, rewards, terminated, truncated, extras = env.step(command)
            if policy_fn is not None:
                policy_obs = obs_dict["policy"]
            if recorder is not None:
                recorder.capture_if_due(step_idx + 1)
            if args.step_sleep_s > 0.0:
                time.sleep(args.step_sleep_s)

            reward_value = rewards[0].item()
            reach_reward_value = env.last_reward_terms["reach_pos_target_tight"][0].item()
            distance_value = env.last_distance[0].item()
            yaw_value = _yaw_from_quat(env.last_root_quat_w[0].tolist())
            local_x = (env.last_root_pos_w[0, 0] - env._terrain.env_origins[0, 0]).item()
            local_y = (env.last_root_pos_w[0, 1] - env._terrain.env_origins[0, 1]).item()
            nav_action_vec = [float(x) for x in env.nav_actions_orig[0].tolist()]
            low_level_command_vec = [float(x) for x in env.slr_commands[0].tolist()]
            body_velocity_vec = [float(x) for x in env._robot.data.root_lin_vel_b[0].tolist()]
            yaw_rate_value = float(env._robot.data.root_ang_vel_b[0, 2].item())
            body_speed_value = math.sqrt(body_velocity_vec[0] ** 2 + body_velocity_vec[1] ** 2)
            if previous_distance_value is not None:
                distance_progress_hist.append(previous_distance_value - distance_value)
            previous_distance_value = distance_value
            policy_command_abs_hist.append(sum(abs(x) for x in raw_command_vec) / len(raw_command_vec))
            nav_action_abs_hist.append(sum(abs(x) for x in nav_action_vec) / len(nav_action_vec))
            low_level_command_abs_hist.append(sum(abs(x) for x in low_level_command_vec) / len(low_level_command_vec))
            body_speed_hist.append(body_speed_value)
            abs_yaw_rate_hist.append(abs(yaw_rate_value))
            distance_hist.append(distance_value)
            policy_command_vec_hist.append(raw_command_vec)
            nav_action_vec_hist.append(nav_action_vec)
            low_level_command_vec_hist.append(low_level_command_vec)
            body_velocity_vec_hist.append(body_velocity_vec)
            episode_reward_sum += reward_value
            episode_reach_reward_sum += reach_reward_value
            min_distance = min(min_distance, distance_value)
            if first_reach_step is None and reach_reward_value > 0.0:
                first_reach_step = step_idx

            if "episode" in extras and extras["episode"]:
                episode_summary = {key: _tensor_scalar(value) for key, value in extras["episode"].items()}

            if args.trace_steps:
                trace_rows.append(
                    {
                        "step": step_idx,
                        "episode_idx": episode_idx,
                        "command": raw_command_vec,
                        "nav_action_scaled": nav_action_vec,
                        "low_level_command": low_level_command_vec,
                        "root_lin_vel_b": body_velocity_vec,
                        "root_ang_vel_b": [float(x) for x in env._robot.data.root_ang_vel_b[0].tolist()],
                        "stay_timer": int(env.stay_timer[0].item()),
                        "goal_hold_timer": int(env.goal_hold_timer[0].item()),
                        "reward": reward_value,
                        "reach_reward": reach_reward_value,
                        "distance": distance_value,
                        "yaw": yaw_value,
                        "x": local_x,
                        "y": local_y,
                        "goal_local_x": env.last_goal_local_pos[0, 0].item(),
                        "goal_local_y": env.last_goal_local_pos[0, 1].item(),
                        "front_clearance": env.last_rays[0].min().item(),
                        "done_contact": bool(env.last_done_contact[0].item()),
                        "done_goal_hold": bool(env.last_done_goal_hold[0].item()),
                        "done_stand": bool(env.last_done_stand[0].item()),
                        "done_fall": bool(env.last_done_fall[0].item()),
                        "done_timeout": bool(env.last_done_timeout[0].item()),
                        "terminated": bool(terminated[0].item()),
                        "truncated": bool(truncated[0].item()),
                    }
                )

            if bool((terminated | truncated).item()):
                done_step = step_idx
                done_flags = {
                    "contact": bool(env.last_done_contact[0].item()),
                    "goal_hold": bool(env.last_done_goal_hold[0].item()),
                    "stand": bool(env.last_done_stand[0].item()),
                    "fall": bool(env.last_done_fall[0].item()),
                    "timeout": bool(env.last_done_timeout[0].item()),
                }
                if done_flags["goal_hold"]:
                    done_reason = "goal_hold"
                    success_count += 1
                elif done_flags["contact"]:
                    done_reason = "contact"
                    collision_failures += 1
                elif done_flags["timeout"] or bool(truncated[0].item()):
                    done_reason = "timeout"
                    timeout_failures += 1
                    if bool(truncated[0].item()):
                        truncated_failures += 1
                elif done_flags["stand"]:
                    done_reason = "stand"
                    stand_failures += 1
                elif done_flags["fall"]:
                    done_reason = "fall"
                    fall_failures += 1
                else:
                    done_reason = "terminated"
                break
        else:
            done_reason = "max_steps"
            max_step_failures += 1

        total_reward_sum += episode_reward_sum
        total_reach_reward_sum += episode_reach_reward_sum
        if min_distance != float("inf"):
            min_distance_values.append(min_distance)
        action_diagnostics = {
            "tail_window_steps": 50,
            "mean_policy_command_abs_tail50": _mean_tail(policy_command_abs_hist),
            "mean_nav_action_abs_tail50": _mean_tail(nav_action_abs_hist),
            "mean_low_level_command_abs_tail50": _mean_tail(low_level_command_abs_hist),
            "mean_body_speed_tail50": _mean_tail(body_speed_hist),
            "mean_abs_yaw_rate_tail50": _mean_tail(abs_yaw_rate_hist),
            "mean_distance_tail50": _mean_tail(distance_hist),
            "mean_distance_progress_tail50": _mean_tail(distance_progress_hist),
            "mean_policy_command_tail50": _mean_vector_tail(policy_command_vec_hist),
            "mean_nav_action_tail50": _mean_vector_tail(nav_action_vec_hist),
            "mean_low_level_command_tail50": _mean_vector_tail(low_level_command_vec_hist),
            "mean_body_velocity_tail50": _mean_vector_tail(body_velocity_vec_hist),
            "last_policy_command": policy_command_vec_hist[-1] if policy_command_vec_hist else None,
            "last_nav_action": nav_action_vec_hist[-1] if nav_action_vec_hist else None,
            "last_low_level_command": low_level_command_vec_hist[-1] if low_level_command_vec_hist else None,
            "last_body_velocity": body_velocity_vec_hist[-1] if body_velocity_vec_hist else None,
        }
        if done_reason == "stand":
            stand_classification = _classify_stand_diagnostic(action_diagnostics)
            action_diagnostics["stand_classification"] = stand_classification
            stand_diagnostic_counts[stand_classification] = stand_diagnostic_counts.get(stand_classification, 0) + 1
        episode_rows.append(
            {
                "episode": episode_idx,
                "start_cell": list(_local_xy_to_cell(start_local_xy)),
                "goal_cell": list(_local_xy_to_cell(goal_local_xy)),
                "start_yaw": start_yaw,
                "reward_sum": episode_reward_sum,
                "reach_reward_sum": episode_reach_reward_sum,
                "min_distance": None if min_distance == float("inf") else min_distance,
                "first_reach_step": first_reach_step,
                "done_step": done_step,
                "done_reason": done_reason,
                "done_flags": done_flags,
                "episode_summary": episode_summary,
                "action_diagnostics": action_diagnostics,
            }
        )

    if recorder is not None:
        recorder.close()

    episodes = max(1, args.episodes)
    collision_free_success_count = sum(
        1 for row in episode_rows if row["done_reason"] == "goal_hold" and not row["done_flags"]["contact"]
    )
    summary = {
        "scenario": args.scenario,
        "repro_mode": getattr(args, "repro_mode", "none"),
        "controller_mode": args.controller_mode,
        "actuator_mode": args.actuator_mode,
        "robot_asset_source": args.robot_asset_source,
        "low_level_controller": args.low_level_controller,
        "robotlab_low_level_policy": args.robotlab_low_level_policy,
        "robotlab_command_clip": args.robotlab_command_clip,
        "cbf_fov_deg": args.cbf_fov_deg,
        "checkpoint": args.checkpoint if args.checkpoint else None,
        "episodes": args.episodes,
        "eval_obstacle_level": args.eval_obstacle_level,
        "max_steps": args.max_steps,
        "episode_length_s": args.episode_length_s,
        "nav_action_scale": list(args.nav_action_scale),
        "stay_steps": args.stay_steps,
        "disable_contact_termination": args.disable_contact_termination,
        "fixed_start_cell": None if fixed_start_cell is None else list(fixed_start_cell),
        "fixed_goal_cell": None if fixed_goal_cell is None else list(fixed_goal_cell),
        "fixed_start_yaw": args.fixed_start_yaw if fixed_start_cell is not None else None,
        "trace_steps": args.trace_steps,
        "policy_stop_radius": args.policy_stop_radius if args.policy_stop_radius >= 0.0 else None,
        "policy_stop_mode": args.policy_stop_mode if args.policy_stop_radius >= 0.0 else None,
        "policy_turn_yaw_threshold": args.policy_turn_yaw_threshold if args.policy_turn_yaw_threshold >= 0.0 else None,
        "policy_turn_forward_floor_pos": args.policy_turn_forward_floor_pos if args.policy_turn_forward_floor_pos >= 0.0 else None,
        "policy_turn_forward_floor_neg": args.policy_turn_forward_floor_neg if args.policy_turn_forward_floor_neg >= 0.0 else None,
        "policy_path_blend_weight": args.policy_path_blend_weight if args.policy_path_blend_weight >= 0.0 else None,
        "policy_path_blend_min_distance": args.policy_path_blend_min_distance if args.policy_path_blend_weight >= 0.0 else None,
        "success_count": success_count,
        "collision_free_success_count": collision_free_success_count,
        "collision_failures": collision_failures,
        "timeout_failures": timeout_failures,
        "stand_failures": stand_failures,
        "fall_failures": fall_failures,
        "truncated_failures": truncated_failures,
        "max_step_failures": max_step_failures,
        "stand_diagnostic_counts": stand_diagnostic_counts,
        "success_rate": success_count / episodes,
        "collision_free_success_rate": collision_free_success_count / episodes,
        "collision_failure_rate": collision_failures / episodes,
        "mean_reward_sum": total_reward_sum / episodes,
        "mean_reach_reward_sum": total_reach_reward_sum / episodes,
        "mean_min_distance": None if not min_distance_values else sum(min_distance_values) / len(min_distance_values),
    }
    trace_rows.extend(episode_rows)
    return summary, trace_rows, room_map


def run_probe(args):
    _ensure_isaaclab_imports()
    _apply_repro_mode(args)
    _configure_probe_tuning(args)
    _preload_probe_runtime_dependencies()
    if (getattr(args, "record_topdown_video", "") or getattr(args, "record_frame_dir", "")) and getattr(
        args, "headless", False
    ):
        raise ValueError(
            "Isaac Lab playback recording currently requires a rendered GUI viewport. "
            "Re-run without --headless for --record-video/--save-frames."
        )
    if (
        getattr(args, "record_topdown_video", "")
        or getattr(args, "record_frame_dir", "")
        or getattr(args, "show_topdown_camera", False)
    ):
        setattr(args, "enable_cameras", True)
    simulation_app = AppLauncher(args).app
    try:
        return _run_with_sim_app(args)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


def main(argv=None):
    args = _parse_args(argv)
    return run_probe(args)


if __name__ == "__main__":
    main()
