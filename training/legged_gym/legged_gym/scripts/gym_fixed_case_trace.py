from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
RSL_RL_ROOT = REPO_ROOT / "training" / "rsl_rl"
if str(RSL_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RSL_RL_ROOT))

ROOM_RESOLUTION = 0.1
DEFAULT_CASE_TRACE = (
    REPO_ROOT
    / "logs"
    / "isaac_lab"
    / "G0_fresh2500_official_playstyle_100eps_20260610"
    / "G0_model2500_playstyle"
    / "level_9"
    / "manual_reward_probe_hard_room_eval"
    / "06_10_20-16-39"
    / "trace.jsonl"
)
DEFAULT_POLICY_CHECKPOINT = (
    REPO_ROOT
    / "logs"
    / "isaac_lab"
    / "G0_gym_equiv_training"
    / "06_10_17-58-56_1024env_2500it_torque_fix_20260610"
    / "model_2500.pt"
)


def _parse_args():
    from isaacgym import gymutil

    custom_parameters = [
        {"name": "--task", "type": str, "default": "go2_pos_rough"},
        {"name": "--resume", "action": "store_true", "default": False},
        {"name": "--experiment_name", "type": str},
        {"name": "--run_name", "type": str},
        {"name": "--load_run", "type": str},
        {"name": "--checkpoint", "type": int},
        {"name": "--no_wandb", "action": "store_true", "default": True},
        {"name": "--test", "action": "store_true", "default": False},
        {"name": "--headless", "action": "store_true", "default": True},
        {"name": "--horovod", "action": "store_true", "default": False},
        {"name": "--rl_device", "type": str, "default": "cuda:0"},
        {"name": "--num_envs", "type": int},
        {"name": "--seed", "type": int, "default": 20260610},
        {"name": "--max_iterations", "type": int},
        {"name": "--case-trace", "type": str, "default": str(DEFAULT_CASE_TRACE)},
        {"name": "--case-episodes", "type": str, "default": "3,6,8,14,20,18,52,0,1,2"},
        {"name": "--preset-room-npy", "type": str, "default": ""},
        {"name": "--policy-checkpoint", "type": str, "default": str(DEFAULT_POLICY_CHECKPOINT)},
        {"name": "--policy-jit", "type": str, "default": ""},
        {"name": "--cbf-fov-deg", "type": float, "default": 180.0},
        {"name": "--episode-length-s", "type": float, "default": 40.0},
        {"name": "--max-steps", "type": int, "default": 2000},
        {"name": "--stay-steps", "type": int, "default": 500},
        {"name": "--goal-reached-steps", "type": int, "default": 150},
        {"name": "--disable-contact-termination", "action": "store_true", "default": True},
        {"name": "--asset-file", "type": str, "default": "{LEGGED_GYM_ROOT_DIR}/resources/go2_description/urdf/go2_description.urdf"},
        {"name": "--trace-steps", "action": "store_true", "default": True},
        {"name": "--trace-rich-fields", "action": "store_true", "default": True},
        {"name": "--policy-stop-radius", "type": float, "default": -1.0},
        {"name": "--policy-stop-mode", "type": str, "default": "zero"},
        {"name": "--log-root", "type": str, "default": str(REPO_ROOT / "logs" / "legged_gym" / "fixed_case_parity")},
    ]
    args = gymutil.parse_arguments(description="SEA-Nav Gym fixed-case parity trace", custom_parameters=custom_parameters)
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    args.wandb = not args.no_wandb
    args.rl_device = args.sim_device
    if args.test:
        args.headless = False
        args.wandb = False
        args.num_envs = 1
    return args


def _parse_episode_list(raw: str) -> list[int]:
    if not raw:
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _load_cases_from_trace(trace_path: str, episode_ids: list[int]) -> list[dict]:
    trace_file = Path(trace_path).expanduser()
    wanted = set(episode_ids)
    cases: dict[int, dict] = {}
    with trace_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "start_cell" not in row or "goal_cell" not in row:
                continue
            episode = int(row["episode"])
            if episode not in wanted:
                continue
            cases[episode] = {
                "source_episode": episode,
                "start_cell": [int(row["start_cell"][0]), int(row["start_cell"][1])],
                "goal_cell": [int(row["goal_cell"][0]), int(row["goal_cell"][1])],
                "start_yaw": float(row.get("start_yaw", 0.0)),
                "source_done_reason": row.get("done_reason"),
                "source_min_distance": row.get("min_distance"),
                "source_first_reach_step": row.get("first_reach_step"),
            }
    missing = [episode for episode in episode_ids if episode not in cases]
    if missing:
        raise ValueError(f"case episodes not found in {trace_file}: {missing}")
    return [cases[episode] for episode in episode_ids]


def _case_room_path(args) -> Path:
    if args.preset_room_npy:
        return Path(args.preset_room_npy).expanduser()
    return Path(args.case_trace).expanduser().with_name("room.npy")


def _cell_to_gym_xy(cell: list[int]) -> tuple[float, float]:
    return float(cell[0]) * ROOM_RESOLUTION, float(cell[1]) * ROOM_RESOLUTION


def _tensor_list(value) -> list[float]:
    tensor = value.detach().cpu().reshape(-1)
    return [float(x) for x in tensor.tolist()]


def _yaw_from_xyzw(quat) -> float:
    q = quat.detach().cpu().reshape(-1)
    x, y, z, w = [float(v) for v in q.tolist()]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _bool_tensor_item(value) -> bool:
    return bool(value.detach().cpu().reshape(-1)[0].item())


def _mean_tail(values: list[float], tail: int = 50):
    if not values:
        return None
    clipped = values[-tail:]
    return float(sum(clipped) / len(clipped))


def _mean_vector_tail(values: list[list[float]], tail: int = 50):
    if not values:
        return None
    clipped = values[-tail:]
    width = len(clipped[0])
    return [float(sum(row[i] for row in clipped) / len(clipped)) for i in range(width)]


def _classify_stand_diagnostic(diagnostics: dict) -> str:
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


def _policy_debug_fields(policy_fn) -> dict:
    module = getattr(policy_fn, "__self__", None)
    fields = {}
    u_bar = getattr(module, "u_bar", None)
    u_s = getattr(module, "u_s", None)
    alpha = getattr(module, "alpha", None)
    if u_bar is not None:
        fields["cbf_u_bar"] = _tensor_list(u_bar[0])
    if u_s is not None:
        fields["cbf_u_s"] = _tensor_list(u_s[0])
    if u_bar is not None and u_s is not None:
        fields["cbf_delta"] = _tensor_list((u_s - u_bar)[0])
    if alpha is not None:
        fields["cbf_alpha"] = float(alpha[0].detach().cpu().reshape(-1)[0].item())
    return fields


def _rich_pre_step_fields(env, policy_fn, policy_obs) -> dict:
    obs_last_step = policy_obs[0, -env.cfg.env.num_obs_one_step :]
    props = obs_last_step[: env.cfg.env.num_props]
    log2_delay_rays = obs_last_step[env.cfg.env.num_props : env.cfg.env.num_props + env.cfg.env.num_rays]
    delay_goal = obs_last_step[-2:]
    fields = {
        "obs_last_step": _tensor_list(obs_last_step),
        "obs_props": _tensor_list(props),
        "obs_log2_delay_rays": _tensor_list(log2_delay_rays),
        "obs_delay_goal": _tensor_list(delay_goal),
        "obs_history_flat": _tensor_list(policy_obs[0]),
        "rays": _tensor_list(env.rays[0]),
        "delay_rays": _tensor_list(env.delay_rays[0]),
        "rays_hist": _tensor_list(env.rays_hist[0]),
        "goal_local_pos": _tensor_list(env.goal_local_pos[0]),
        "delay_goal": _tensor_list(env.delay_goal[0]),
        "goal_hist": _tensor_list(env.goal_hist[0]),
        "projected_gravity_b": _tensor_list(env.projected_gravity[0]),
        "base_lin_vel_b": _tensor_list(env.base_lin_vel[0]),
        "base_ang_vel_b": _tensor_list(env.base_ang_vel[0]),
        "slr_commands": _tensor_list(env.slr_commands[0]),
        "ray_angles": _tensor_list(env.ray_angles),
    }
    fields.update(_policy_debug_fields(policy_fn))
    return fields


def _rich_post_step_fields(env) -> dict:
    return {
        "reward_terms": {key: float(value[0].item()) for key, value in env.last_reward_terms.items()},
        "reach_goal": _bool_tensor_item(env.reach_goal),
        "static": _bool_tensor_item(env.last_static),
        "v_low": _bool_tensor_item(env.last_v_low),
        "d_low": _bool_tensor_item(env.last_d_low),
        "stand_still_flag": _bool_tensor_item(env.stand_still_flag),
        "goal_reached_flag": _bool_tensor_item(env.goal_reached_flag),
    }


def _load_direct_policy(env, checkpoint_path: str, cbf_fov_deg: float):
    from rsl_rl.modules.cbf_actor_critic import DifferentiableSafeActorCritic

    actor_critic = DifferentiableSafeActorCritic(
        num_actions=env.num_nav_actions,
        num_props=env.cfg.env.num_props,
        his_len=env.cfg.env.his_len,
        num_rays=env.rays.shape[1],
        init_noise_std=1.5,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        cbf_fov_deg=cbf_fov_deg,
    ).to(env.device)
    loaded = torch.load(checkpoint_path, map_location=env.device)
    actor_critic.load_state_dict(loaded["model_state_dict"])
    actor_critic.eval()
    print(f"[GYM TRACE] loaded direct policy: {checkpoint_path} cbf_fov_deg={cbf_fov_deg}")
    return actor_critic.act_inference


def _load_policy(env, train_cfg, args):
    if args.policy_jit:
        policy = torch.jit.load(args.policy_jit, map_location=env.device).eval()
        print(f"[GYM TRACE] loaded jit policy: {args.policy_jit}")
        return policy
    if args.policy_checkpoint:
        return _load_direct_policy(env, args.policy_checkpoint, args.cbf_fov_deg)

    from legged_gym.utils import task_registry

    train_cfg.runner.resume = True
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    print(f"[GYM TRACE] loaded runner policy: {task_registry.loaded_policy_path}")
    return ppo_runner.get_inference_policy(device=env.device)


def _configure_env(args, preset_room_npy: Path):
    import legged_gym.envs  # noqa: F401
    from legged_gym.utils import task_registry

    args.num_envs = 1
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.terrain_types = ["hard_room"]
    env_cfg.terrain.terrain_proportions = [1.0]
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = 0
    env_cfg.terrain.preset_room_npy = str(preset_room_npy)
    env_cfg.asset.file = args.asset_file
    env_cfg.replay.enable_collision_replay = False
    env_cfg.replay.early_reset_prob_range = [0.0, 0.0]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.max_push_vel_xy = 0.0
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.added_mass_range = [0, 0]
    env_cfg.domain_rand.randomize_yaw = False
    env_cfg.domain_rand.randomize_roll = False
    env_cfg.domain_rand.randomize_pitch = False
    env_cfg.domain_rand.randomize_xy = False
    env_cfg.domain_rand.randomize_velo = False
    env_cfg.env.episode_length_s = args.episode_length_s
    env_cfg.env.stay_time = args.stay_steps
    env_cfg.env.goal_reached_time = args.goal_reached_steps
    env_cfg.env.debug_viz = False
    if args.disable_contact_termination:
        env_cfg.asset.terminate_after_contacts_on = []
    return env_cfg, train_cfg


def _done_flags(env) -> dict[str, bool]:
    return {key: _bool_tensor_item(value) for key, value in env.last_done_flags.items()}


def _done_reason(flags: dict[str, bool], hit_max_steps: bool) -> str:
    if flags.get("goal_hold"):
        return "goal_hold"
    if flags.get("contact"):
        return "contact"
    if flags.get("stand"):
        return "stand"
    if flags.get("fall"):
        return "fall"
    if flags.get("timeout"):
        return "timeout"
    if hit_max_steps:
        return "max_steps"
    return "unknown"


def _run_case(env, policy_fn, case: dict, case_idx: int, args) -> tuple[dict, list[dict]]:
    env.reset()
    env.do_reset = False
    env_id = torch.tensor([0], device=env.device, dtype=torch.long)
    start_xy = torch.tensor(_cell_to_gym_xy(case["start_cell"]), device=env.device, dtype=torch.float)
    goal_xy = torch.tensor(_cell_to_gym_xy(case["goal_cell"]), device=env.device, dtype=torch.float)
    env.set_manual_start_and_goal(env_id, start_xy, goal_xy, yaw=case["start_yaw"], root_height=0.42)
    policy_obs = env.get_observations()

    trace_rows = []
    min_distance = float("inf")
    first_reach_step = None
    reward_sum = 0.0
    reach_reward_sum = 0.0
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
    body_velocity_hist = []
    prev_distance = None
    done_step = args.max_steps - 1
    done_flags = {"contact": False, "goal_hold": False, "stand": False, "fall": False, "timeout": False}

    for step in range(args.max_steps):
        with torch.no_grad():
            raw_command = policy_fn(policy_obs.detach())
        command = raw_command
        if args.policy_stop_radius >= 0.0 and float(env.distance[0].item()) <= args.policy_stop_radius:
            if args.policy_stop_mode == "zero":
                command = torch.zeros_like(command)
            else:
                scale = max(0.0, min(1.0, float(env.distance[0].item()) / args.policy_stop_radius))
                command = command * scale

        rich_pre = _rich_pre_step_fields(env, policy_fn, policy_obs.detach()) if args.trace_steps and args.trace_rich_fields else {}
        body_velocity = env.base_lin_vel.detach().clone()
        yaw_rate = env.base_ang_vel[:, 2].detach().clone()
        obs, _, rewards, dones, _ = env.step(command.detach())
        policy_obs = obs

        reward_value = float(rewards[0].item())
        reach_reward_value = float(env.last_reward_terms.get("reach_pos_target_tight", torch.zeros(1, device=env.device))[0].item())
        distance_value = float(env.distance[0].item())
        min_distance = min(min_distance, distance_value)
        if first_reach_step is None and bool(env.reach_goal[0].item()):
            first_reach_step = step
        reward_sum += reward_value
        reach_reward_sum += reach_reward_value

        raw_command_vec = [float(x) for x in raw_command[0].detach().cpu().tolist()]
        command_vec = [float(x) for x in command[0].detach().cpu().tolist()]
        nav_action_vec = [float(x) for x in env.nav_actions_orig[0].detach().cpu().tolist()]
        low_level_command_vec = [float(x) for x in env.slr_commands[0].detach().cpu().tolist()]
        body_velocity_vec = [float(x) for x in body_velocity[0].detach().cpu().tolist()]
        yaw_rate_value = float(yaw_rate[0].item())
        body_speed = float(torch.norm(body_velocity[0, :2]).item())
        distance_progress = 0.0 if prev_distance is None else prev_distance - distance_value
        prev_distance = distance_value

        policy_command_abs_hist.append(sum(abs(x) for x in raw_command_vec) / len(raw_command_vec))
        nav_action_abs_hist.append(sum(abs(x) for x in nav_action_vec) / len(nav_action_vec))
        low_level_command_abs_hist.append(sum(abs(x) for x in low_level_command_vec) / len(low_level_command_vec))
        body_speed_hist.append(body_speed)
        abs_yaw_rate_hist.append(abs(yaw_rate_value))
        distance_hist.append(distance_value)
        distance_progress_hist.append(distance_progress)
        policy_command_vec_hist.append(raw_command_vec)
        nav_action_vec_hist.append(nav_action_vec)
        low_level_command_vec_hist.append(low_level_command_vec)
        body_velocity_hist.append(body_velocity_vec)

        done_flags = _done_flags(env)
        done = bool(dones[0].item())
        hit_max_steps = step >= args.max_steps - 1
        if args.trace_steps:
            row = {
                "step": step,
                "episode_idx": case_idx,
                "source_episode": case["source_episode"],
                "command": command_vec,
                "raw_policy_command": raw_command_vec,
                "nav_action_scaled": nav_action_vec,
                "low_level_command": low_level_command_vec,
                "root_lin_vel_b": body_velocity_vec,
                "root_ang_vel_b": [float(x) for x in env.base_ang_vel[0].detach().cpu().tolist()],
                "stay_timer": int(env.stay_timer[0].item()),
                "goal_hold_timer": int(env.goal_hold_timer[0].item()),
                "reward": reward_value,
                "reach_reward": reach_reward_value,
                "distance": distance_value,
                "yaw": _yaw_from_xyzw(env.root_states[0, 3:7]),
                "x": float(env.root_states[0, 0].item() - 5.0),
                "y": float(env.root_states[0, 1].item() - 5.0),
                "goal_local_x": float(env.goal_local_pos[0, 0].item()),
                "goal_local_y": float(env.goal_local_pos[0, 1].item()),
                "front_clearance": float(env.rays[0, env.rays.shape[1] // 2].item()),
                "done_contact": done_flags["contact"],
                "done_goal_hold": done_flags["goal_hold"],
                "done_stand": done_flags["stand"],
                "done_fall": done_flags["fall"],
                "done_timeout": done_flags["timeout"],
                "terminated": done and not done_flags["timeout"],
                "truncated": done_flags["timeout"],
            }
            row.update(rich_pre)
            if args.trace_rich_fields:
                row.update(_rich_post_step_fields(env))
            trace_rows.append(row)

        if done or hit_max_steps:
            done_step = step
            break

    diagnostics = {
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
        "mean_body_velocity_tail50": _mean_vector_tail(body_velocity_hist),
        "last_policy_command": policy_command_vec_hist[-1] if policy_command_vec_hist else None,
        "last_nav_action": nav_action_vec_hist[-1] if nav_action_vec_hist else None,
        "last_low_level_command": low_level_command_vec_hist[-1] if low_level_command_vec_hist else None,
        "last_body_velocity": body_velocity_hist[-1] if body_velocity_hist else None,
    }
    if done_flags.get("stand"):
        diagnostics["stand_classification"] = _classify_stand_diagnostic(diagnostics)

    episode_row = {
        "episode": case_idx,
        "source_episode": case["source_episode"],
        "source_done_reason": case["source_done_reason"],
        "source_min_distance": case["source_min_distance"],
        "source_first_reach_step": case["source_first_reach_step"],
        "start_cell": case["start_cell"],
        "goal_cell": case["goal_cell"],
        "start_xy_world": list(_cell_to_gym_xy(case["start_cell"])),
        "goal_xy_world": list(_cell_to_gym_xy(case["goal_cell"])),
        "start_yaw": case["start_yaw"],
        "reward_sum": reward_sum,
        "reach_reward_sum": reach_reward_sum,
        "min_distance": min_distance,
        "first_reach_step": first_reach_step,
        "done_step": done_step,
        "done_reason": _done_reason(done_flags, done_step >= args.max_steps - 1),
        "done_flags": done_flags,
        "action_diagnostics": diagnostics,
    }
    return episode_row, trace_rows


def _write_outputs(log_dir: Path, summary: dict, trace_rows: list[dict], room_path: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    room = np.load(room_path)
    np.save(log_dir / "room.npy", room.astype(np.float32))
    summary["room_path"] = str(log_dir / "room.npy")
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    with (log_dir / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in trace_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main():
    args = _parse_args()
    room_path = _case_room_path(args)
    cases = _load_cases_from_trace(args.case_trace, _parse_episode_list(args.case_episodes))
    env_cfg, train_cfg = _configure_env(args, room_path)

    from legged_gym.utils import task_registry

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.do_reset = False
    policy_fn = _load_policy(env, train_cfg, args)

    log_dir = Path(args.log_root).expanduser() / datetime.now().strftime("%m_%d_%H-%M-%S")
    trace_rows = []
    episode_rows = []
    for case_idx, case in enumerate(cases):
        episode_row, step_rows = _run_case(env, policy_fn, case, case_idx, args)
        trace_rows.extend(step_rows)
        trace_rows.append(episode_row)
        episode_rows.append(episode_row)
        print(
            "[GYM TRACE] "
            f"case={case_idx} source_ep={case['source_episode']} "
            f"reason={episode_row['done_reason']} min_distance={episode_row['min_distance']:.3f} "
            f"first_reach={episode_row['first_reach_step']}"
        )

    success_count = sum(1 for row in episode_rows if row["done_reason"] == "goal_hold")
    stand_count = sum(1 for row in episode_rows if row["done_reason"] == "stand")
    timeout_count = sum(1 for row in episode_rows if row["done_reason"] == "timeout")
    fall_count = sum(1 for row in episode_rows if row["done_reason"] == "fall")
    contact_count = sum(1 for row in episode_rows if row["done_reason"] == "contact")
    first_reach_count = sum(1 for row in episode_rows if row["first_reach_step"] is not None)
    summary = {
        "mode": "gym_fixed_case_trace",
        "case_trace": args.case_trace,
        "case_episodes": [case["source_episode"] for case in cases],
        "policy_checkpoint": args.policy_checkpoint or None,
        "policy_jit": args.policy_jit or None,
        "cbf_fov_deg": args.cbf_fov_deg,
        "asset_file": args.asset_file,
        "episode_length_s": args.episode_length_s,
        "max_steps": args.max_steps,
        "stay_steps": args.stay_steps,
        "goal_reached_steps": args.goal_reached_steps,
        "disable_contact_termination": args.disable_contact_termination,
        "num_cases": len(episode_rows),
        "success_count": success_count,
        "first_reach_count": first_reach_count,
        "stand_count": stand_count,
        "timeout_count": timeout_count,
        "fall_count": fall_count,
        "contact_count": contact_count,
        "episodes": episode_rows,
    }
    _write_outputs(log_dir, summary, trace_rows, room_path)
    print(f"[GYM TRACE] output_dir={log_dir}")


if __name__ == "__main__":
    main()
