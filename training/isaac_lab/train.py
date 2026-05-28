from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import traceback
import types
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
GO2_URDF_PATH = (
    REPO_ROOT
    / "training"
    / "legged_gym"
    / "resources"
    / "go2_description"
    / "urdf"
    / "go2_description_v8.urdf"
)
DEFAULT_USD_DIR = Path("/tmp/sea-nav-usd-cache")
ROOM_CELLS = 100


def _ensure_isaaclab_imports():
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    import isaaclab

    source_root = Path(isaaclab.__file__).resolve().parent / "source"
    isaaclab_pkg_root = source_root / "isaaclab" / "isaaclab"
    if isaaclab_pkg_root.exists() and str(isaaclab_pkg_root) not in isaaclab.__path__:
        isaaclab.__path__.append(str(isaaclab_pkg_root))
    for package_name in ("isaaclab_rl", "isaaclab_assets", "isaaclab_tasks"):
        package_root = source_root / package_name
        if package_root.exists() and str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
    return isaaclab


_ensure_isaaclab_imports()
from isaaclab.app import AppLauncher

torch = None
SummaryWriter = None


def _inject_wandb_stub():
    if importlib.util.find_spec("wandb") is not None:
        return

    stub = types.ModuleType("wandb")
    stub.init = lambda *args, **kwargs: None
    stub.log = lambda *args, **kwargs: None
    sys.modules["wandb"] = stub


def _build_train_cfg(args) -> dict:
    action_reg_min = args.action_reg_min
    action_reg_max = args.action_reg_max
    if action_reg_min is None:
        action_reg_min = (-1.0, -1.0, -1.0) if args.low_level_controller == "robotlab" else (-0.5, -0.8, -1.0)
    if action_reg_max is None:
        action_reg_max = (1.0, 1.0, 1.0) if args.low_level_controller == "robotlab" else (1.7, 0.8, 1.0)
    return {
        "runner": {
            "policy_class_name": "DifferentiableSafeActorCritic",
            "algorithm_class_name": "PPO",
            "num_steps_per_env": args.num_steps_per_env,
            "max_iterations": args.max_iterations,
            "save_interval": args.save_interval,
            "experiment_name": args.experiment_name,
            "run_name": args.run_name,
        },
        "algorithm": {
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": args.entropy_coef,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": args.learning_rate,
            "schedule": args.lr_schedule,
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "learning_rate_min": args.learning_rate_min,
            "learning_rate_max": args.learning_rate_max,
            "max_grad_norm": 1.0,
            "policy_anchor_coef": args.policy_anchor_coef,
            "action_reg_min": list(action_reg_min),
            "action_reg_max": list(action_reg_max),
        },
        "policy": {
            "init_noise_std": 1.5,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
    }


def _build_log_dir(args) -> Path:
    timestamp = datetime.now().strftime("%m_%d_%H-%M-%S")
    suffix = f"_{args.run_name}" if args.run_name else ""
    return Path(args.log_root) / args.experiment_name / f"{timestamp}{suffix}"


def _reward_config(args) -> dict:
    return {
        "termination": args.reward_scale_termination,
        "collision": args.reward_scale_collision,
        "close_obst_vel": args.reward_scale_close_obst_vel,
        "stuck": args.reward_scale_stuck,
        "progress": args.reward_scale_progress,
        "far_goal_stand": args.reward_scale_far_goal_stand,
        "velo_dir": args.reward_scale_velo_dir,
        "reach_pos_target_tight": args.reward_scale_reach_pos_target_tight,
    }


def _apply_policy_freeze(actor_critic, args):
    if not args.freeze_policy_trunk:
        total_params = sum(param.numel() for param in actor_critic.parameters())
        trainable_params = sum(param.numel() for param in actor_critic.parameters() if param.requires_grad)
        return {
            "enabled": False,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_prefixes": [],
        }

    frozen_prefixes = ("encoder.", "backbone.")
    for name, param in actor_critic.named_parameters():
        if name.startswith(frozen_prefixes):
            param.requires_grad_(False)

    total_params = sum(param.numel() for param in actor_critic.parameters())
    trainable_params = sum(param.numel() for param in actor_critic.parameters() if param.requires_grad)
    return {
        "enabled": True,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_prefixes": list(frozen_prefixes),
    }


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


def _make_turn_open_room(np):
    room = _make_empty_room(np)
    room[10:46, 30] = 1.0
    room[10:46, 40] = 1.0
    room[60, 35:86] = 1.0
    room[70, 35:86] = 1.0
    return room


def _load_preset_start_goal_cases(trace_path: str, episodes_arg: str) -> list[dict[str, object]]:
    trace_file = Path(trace_path)
    if not trace_file.is_file():
        raise FileNotFoundError(f"preset case trace does not exist: {trace_file}")

    requested_episodes = None
    if episodes_arg.strip():
        requested_episodes = {int(part.strip()) for part in episodes_arg.split(",") if part.strip()}
        if not requested_episodes:
            raise ValueError("--preset-case-episodes did not contain any valid episode index")

    cases: list[dict[str, object]] = []
    with trace_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            episode = int(row["episode"])
            if requested_episodes is not None and episode not in requested_episodes:
                continue
            start_cell = row.get("start_cell")
            goal_cell = row.get("goal_cell")
            start_yaw = row.get("start_yaw")
            if start_cell is None or goal_cell is None or start_yaw is None:
                raise ValueError(f"trace row for episode {episode} is missing start/goal/yaw")
            cases.append(
                {
                    "episode": episode,
                    "start_cell": [int(start_cell[0]), int(start_cell[1])],
                    "goal_cell": [int(goal_cell[0]), int(goal_cell[1])],
                    "start_yaw": float(start_yaw),
                }
            )

    if not cases:
        raise ValueError(f"no preset cases were loaded from {trace_file}")
    return cases


def _maybe_load_preset_room_from_trace(trace_path: str):
    import numpy as np

    trace_file = Path(trace_path)
    room_file = trace_file.with_name("room.npy")
    if not room_file.is_file():
        return None
    return np.load(room_file)


class SeaNavVecEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.num_obs = env.num_obs
        self.num_privileged_obs = None
        self.num_actions = env.num_nav_actions
        self.num_nav_actions = env.num_nav_actions
        self.num_props = env.num_props
        self.max_episode_length = env.max_episode_length
        self.device = env.device
        self.cfg = SimpleNamespace(env=SimpleNamespace(his_len=env.history_length))
        self._last_obs = None

    @property
    def rays(self):
        return self.env.rays

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    def reset(self):
        obs_dict, _ = self.env.reset()
        self._last_obs = obs_dict["policy"]
        return self._last_obs, None

    def step(self, actions: torch.Tensor):
        obs_dict, rewards, terminated, truncated, extras = self.env.step(actions)
        self._last_obs = obs_dict["policy"]
        dones = (terminated | truncated).to(dtype=torch.long)
        if not self.env.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return self._last_obs, None, rewards, dones, extras

    def get_observations(self):
        if self._last_obs is None:
            self._last_obs = self.env._get_observations()["policy"]
        return self._last_obs

    def get_privileged_observations(self):
        return None

    def get_extras(self):
        return self.env.extras

    def close(self):
        self.env.close()


def _parse_args():
    parser = argparse.ArgumentParser(description="SEA-Nav Isaac Lab training entrypoint")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--actuator-mode", choices=("implicit", "ideal_pd", "gym_torque", "robotlab_dc"), default="ideal_pd"
    )
    parser.add_argument("--low-level-controller", choices=("sea_nav_jit", "robotlab"), default="sea_nav_jit")
    parser.add_argument(
        "--robotlab-low-level-policy",
        type=str,
        default=str(REPO_ROOT / "training" / "isaac_lab" / "low_level_policies" / "robotlab_go2_flat_20260527" / "policy.pt"),
    )
    parser.add_argument("--robot-asset-source", choices=("converted_urdf", "native_go2"), default="converted_urdf")
    parser.add_argument("--sim-device", type=str, default="cuda:0")
    parser.add_argument("--rl-device", type=str, default="cuda:0")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--episode-length-s", type=float, default=60.0)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--num-steps-per-env", type=int, default=48)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--experiment-name", type=str, default="Go2_pos_rough_isaaclab")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--learning-rate-min", type=float, default=1.0e-5)
    parser.add_argument("--learning-rate-max", type=float, default=1.0e-2)
    parser.add_argument("--lr-schedule", choices=("adaptive", "fixed"), default="adaptive")
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--action-reg-min", type=float, nargs=3, metavar=("VX", "VY", "WZ"), default=None)
    parser.add_argument("--action-reg-max", type=float, nargs=3, metavar=("VX", "VY", "WZ"), default=None)
    parser.add_argument("--log-root", type=str, default=str(REPO_ROOT / "logs" / "isaac_lab"))
    parser.add_argument("--usd-dir", type=str, default=str(DEFAULT_USD_DIR))
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--resume-load-optimizer", action="store_true", default=False)
    parser.add_argument("--policy-anchor-coef", type=float, default=0.0)
    parser.add_argument("--policy-anchor-from", type=str, default="")
    parser.add_argument("--freeze-policy-trunk", action="store_true", default=False)
    parser.add_argument("--preset-room-scenario", choices=("none", "straight", "turn_open"), default="none")
    parser.add_argument("--preset-case-trace", type=str, default="")
    parser.add_argument("--preset-case-episodes", type=str, default="")
    parser.add_argument("--preset-case-prob", type=float, default=1.0)
    parser.add_argument("--terrain-rows", type=int, default=10)
    parser.add_argument("--terrain-cols", type=int, default=10)
    parser.add_argument("--obstacle-level", type=int, default=9)
    parser.add_argument("--goal-stop-radius", type=float, default=-1.0)
    parser.add_argument("--goal-stop-mode", choices=("zero", "linear"), default="zero")
    parser.add_argument("--nav-action-scale", type=float, nargs=3, metavar=("VX", "VY", "WZ"), default=(1.0, 1.0, 1.0))
    parser.add_argument("--turn-yaw-threshold", type=float, default=-1.0)
    parser.add_argument("--turn-forward-floor-pos", type=float, default=-1.0)
    parser.add_argument("--turn-forward-floor-neg", type=float, default=-1.0)
    parser.add_argument("--reward-scale-termination", type=float, default=-100.0)
    parser.add_argument("--reward-scale-collision", type=float, default=-4.0)
    parser.add_argument("--reward-scale-close-obst-vel", type=float, default=5.0)
    parser.add_argument("--reward-scale-stuck", type=float, default=-5.0)
    parser.add_argument("--reward-scale-progress", type=float, default=0.0)
    parser.add_argument("--reward-scale-far-goal-stand", type=float, default=0.0)
    parser.add_argument("--reward-scale-velo-dir", type=float, default=4.0)
    parser.add_argument("--reward-scale-reach-pos-target-tight", type=float, default=10.0)
    parser.add_argument("--reward-scale-ang-vel-xy", type=float, default=-0.05)
    parser.add_argument("--disable-friction-rand", action="store_true", default=False)
    parser.add_argument("--disable-base-mass-rand", action="store_true", default=False)
    parser.add_argument("--disable-obs-noise", action="store_true", default=False)
    parser.add_argument("--training-loop", choices=("original", "metrics"), default="original")
    parser.add_argument("--wandb", action="store_true", default=False)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _make_go2_usd(args):
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    usd_dir = Path(args.usd_dir)
    usd_dir.mkdir(parents=True, exist_ok=True)
    converter_cfg = UrdfConverterCfg(
        asset_path=str(GO2_URDF_PATH),
        usd_dir=str(usd_dir),
        fix_base=False,
        merge_fixed_joints=True,
        self_collision=False,
        replace_cylinders_with_capsules=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=30.0, damping=0.75),
        ),
    )
    return UrdfConverter(converter_cfg).usd_path


def _print_smoke_stats(step_idx: int, rewards: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor, obs: torch.Tensor):
    print(
        "[SMOKE] "
        f"step={step_idx} "
        f"reward_mean={rewards.mean().item():.4f} "
        f"terminated={int(terminated.sum().item())} "
        f"truncated={int(truncated.sum().item())} "
        f"obs_shape={tuple(obs.shape)}"
    )


def _learn_with_metrics(runner, num_learning_iterations: int, init_at_random_ep_len: bool, writer: SummaryWriter | None):
    if init_at_random_ep_len:
        runner.env.episode_length_buf = torch.randint_like(
            runner.env.episode_length_buf, high=int(runner.env.max_episode_length)
        )

    obs = runner.env.get_observations()
    privileged_obs = runner.env.get_privileged_observations()
    infos = runner.env.get_extras()
    critic_obs = privileged_obs if privileged_obs is not None else obs
    obs, critic_obs = obs.to(runner.device), critic_obs.to(runner.device)
    runner.alg.actor_critic.train()

    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(runner.env.num_envs, dtype=torch.float, device=runner.device)
    cur_episode_length = torch.zeros(runner.env.num_envs, dtype=torch.float, device=runner.device)
    best_mean_reward = float("-inf")
    best_reach_metric = float("-inf")
    best_goal_hold_metric = float("-inf")

    start_iter = runner.current_learning_iteration
    end_iter = start_iter + num_learning_iterations
    for it in range(start_iter, end_iter):
        rollout_start = time.time()
        ep_infos = []
        rollout_reward_mean = 0.0

        with torch.no_grad():
            for _ in range(runner.num_steps_per_env):
                actions = runner.alg.act(obs, critic_obs)
                obs, privileged_obs, rewards, dones, infos = runner.env.step(actions)
                critic_obs = privileged_obs if privileged_obs is not None else obs
                obs = obs.to(runner.device)
                critic_obs = critic_obs.to(runner.device)
                rewards = rewards.to(runner.device)
                dones = dones.to(runner.device)
                runner.alg.process_env_step(obs, rewards, dones, infos)

                rollout_reward_mean += rewards.mean().item()
                if "episode" in infos:
                    ep_infos.append(infos["episode"])
                cur_reward_sum += rewards
                cur_episode_length += 1
                new_ids = (dones > 0).nonzero(as_tuple=False)
                if len(new_ids) > 0:
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

            collection_time = time.time() - rollout_start
            learn_start = time.time()
            runner.alg.compute_returns(critic_obs, infos)

        (
            mean_value_loss,
            mean_surrogate_loss,
            mean_regularization_loss,
            mean_smooth_loss,
            mean_interv_loss,
            mean_policy_anchor_loss,
        ) = runner.alg.update()
        learn_time = time.time() - learn_start
        rollout_reward_mean /= max(1, runner.num_steps_per_env)

        ep_metric_sums: dict[str, float] = {}
        ep_metric_counts: dict[str, int] = {}
        for ep_info in ep_infos:
            for key, value in ep_info.items():
                if isinstance(value, torch.Tensor):
                    scalar = value.float().mean().item() if value.ndim > 0 else value.float().item()
                else:
                    scalar = float(value)
                ep_metric_sums[key] = ep_metric_sums.get(key, 0.0) + scalar
                ep_metric_counts[key] = ep_metric_counts.get(key, 0) + 1
        ep_metrics = {key: ep_metric_sums[key] / ep_metric_counts[key] for key in ep_metric_sums}

        if writer is not None:
            writer.add_scalar("Loss/value_function", mean_value_loss, it)
            writer.add_scalar("Loss/surrogate", mean_surrogate_loss, it)
            writer.add_scalar("Loss/regularization", mean_regularization_loss, it)
            writer.add_scalar("Loss/smooth", mean_smooth_loss, it)
            writer.add_scalar("Loss/intervention", mean_interv_loss, it)
            writer.add_scalar("Loss/policy_anchor", mean_policy_anchor_loss, it)
            writer.add_scalar("Train/rollout_reward_mean", rollout_reward_mean, it)
            writer.add_scalar("Perf/collection_time_s", collection_time, it)
            writer.add_scalar("Perf/learn_time_s", learn_time, it)
            if len(rewbuffer) > 0:
                writer.add_scalar("Train/episode_reward_mean", sum(rewbuffer) / len(rewbuffer), it)
                writer.add_scalar("Train/episode_length_mean", sum(lenbuffer) / len(lenbuffer), it)
            for key, scalar in ep_metrics.items():
                writer.add_scalar(f"Episode/{key}", scalar, it)

        if len(rewbuffer) > 0:
            mean_reward = sum(rewbuffer) / len(rewbuffer)
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                runner.save(os.path.join(runner.log_dir, "best_mean_reward.pt"), iter_override=it)
        reach_metric = ep_metrics.get("rew_reach_pos_target_tight")
        if reach_metric is not None and reach_metric > best_reach_metric:
            best_reach_metric = reach_metric
            runner.save(os.path.join(runner.log_dir, "best_reach.pt"), iter_override=it)
        goal_hold_metric = ep_metrics.get("goal_hold_success")
        if goal_hold_metric is not None and goal_hold_metric > best_goal_hold_metric:
            best_goal_hold_metric = goal_hold_metric
            runner.save(os.path.join(runner.log_dir, "best_goal_hold.pt"), iter_override=it)

        message = (
            f"[TRAIN] iter={it} rollout_reward_mean={rollout_reward_mean:.4f} "
            f"value_loss={mean_value_loss:.4f} surrogate_loss={mean_surrogate_loss:.4f} "
            f"regularization_loss={mean_regularization_loss:.4f} smooth_loss={mean_smooth_loss:.4f} "
            f"interv_loss={mean_interv_loss:.4f} policy_anchor_loss={mean_policy_anchor_loss:.4f}"
        )
        if len(rewbuffer) > 0:
            message += (
                f" episode_reward_mean={sum(rewbuffer) / len(rewbuffer):.4f}"
                f" episode_length_mean={sum(lenbuffer) / len(lenbuffer):.2f}"
            )
        print(message)

        if runner.save_interval > 0 and (it + 1) % runner.save_interval == 0:
            runner.save(os.path.join(runner.log_dir, f"model_{it + 1}.pt"), iter_override=it + 1)

    runner.current_learning_iteration += num_learning_iterations
    runner.save(os.path.join(runner.log_dir, f"model_{runner.current_learning_iteration}.pt"))
    if writer is not None:
        writer.flush()


def main():
    global torch, SummaryWriter
    args = _parse_args()
    print(f"[INFO] parsed args: num_envs={args.num_envs} smoke_steps={args.smoke_steps} max_iterations={args.max_iterations}")
    if not args.wandb:
        _inject_wandb_stub()

    rsl_rl_root = REPO_ROOT / "training" / "rsl_rl"
    if str(rsl_rl_root) not in sys.path:
        sys.path.insert(0, str(rsl_rl_root))

    simulation_app = AppLauncher(args).app
    print("[INFO] simulation app launched")

    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter

    from rsl_rl.runners.on_policy_runner import OnPolicyRunner
    from sea_nav_env import SeaNavIsaacLabEnv, make_sea_nav_env_cfg
    print("[INFO] project imports resolved")

    env = None
    try:
        if args.robot_asset_source == "converted_urdf":
            print(f"[INFO] converting URDF: {GO2_URDF_PATH}")
            go2_usd_path = _make_go2_usd(args)
            print(f"[INFO] USD ready: {go2_usd_path}")
        else:
            go2_usd_path = None
            print("[INFO] using native Isaac Lab Go2 asset")
        preset_room = None
        preset_start_goal_cases = None
        terrain_rows = args.terrain_rows
        terrain_cols = args.terrain_cols
        obstacle_level = args.obstacle_level
        if args.preset_room_scenario == "straight":
            preset_room = _make_straight_room(np)
            terrain_rows = 1
            terrain_cols = 1
            obstacle_level = 0
        elif args.preset_room_scenario == "turn_open":
            preset_room = _make_turn_open_room(np)
            terrain_rows = 1
            terrain_cols = 1
            obstacle_level = 0
        if args.preset_case_trace:
            if not (0.0 <= args.preset_case_prob <= 1.0):
                raise ValueError("--preset-case-prob must be within [0, 1]")
            preset_start_goal_cases = _load_preset_start_goal_cases(args.preset_case_trace, args.preset_case_episodes)
            preset_room_from_trace = _maybe_load_preset_room_from_trace(args.preset_case_trace)
            if preset_room_from_trace is not None:
                preset_room = preset_room_from_trace
            terrain_rows = 1
            terrain_cols = 1
            obstacle_level = 9
            print(
                "[INFO] loaded preset start/goal cases "
                f"count={len(preset_start_goal_cases)} "
                f"trace={args.preset_case_trace} "
                f"episodes={args.preset_case_episodes or 'all'} "
                f"prob={args.preset_case_prob} "
                f"room={'yes' if preset_room_from_trace is not None else 'no'}"
            )
        env_cfg = make_sea_nav_env_cfg(
            go2_usd_path=go2_usd_path,
            num_envs=args.num_envs,
            seed=args.seed,
            actuator_mode=args.actuator_mode,
            robot_asset_source=args.robot_asset_source,
            terrain_rows=terrain_rows,
            terrain_cols=terrain_cols,
            obstacle_level=obstacle_level,
            episode_length_s=args.episode_length_s,
            goal_stop_radius=args.goal_stop_radius,
            goal_stop_mode=args.goal_stop_mode,
            nav_action_scale=tuple(args.nav_action_scale),
            turn_yaw_threshold=args.turn_yaw_threshold,
            turn_forward_floor_pos=args.turn_forward_floor_pos,
            turn_forward_floor_neg=args.turn_forward_floor_neg,
            low_level_controller=args.low_level_controller,
            robotlab_policy_path=args.robotlab_low_level_policy,
            reward_scale_termination=args.reward_scale_termination,
            reward_scale_collision=args.reward_scale_collision,
            reward_scale_close_obst_vel=args.reward_scale_close_obst_vel,
            reward_scale_stuck=args.reward_scale_stuck,
            reward_scale_progress=args.reward_scale_progress,
            reward_scale_far_goal_stand=args.reward_scale_far_goal_stand,
            reward_scale_velo_dir=args.reward_scale_velo_dir,
            reward_scale_reach_pos_target_tight=args.reward_scale_reach_pos_target_tight,
            reward_scale_ang_vel_xy=args.reward_scale_ang_vel_xy,
            randomize_friction=not args.disable_friction_rand,
            randomize_base_mass=not args.disable_base_mass_rand,
            preset_room=preset_room,
            preset_start_goal_cases=preset_start_goal_cases,
            preset_start_goal_case_prob=args.preset_case_prob,
        )
        env_cfg.sim.device = args.sim_device
        if args.disable_obs_noise:
            env_cfg.add_noise = False
        print(f"[INFO] building env on device={env_cfg.sim.device}")
        env = SeaNavIsaacLabEnv(cfg=env_cfg, render_mode=None)

        if args.smoke_steps > 0:
            obs_dict, _ = env.reset()
            policy_obs = obs_dict["policy"]
            print(
                "[INFO] reset complete "
                f"obs_shape={tuple(policy_obs.shape)} "
                f"action_dim={env.num_nav_actions} "
                f"device={env.device}"
            )
            zero_actions = torch.zeros(env.num_envs, env.num_nav_actions, device=env.device)
            for step_idx in range(1, args.smoke_steps + 1):
                obs_dict, rewards, terminated, truncated, _ = env.step(zero_actions)
                _print_smoke_stats(step_idx, rewards, terminated, truncated, obs_dict["policy"])
            return

        log_dir = _build_log_dir(args)
        log_dir.mkdir(parents=True, exist_ok=True)
        wrapper = SeaNavVecEnvWrapper(env)
        train_cfg = _build_train_cfg(args)
        (log_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
        (log_dir / "train_cfg.json").write_text(json.dumps(train_cfg, indent=2, sort_keys=True) + "\n")
        runner = OnPolicyRunner(wrapper, train_cfg, log_dir=str(log_dir), args=args, device=args.rl_device)
        if args.resume_from:
            runner.load(args.resume_from, load_optimizer=args.resume_load_optimizer)
            runner.alg.learning_rate = args.learning_rate
            for param_group in runner.alg.optimizer.param_groups:
                param_group["lr"] = args.learning_rate
            print(
                "[INFO] resumed runner "
                f"checkpoint={args.resume_from} "
                f"load_optimizer={args.resume_load_optimizer} "
                f"current_iteration={runner.current_learning_iteration} "
                f"learning_rate={args.learning_rate} "
                f"schedule={args.lr_schedule}"
            )
        if args.policy_anchor_coef > 0.0:
            anchor_source = args.policy_anchor_from or args.resume_from
            if not anchor_source:
                raise ValueError("--policy-anchor-coef requires --policy-anchor-from or --resume-from")
            reference_actor_critic = copy.deepcopy(runner.alg.actor_critic)
            if not args.resume_from or os.path.abspath(anchor_source) != os.path.abspath(args.resume_from):
                loaded_anchor = torch.load(anchor_source, map_location=runner.device)
                reference_actor_critic.load_state_dict(loaded_anchor["model_state_dict"])
            runner.alg.set_reference_actor_critic(reference_actor_critic)
            print(
                "[INFO] policy anchor enabled "
                f"coef={args.policy_anchor_coef} "
                f"source={anchor_source}"
            )
        freeze_info = _apply_policy_freeze(runner.alg.actor_critic, args)
        if freeze_info["enabled"]:
            print(
                "[INFO] policy trunk freeze enabled "
                f"frozen_prefixes={freeze_info['frozen_prefixes']} "
                f"trainable_params={freeze_info['trainable_params']} "
                f"total_params={freeze_info['total_params']}"
            )
        print(
            "[INFO] runner ready "
            f"training_loop={args.training_loop} "
            f"obs_dim={wrapper.num_obs} "
            f"action_dim={wrapper.num_nav_actions} "
            f"device={wrapper.device}"
        )
        if args.training_loop == "original":
            runner.learn(
                num_learning_iterations=args.max_iterations,
                init_at_random_ep_len=True,
                config=_reward_config(args),
            )
        else:
            writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))
            _learn_with_metrics(
                runner=runner,
                num_learning_iterations=args.max_iterations,
                init_at_random_ep_len=True,
                writer=writer,
            )
            writer.close()
            print(f"[INFO] tensorboard_dir={log_dir / 'tensorboard'}")
        print(f"[INFO] training finished log_dir={log_dir}")
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
