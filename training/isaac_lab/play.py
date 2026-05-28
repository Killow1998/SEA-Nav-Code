from __future__ import annotations

import argparse
from pathlib import Path

from manual_reward_probe import PROBE_TUNING, REPO_ROOT, run_probe
from train import AppLauncher, DEFAULT_USD_DIR


DEFAULT_LOG_ROOT = Path("/tmp/sea-nav-play")
DEFAULT_REPO_LOG_ROOT = REPO_ROOT / "logs" / "isaac_lab"
DEFAULT_MAIN_RUN_DIR = (
    REPO_ROOT
    / "logs"
    / "isaac_lab"
    / "Go2_pos_rough_isaaclab_robotlab_contract_curriculum"
    / "05_27_22-33-03_256env_16000it_adaptive_limited_curriculum_equiv_steps_v2"
)
DEFAULT_ROBOTLAB_POLICY = (
    REPO_ROOT / "training" / "isaac_lab" / "low_level_policies" / "robotlab_go2_flat_20260527" / "policy.pt"
)


def _resolve_default_checkpoint() -> Path:
    preferred = [
        DEFAULT_MAIN_RUN_DIR / "model_3500.pt",
        DEFAULT_MAIN_RUN_DIR / "best_goal_hold.pt",
        DEFAULT_MAIN_RUN_DIR / "best_reach.pt",
        DEFAULT_MAIN_RUN_DIR / "best_mean_reward.pt",
    ]
    for candidate in preferred:
        if candidate.is_file():
            return candidate

    model_candidates = []
    for path in DEFAULT_MAIN_RUN_DIR.glob("model_*.pt"):
        stem = path.stem
        try:
            step = int(stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        model_candidates.append((step, path))
    if model_candidates:
        model_candidates.sort()
        return model_candidates[-1][1]

    raise FileNotFoundError(
        "No default Isaac Lab playback checkpoint was found under "
        f"{DEFAULT_MAIN_RUN_DIR}. Pass --checkpoint explicitly."
    )


def _build_parser():
    parser = argparse.ArgumentParser(description="Isaac Lab playback script for SEA-Nav policies")

    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--episode-length-s", type=float, default=40.0)
    parser.add_argument("--eval-obstacle-level", type=int, default=3)
    parser.add_argument("--eval-room-profile", choices=("random", "visual_progressive"), default="random")
    parser.add_argument("--max-init-terrain-level", type=int, default=3)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--stay-steps", type=int, default=500)
    parser.add_argument("--enable-contact-termination", dest="disable_contact_termination", action="store_false")
    parser.set_defaults(disable_contact_termination=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--actuator-mode", choices=("implicit", "ideal_pd", "gym_torque", "robotlab_dc"), default="robotlab_dc")
    parser.add_argument("--low-level-controller", choices=("sea_nav_jit", "robotlab"), default="robotlab")
    parser.add_argument("--robotlab-low-level-policy", type=str, default=str(DEFAULT_ROBOTLAB_POLICY))
    parser.add_argument("--robot-asset-source", choices=("converted_urdf", "native_go2"), default="native_go2")
    parser.add_argument("--sim-device", type=str, default="cuda:0")
    parser.add_argument("--usd-dir", type=str, default=str(DEFAULT_USD_DIR))
    parser.add_argument("--log-root", type=str, default=str(DEFAULT_LOG_ROOT))
    parser.add_argument("--use-repo-log-root", action="store_true", default=False)
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
    parser.add_argument("--record-topdown-video", type=str, default="")
    parser.add_argument("--record-video-fps", type=float, default=24.0)
    parser.add_argument("--record-video-width", type=int, default=1024)
    parser.add_argument("--record-video-height", type=int, default=1024)
    parser.add_argument("--record-every-n-steps", type=int, default=4)
    parser.add_argument("--record-max-frames", type=int, default=240)
    parser.add_argument("--record-camera-height", type=float, default=16.0)
    parser.add_argument("--show-topdown-camera", action="store_true", default=False)
    parser.add_argument("--show-start-goal-markers", action="store_true", default=False)
    parser.add_argument("--record-start-delay-s", type=float, default=0.0)
    parser.add_argument("--step-sleep-s", type=float, default=0.0)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _to_probe_args(args):
    probe_kwargs = vars(args).copy()
    probe_kwargs.update(
        {
            "scenario": "hard_room_eval",
            "controller_mode": "policy",
            "policy_turn_yaw_threshold": -1.0,
            "policy_turn_forward_floor_pos": -1.0,
            "policy_turn_forward_floor_neg": -1.0,
            "policy_path_blend_weight": -1.0,
            "policy_path_blend_min_distance": 1.0,
            "turn_entry_x_cell": PROBE_TUNING["turn_entry_x_cell"],
            "pivot_yaw_target": PROBE_TUNING["pivot_yaw_target"],
            "pre_turn_vx": PROBE_TUNING["pre_turn_vx"],
            "turn_forward_vx": PROBE_TUNING["turn_forward_vx"],
            "turn_yaw_rate": PROBE_TUNING["turn_yaw_rate"],
            "path_lookahead": PROBE_TUNING["path_lookahead"],
        }
    )
    if probe_kwargs["use_repo_log_root"]:
        probe_kwargs["log_root"] = str(DEFAULT_REPO_LOG_ROOT)
    probe_kwargs.pop("use_repo_log_root", None)
    if not probe_kwargs["checkpoint"]:
        probe_kwargs["checkpoint"] = str(_resolve_default_checkpoint())
    return argparse.Namespace(**probe_kwargs)


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    return run_probe(_to_probe_args(args))


if __name__ == "__main__":
    main()
