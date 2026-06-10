from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg, TerrainGenerator
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass


REPO_ROOT = Path(__file__).resolve().parents[2]
CTRL_MODEL_DIR = REPO_ROOT / "training" / "legged_gym" / "legged_gym" / "ctrl_model"
ROBOTLAB_LOW_LEVEL_POLICY_PATH = (
    REPO_ROOT / "training" / "isaac_lab" / "low_level_policies" / "robotlab_go2_flat_20260527" / "policy.pt"
)

ORIGINAL_GYM_JOINT_ORDER = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]

# The low-level controller was trained against the legacy Isaac Gym ordering:
# Gym dof order was FL, FR, RL, RR by leg, and the original code swapped
# front and rear leg blocks via [3,4,5,0,1,2,9,10,11,6,7,8].
LOW_LEVEL_JOINT_ORDER = [ORIGINAL_GYM_JOINT_ORDER[i] for i in (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)]

DEFAULT_JOINT_ANGLES = {
    "FL_hip_joint": 0.1,
    "RL_hip_joint": 0.1,
    "FR_hip_joint": -0.1,
    "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,
    "FR_thigh_joint": 0.8,
    "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5,
    "RL_calf_joint": -1.5,
    "FR_calf_joint": -1.5,
    "RR_calf_joint": -1.5,
}

GO2_URDF_EFFORT_LIMITS = {
    "hip": 23.7,
    "thigh": 23.7,
    "calf": 45.43,
}


def is_path_with_obstacle(room: np.ndarray, robot_pos: list[int], goal_pos: list[int]) -> bool:
    x1, y1 = robot_pos
    x2, y2 = goal_pos
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    if dx > dy:
        err = dx / 2.0
        while x1 != x2:
            if room[x1, y1] > 0:
                return True
            err -= dy
            if err < 0:
                y1 += sy
                err += dx
            x1 += sx
    else:
        err = dy / 2.0
        while y1 != y2:
            if room[x1, y1] > 0:
                return True
            err -= dx
            if err < 0:
                x1 += sx
                err += dy
            y1 += sy
    return False


def is_far_from_obstacles(room: np.ndarray, pos: list[int], min_distance: int) -> bool:
    y_start = max(0, pos[0] - min_distance)
    y_end = min(room.shape[0], pos[0] + min_distance + 1)
    x_start = max(0, pos[1] - min_distance)
    x_end = min(room.shape[1], pos[1] + min_distance + 1)
    neighborhood = room[y_start:y_end, x_start:x_end]
    return neighborhood.size > 0 and (neighborhood <= 0.1).all()


def place_robot_and_goal(
    room: np.ndarray,
    min_distance: int = 5,
    min_goal_distance: int = 35,
    max_attempts: int = 1000,
) -> tuple[list[int], list[int]]:
    grid_size = room.shape[0]
    fallback: tuple[list[int], list[int]] | None = None
    for _ in range(max_attempts):
        robot_pos = [np.random.randint(1, grid_size - 1), np.random.randint(1, grid_size - 1)]
        goal_pos = [np.random.randint(1, grid_size - 1), np.random.randint(1, grid_size - 1)]
        if room[tuple(robot_pos)] != 0 or room[tuple(goal_pos)] != 0:
            continue
        if not is_far_from_obstacles(room, robot_pos, min_distance):
            continue
        if not is_far_from_obstacles(room, goal_pos, min_distance):
            continue
        if np.linalg.norm(np.array(robot_pos) - np.array(goal_pos)) <= min_goal_distance:
            continue
        if fallback is None:
            fallback = (robot_pos, goal_pos)
        if is_path_with_obstacle(room, robot_pos, goal_pos):
            return robot_pos, goal_pos
    if fallback is not None:
        return fallback
    # Extremely dense maps can make the clearance constraint impossible; fall
    # back to free distant cells rather than hanging reset forever.
    for _ in range(max_attempts):
        robot_pos = [np.random.randint(1, grid_size - 1), np.random.randint(1, grid_size - 1)]
        goal_pos = [np.random.randint(1, grid_size - 1), np.random.randint(1, grid_size - 1)]
        if room[tuple(robot_pos)] != 0 or room[tuple(goal_pos)] != 0:
            continue
        if np.linalg.norm(np.array(robot_pos) - np.array(goal_pos)) > min_goal_distance:
            return robot_pos, goal_pos
    raise RuntimeError("Failed to sample valid robot/goal positions")


def create_room(grid_size: int = 10) -> np.ndarray:
    room = np.zeros((grid_size, grid_size), dtype=float)
    room[0, :] = 1.0
    room[-1, :] = 1.0
    room[:, 0] = 1.0
    room[:, -1] = 1.0
    return room


def generate_random_shape(shape_size: int = 30, num_cells: int = 20) -> np.ndarray:
    shape = np.zeros((shape_size, shape_size))
    start_row = np.random.randint(1, shape_size - 1)
    start_col = np.random.randint(1, shape_size - 1)
    shape[start_row, start_col] = 1
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    cells = [(start_row, start_col)]
    while len(cells) < num_cells:
        current_row, current_col = cells[np.random.randint(len(cells))]
        direction = directions[np.random.randint(4)]
        new_row = current_row + direction[0]
        new_col = current_col + direction[1]
        if 0 <= new_row < shape_size and 0 <= new_col < shape_size:
            shape[new_row, new_col] = np.random.randint(40, 100) * 0.01
            cells.append((new_row, new_col))
    return shape


def add_obstacles(room: np.ndarray, level: int, grid_size: int = 100, shape_size: int = 3) -> np.ndarray:
    num_obstacles = int(level)
    for _ in range(num_obstacles):
        num_cells = np.random.randint(2, 4)
        shape = generate_random_shape(shape_size, num_cells)
        start_row = np.random.randint(1, grid_size - shape_size)
        start_col = np.random.randint(1, grid_size - shape_size)
        room_section = room[start_row : start_row + shape_size, start_col : start_col + shape_size]
        room[start_row : start_row + shape_size, start_col : start_col + shape_size] = np.maximum(room_section, shape)
    for _ in range(num_obstacles * 5):
        shape = generate_random_shape(3, 1)
        shape *= np.random.randint(40, 100) * 0.01
        start_row = np.random.randint(1, grid_size - 3)
        start_col = np.random.randint(1, grid_size - 3)
        room_section = room[start_row : start_row + 3, start_col : start_col + 3]
        room[start_row : start_row + 3, start_col : start_col + 3] = np.maximum(room_section, shape)
    return room


def scale_room(room: np.ndarray, scale_factor: int) -> np.ndarray:
    scaled_size = room.shape[0] * scale_factor
    scaled_room = np.zeros((scaled_size, scaled_size), dtype=float)
    for i in range(room.shape[0]):
        for j in range(room.shape[1]):
            scaled_room[i * scale_factor : (i + 1) * scale_factor, j * scale_factor : (j + 1) * scale_factor] = room[i, j]
    return scaled_room


def create_rand_room(level: int, grid_size: int = 20, target_size: int = 100) -> np.ndarray:
    room = create_room(grid_size)
    room = add_obstacles(room, level, grid_size)
    scale_factor = target_size // grid_size
    return scale_room(room, scale_factor)


def batch_ray_cast_torch(
    grid_batch: torch.Tensor,
    base_row: int,
    base_col: int,
    angles: torch.Tensor,
    max_radius: float,
    step_r: float,
) -> torch.Tensor:
    device = grid_batch.device
    num_envs, n_x, n_y = grid_batch.shape
    r_vals = torch.arange(0.0, max_radius + 1e-9, step_r, device=device)
    num_steps = r_vals.shape[0]
    num_rays = angles.shape[0]
    r_2d = r_vals.view(1, num_steps)
    angles_2d = angles.view(num_rays, 1)
    x_2d = r_2d * torch.sin(angles_2d)
    y_2d = -r_2d * torch.cos(angles_2d)
    row_2d = base_row - y_2d
    col_2d = base_col + x_2d
    row_2d_int = torch.round(row_2d).long()
    col_2d_int = torch.round(col_2d).long()
    row_3d = row_2d_int.unsqueeze(0).expand(num_envs, -1, -1)
    col_3d = col_2d_int.unsqueeze(0).expand(num_envs, -1, -1)
    valid_mask_3d = (row_3d >= 0) & (row_3d < n_x) & (col_3d >= 0) & (col_3d < n_y)
    row_clamped = row_3d.clamp(0, n_x - 1)
    col_clamped = col_3d.clamp(0, n_y - 1)
    env_idx = torch.arange(num_envs, device=device).view(num_envs, 1, 1).expand(-1, row_3d.shape[1], row_3d.shape[2])
    grid_vals_3d = torch.zeros_like(row_3d, dtype=grid_batch.dtype)
    grid_vals_3d[valid_mask_3d] = grid_batch[env_idx[valid_mask_3d], row_clamped[valid_mask_3d], col_clamped[valid_mask_3d]]
    boundary_mask_3d = (~valid_mask_3d) | (grid_vals_3d == 1)
    boundary_cum_bool = torch.cumsum(boundary_mask_3d.int(), dim=2) > 0
    idx = torch.arange(num_steps, device=device).view(1, 1, -1)
    masked_idx = torch.where(boundary_cum_bool, idx, torch.full_like(idx, num_steps))
    boundary_idx_2d = masked_idx.min(dim=2).values
    boundary_exists_2d = boundary_cum_bool.any(dim=2)
    final_step_2d = torch.where(boundary_exists_2d, boundary_idx_2d - 1, (num_steps - 1) * torch.ones_like(boundary_idx_2d))
    final_step_2d = torch.clamp(final_step_2d, min=0)
    return r_vals[final_step_2d]


@height_field_to_mesh
def sea_nav_room_terrain(difficulty: float, cfg: "SeaNavRoomTerrainCfg") -> np.ndarray:
    target_size = int(round(cfg.size[0] / cfg.horizontal_scale))
    if cfg.preset_room is not None:
        room = np.array(cfg.preset_room, copy=True)
        if room.shape != (target_size, target_size):
            raise ValueError(
                f"preset_room shape {room.shape} does not match expected {(target_size, target_size)}"
            )
        cfg.latest_obstacle_level = -1
    else:
        obstacle_level = int(round(float(cfg.obstacle_level) * float(np.clip(difficulty, 0.0, 1.0))))
        room = create_rand_room(obstacle_level, grid_size=cfg.grid_size, target_size=target_size)
        cfg.latest_obstacle_level = obstacle_level
    cfg.latest_room = room.copy()
    return np.rint(room / cfg.vertical_scale).astype(np.int16)


@configclass
class SeaNavRoomTerrainCfg(HfTerrainBaseCfg):
    function = sea_nav_room_terrain
    grid_size: int = 20
    target_size: int = 100
    obstacle_level: int = 9
    latest_room: np.ndarray | None = None
    latest_obstacle_level: int = -1
    preset_room: np.ndarray | None = None


class SeaNavTerrainGenerator(TerrainGenerator):
    def __init__(self, cfg: TerrainGeneratorCfg, device: str = "cpu"):
        self.generated_rooms: list[np.ndarray] = []
        super().__init__(cfg=cfg, device=device)

    def _get_terrain_mesh(self, difficulty: float, cfg):
        cfg = cfg.copy()
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed
        meshes, origin = cfg.function(difficulty, cfg)
        if getattr(cfg, "latest_room", None) is not None:
            self.generated_rooms.append(np.array(cfg.latest_room, copy=True))
        mesh = trimesh.util.concatenate(meshes)
        transform = np.eye(4)
        transform[0:2, -1] = -cfg.size[0] * 0.5, -cfg.size[1] * 0.5
        mesh.apply_transform(transform)
        origin += transform[0:3, -1]
        return mesh, origin


class SeaNavTerrainImporter(TerrainImporter):
    def __init__(self, cfg: TerrainImporterCfg):
        if cfg.terrain_type != "generator":
            super().__init__(cfg)
            self.generated_rooms = []
            return

        cfg.validate()
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device
        self.terrain_prim_paths = []
        self.terrain_origins = None
        self.env_origins = None
        self._terrain_flat_patches = {}

        terrain_generator = self.cfg.terrain_generator.class_type(cfg=self.cfg.terrain_generator, device=self.device)
        self.generated_rooms = terrain_generator.generated_rooms
        self.import_mesh("terrain", terrain_generator.terrain_mesh)
        if self.cfg.use_terrain_origins:
            self.configure_env_origins(terrain_generator.terrain_origins)
        else:
            self.configure_env_origins()
        self._terrain_flat_patches = terrain_generator.flat_patches
        self.set_debug_vis(self.cfg.debug_vis)


@configclass
class SeaNavEnvCfg(DirectRLEnvCfg):
    seed: int = 1
    episode_length_s = 60.0
    decimation = 4
    action_space = 3
    observation_space = 550
    state_space = 0
    is_finite_horizon = False
    num_rerenders_on_reset = 0
    room_size = 10.0
    room_resolution = 0.1
    room_cells = 100
    num_props = 12
    num_rays = 41
    history_length = 10
    goal_reached_steps = 150
    stay_steps = 150
    goal_stop_radius = -1.0
    goal_stop_mode = "zero"
    nav_action_scale = (1.0, 1.0, 1.0)
    turn_yaw_threshold = -1.0
    turn_forward_floor_pos = -1.0
    turn_forward_floor_neg = -1.0
    low_level_controller = "sea_nav_jit"
    robotlab_policy_path = str(ROBOTLAB_LOW_LEVEL_POLICY_PATH)
    robotlab_command_clip = 1.0
    actuator_mode = "ideal_pd"
    joint_action_scale = 0.25
    joint_stiffness = 30.0
    joint_damping = 0.75
    joint_effort_limit = 80.0
    preset_start_goal_cases: list[dict[str, object]] | None = None
    preset_start_goal_case_prob = 1.0
    enable_contact_termination = True
    termination_body_patterns = ("base", "Head_upper", "Head_lower")
    penalized_body_patterns = ("base", "Head_upper", "Head_lower", ".*_thigh", ".*_calf")
    command_delay_s = 0.1
    add_noise = True
    slr_noise_ang_vel = 0.1
    slr_noise_gravity = 0.05
    slr_noise_dof_pos = 0.01
    slr_noise_dof_vel = 1.0
    nav_noise_gravity = 0.05
    nav_noise_lin_vel = 0.1
    nav_noise_ang_vel = 0.1
    position_target_sigma_soft = 2.0
    position_target_sigma_tight = 0.5
    reward_scale_termination = -100.0
    reward_scale_collision = -4.0
    reward_scale_close_obst_vel = 5.0
    reward_scale_stuck = -5.0
    reward_scale_progress = 0.0
    reward_scale_far_goal_stand = 0.0
    reward_scale_velo_dir = 4.0
    reward_scale_reach_pos_target_tight = 10.0
    reward_scale_ang_vel_xy = -0.05
    randomize_friction = True
    friction_range = (-0.2, 1.25)
    randomize_base_mass = True
    added_mass_range = (-1.5, 1.5)
    replay_len = 100
    enable_collision_replay = True
    collision_replay_prob = 0.8
    collision_replay_early_reset_prob_range = (0.1, 0.5)
    collision_replay_undo_steps_range = (100, 150)
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=10.0, replicate_physics=True)
    terrain: TerrainImporterCfg | None = None
    robot: ArticulationCfg | None = None
    contact_sensor: ContactSensorCfg | None = None


def build_go2_articulation_cfg(
    go2_usd_path: str | None,
    actuator_mode: str = "ideal_pd",
    robot_asset_source: str = "converted_urdf",
) -> ArticulationCfg:
    if robot_asset_source == "converted_urdf":
        if go2_usd_path is None:
            raise ValueError("go2_usd_path must be provided when robot_asset_source='converted_urdf'")
        usd_path = go2_usd_path
    elif robot_asset_source == "native_go2":
        from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

        usd_path = UNITREE_GO2_CFG.spawn.usd_path
    else:
        raise ValueError(f"Unsupported robot_asset_source: {robot_asset_source}")

    if actuator_mode == "implicit":
        actuator_cfg = ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit_sim=80.0,
            velocity_limit_sim=100.0,
            stiffness=30.0,
            damping=0.75,
        )
    elif actuator_mode == "ideal_pd":
        actuator_cfg = IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=80.0,
            velocity_limit=100.0,
            stiffness=30.0,
            damping=0.75,
        )
    elif actuator_mode == "gym_torque":
        actuator_cfg = IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=80.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=0.0,
        )
    elif actuator_mode == "robotlab_dc":
        actuator_cfg = DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        )
    else:
        raise ValueError(f"Unsupported actuator_mode: {actuator_mode}")
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.42), joint_pos=DEFAULT_JOINT_ANGLES),
        actuators={"legs": actuator_cfg},
        soft_joint_pos_limit_factor=0.95,
    ).replace(prim_path="/World/envs/env_.*/Robot")


def make_sea_nav_env_cfg(
    go2_usd_path: str | None,
    num_envs: int = 256,
    seed: int = 1,
    actuator_mode: str = "ideal_pd",
    robot_asset_source: str = "converted_urdf",
    terrain_rows: int = 10,
    terrain_cols: int = 10,
    obstacle_level: int = 9,
    goal_stop_radius: float = -1.0,
    goal_stop_mode: str = "zero",
    episode_length_s: float | None = None,
    nav_action_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    turn_yaw_threshold: float = -1.0,
    turn_forward_floor_pos: float = -1.0,
    turn_forward_floor_neg: float = -1.0,
    low_level_controller: str = "sea_nav_jit",
    robotlab_policy_path: str = str(ROBOTLAB_LOW_LEVEL_POLICY_PATH),
    robotlab_command_clip: float = 1.0,
    reward_scale_termination: float = -100.0,
    reward_scale_collision: float = -4.0,
    reward_scale_close_obst_vel: float = 5.0,
    reward_scale_stuck: float = -5.0,
    reward_scale_progress: float = 0.0,
    reward_scale_far_goal_stand: float = 0.0,
    reward_scale_velo_dir: float = 4.0,
    reward_scale_reach_pos_target_tight: float = 10.0,
    reward_scale_ang_vel_xy: float = -0.05,
    randomize_friction: bool = True,
    friction_range: tuple[float, float] = (-0.2, 1.25),
    randomize_base_mass: bool = True,
    added_mass_range: tuple[float, float] = (-1.5, 1.5),
    terrain_difficulty_range: tuple[float, float] | None = None,
    preset_room: np.ndarray | None = None,
    preset_start_goal_cases: list[dict[str, object]] | None = None,
    preset_start_goal_case_prob: float = 1.0,
) -> SeaNavEnvCfg:
    cfg = SeaNavEnvCfg()
    terrain_side = cfg.room_size + 1.1 * cfg.room_resolution
    cfg.seed = seed
    cfg.actuator_mode = actuator_mode
    cfg.scene.num_envs = num_envs
    cfg.scene.env_spacing = terrain_side
    if episode_length_s is not None:
        cfg.episode_length_s = episode_length_s
    cfg.goal_stop_radius = goal_stop_radius
    cfg.goal_stop_mode = goal_stop_mode
    cfg.nav_action_scale = nav_action_scale
    cfg.turn_yaw_threshold = turn_yaw_threshold
    cfg.turn_forward_floor_pos = turn_forward_floor_pos
    cfg.turn_forward_floor_neg = turn_forward_floor_neg
    cfg.low_level_controller = low_level_controller
    cfg.robotlab_policy_path = robotlab_policy_path
    cfg.robotlab_command_clip = robotlab_command_clip
    cfg.reward_scale_termination = reward_scale_termination
    cfg.reward_scale_collision = reward_scale_collision
    cfg.reward_scale_close_obst_vel = reward_scale_close_obst_vel
    cfg.reward_scale_stuck = reward_scale_stuck
    cfg.reward_scale_progress = reward_scale_progress
    cfg.reward_scale_far_goal_stand = reward_scale_far_goal_stand
    cfg.reward_scale_velo_dir = reward_scale_velo_dir
    cfg.reward_scale_reach_pos_target_tight = reward_scale_reach_pos_target_tight
    cfg.reward_scale_ang_vel_xy = reward_scale_ang_vel_xy
    cfg.randomize_friction = randomize_friction
    cfg.friction_range = friction_range
    cfg.randomize_base_mass = randomize_base_mass
    cfg.added_mass_range = added_mass_range
    cfg.preset_start_goal_cases = preset_start_goal_cases
    cfg.preset_start_goal_case_prob = preset_start_goal_case_prob
    cfg.robot = build_go2_articulation_cfg(
        go2_usd_path,
        actuator_mode=actuator_mode,
        robot_asset_source=robot_asset_source,
    )
    cfg.contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=cfg.sim.dt,
        track_air_time=False,
    )
    cfg.terrain = TerrainImporterCfg(
        class_type=SeaNavTerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            class_type=SeaNavTerrainGenerator,
            seed=seed,
            curriculum=True,
            size=(terrain_side, terrain_side),
            border_width=25.0,
            border_height=1.0,
            num_rows=terrain_rows,
            num_cols=terrain_cols,
            horizontal_scale=cfg.room_resolution,
            vertical_scale=0.005,
            slope_threshold=0.75,
            difficulty_range=terrain_difficulty_range if terrain_difficulty_range is not None else (0.0, 1.0),
            use_cache=False,
            sub_terrains={
                "hard_room": SeaNavRoomTerrainCfg(
                    proportion=1.0,
                    obstacle_level=obstacle_level,
                    preset_room=preset_room,
                )
            },
        ),
        max_init_terrain_level=min(max(terrain_rows - 1, 0), 2),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
        debug_vis=False,
    )
    return cfg


class SeaNavIsaacLabEnv(DirectRLEnv):
    cfg: SeaNavEnvCfg

    def __init__(self, cfg: SeaNavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.num_nav_actions = 3
        self.num_props = cfg.num_props
        self.num_obs_one_step = cfg.num_props + cfg.num_rays + 2
        self.history_length = cfg.history_length
        self.num_obs = self.num_obs_one_step * self.history_length
        self.room_resolution = cfg.room_resolution
        self.room_cells = cfg.room_cells
        self.room_size = cfg.room_size
        self.c_x = int((3.2 / self.room_resolution))
        self.c_y = int((3.2 / self.room_resolution))
        self.commands_scale = torch.tensor([1.0, 1.0, 1.0], device=self.device)
        self.slr_commands_scale = torch.tensor([2.0, 2.0, 0.25], device=self.device)
        self.nav_action_scale = torch.tensor(self.cfg.nav_action_scale, dtype=torch.float, device=self.device)
        if self.cfg.low_level_controller == "robotlab":
            command_clip = float(self.cfg.robotlab_command_clip)
            self.nav_clip_min = torch.tensor([-command_clip, -command_clip, -command_clip], device=self.device)
            self.nav_clip_max = torch.tensor([command_clip, command_clip, command_clip], device=self.device)
        else:
            self.nav_clip_min = torch.tensor([-0.5, -1.0, -1.0], device=self.device)
            self.nav_clip_max = torch.tensor([2.0, 1.0, 1.0], device=self.device)
        self.ray_angles = torch.arange(-2 * math.pi / 3, 2 * math.pi / 3 + 1.0e-4, math.pi / 30, device=self.device)
        x = torch.arange(-3.2, 3.2 + 1.0e-6, self.room_resolution, device=self.device)
        y = torch.arange(-3.2, 3.2 + 1.0e-6, self.room_resolution, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        self.height_points = torch.stack((xx, yy), dim=-1).view(-1, 2)
        self.num_height_points = self.height_points.shape[0]
        self._build_joint_order_mappings()
        self._load_room_maps()
        self._load_low_level_models()
        self._allocate_buffers()
        self._apply_initial_domain_randomization()
        self._resolve_contact_bodies()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _load_room_maps(self):
        flat_rooms = getattr(self._terrain, "generated_rooms", [])
        num_rows = self.cfg.terrain.terrain_generator.num_rows
        num_cols = self.cfg.terrain.terrain_generator.num_cols
        if len(flat_rooms) != num_rows * num_cols:
            raise RuntimeError("Generated room cache does not match terrain grid size.")
        rooms_np = np.stack(flat_rooms)
        rooms_np = rooms_np.reshape(num_cols, num_rows, self.room_cells, self.room_cells).transpose(1, 0, 2, 3)
        self.room_maps = torch.tensor(rooms_np, device=self.device, dtype=torch.float)

    def _load_low_level_models(self):
        map_location = torch.device(self.device)
        self.slr_body = None
        self.slr_encoder_vel = None
        self.slr_encoder_latent = None
        self.robotlab_policy = None
        if self.cfg.low_level_controller == "sea_nav_jit":
            self.slr_body = torch.jit.load(str(CTRL_MODEL_DIR / "body_latest.jit"), map_location=map_location).eval()
            self.slr_encoder_vel = torch.jit.load(
                str(CTRL_MODEL_DIR / "encoder_vel.jit"), map_location=map_location
            ).eval()
            self.slr_encoder_latent = torch.jit.load(
                str(CTRL_MODEL_DIR / "encoder_latent.jit"), map_location=map_location
            ).eval()
        elif self.cfg.low_level_controller == "robotlab":
            policy_path = Path(self.cfg.robotlab_policy_path)
            if not policy_path.is_file():
                raise FileNotFoundError(f"RobotLab low-level policy not found: {policy_path}")
            self.robotlab_policy = torch.jit.load(str(policy_path), map_location=map_location).eval()
            print(f"[INFO] robotlab low-level policy loaded path={policy_path}")
        else:
            raise ValueError(f"Unsupported low_level_controller: {self.cfg.low_level_controller}")

    def _build_joint_order_mappings(self):
        joint_names = list(self._robot.joint_names)
        if len(joint_names) != 12:
            raise RuntimeError(f"Expected 12 robot joints, got {len(joint_names)}: {joint_names}")

        live_joint_to_id = {name: idx for idx, name in enumerate(joint_names)}
        missing_names = [name for name in LOW_LEVEL_JOINT_ORDER if name not in live_joint_to_id]
        if missing_names:
            raise RuntimeError(
                f"Live articulation order is missing low-level joints {missing_names}. Live joints: {joint_names}"
            )

        self.live_joint_names = joint_names
        self.low_level_joint_names = list(LOW_LEVEL_JOINT_ORDER)
        self.live_to_low_level_ids = torch.tensor(
            [live_joint_to_id[name] for name in self.low_level_joint_names],
            dtype=torch.long,
            device=self.device,
        )
        low_level_to_id = {name: idx for idx, name in enumerate(self.low_level_joint_names)}
        self.low_level_to_live_ids = torch.tensor(
            [low_level_to_id[name] for name in self.live_joint_names],
            dtype=torch.long,
            device=self.device,
        )
        print(
            "[INFO] low-level joint mapping "
            f"live_joint_names={self.live_joint_names} "
            f"low_level_joint_names={self.low_level_joint_names}"
        )

    def _allocate_buffers(self):
        n = self.num_envs
        self.replay_len = int(self.cfg.replay_len)
        self.rays = torch.ones(n, self.cfg.num_rays, device=self.device) * 5.0
        self.last_rays = self.rays.clone()
        self.delay_rays = self.rays.clone()
        self.rays_hist = torch.ones(n, self.history_length, self.cfg.num_rays, device=self.device) * 5.0
        self.goal_hist = torch.zeros(n, self.history_length, 2, device=self.device)
        self.delay_goal = torch.zeros(n, 2, device=self.device)
        self.goal_local_pos = torch.zeros(n, 2, device=self.device)
        self.last_goal_local_pos = torch.zeros(n, 2, device=self.device)
        self.position_targets = torch.zeros(n, 3, device=self.device)
        self.distance = torch.zeros(n, device=self.device)
        self.last_distance = torch.zeros(n, device=self.device)
        self.prev_distance = torch.zeros(n, device=self.device)
        self.goal_levels = torch.zeros(n, dtype=torch.float, device=self.device)
        self.reach_goal = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.goal_hold_timer = torch.zeros(n, dtype=torch.long, device=self.device)
        self.stay_timer = torch.zeros(n, dtype=torch.long, device=self.device)
        self.goal_reached_flag = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.stand_still_flag = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.nav_actions_orig = torch.zeros(n, 3, device=self.device)
        self.nav_actions_filtered = torch.zeros(n, 3, device=self.device)
        self.slr_commands = torch.zeros(n, 3, device=self.device)
        self.slr_obs_buf = torch.zeros(n, 45, device=self.device)
        self.slr_obs_hist = torch.zeros(n, self.history_length, 45, device=self.device)
        self.obs_history_buf = torch.zeros(n, self.history_length, self.num_obs_one_step, device=self.device)
        self.pos_hist = torch.zeros(n, self.history_length, 2, device=self.device)
        self.prop_buf = torch.zeros(n, self.cfg.num_props, device=self.device)
        self.actions_orig = torch.zeros(n, 12, device=self.device)
        self._joint_targets = self._robot.data.default_joint_pos.clone()
        self._joint_efforts = torch.zeros_like(self._joint_targets)
        self.bad_masks = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.is_replay = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.collision_occurred = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.last_collision_active = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.replay_root_states = torch.zeros(n, self.replay_len, 13, device=self.device)
        self.replay_joint_pos = torch.zeros(n, self.replay_len, self._robot.num_joints, device=self.device)
        self.replay_joint_vel = torch.zeros(n, self.replay_len, self._robot.num_joints, device=self.device)
        self.measured_heights = torch.zeros(n, 65, 65, device=self.device)
        self.guide_ray_idx = torch.zeros(n, dtype=torch.long, device=self.device)
        self._terminated = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.last_done_contact = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.last_done_goal_hold = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.last_done_stand = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.last_done_fall = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.last_done_timeout = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.last_static = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.last_v_low = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.last_d_low = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.last_root_pos_w = torch.zeros(n, 3, device=self.device)
        self.last_root_quat_w = torch.zeros(n, 4, device=self.device)
        self.last_reward_terms = {
            key: torch.zeros(n, dtype=torch.float, device=self.device)
            for key in [
                "collision",
                "close_obst_vel",
                "stuck",
                "progress",
                "far_goal_stand",
                "velo_dir",
                "reach_pos_target_tight",
                "ang_vel_xy",
                "termination",
            ]
        }
        self.episode_sums = {
            key: torch.zeros(n, dtype=torch.float, device=self.device)
            for key in [
                "collision",
                "close_obst_vel",
                "stuck",
                "progress",
                "far_goal_stand",
                "velo_dir",
                "reach_pos_target_tight",
                "ang_vel_xy",
                "termination",
            ]
        }
        self._default_joint_pos_low_level = self._live_to_low_level(self._robot.data.default_joint_pos)
        if self.cfg.low_level_controller == "robotlab":
            scales = [0.125 if "hip" in name else 0.25 for name in self.low_level_joint_names]
        else:
            scales = [self.cfg.joint_action_scale for _ in self.low_level_joint_names]
        self._joint_action_scales_low_level = torch.tensor(scales, dtype=torch.float, device=self.device).unsqueeze(0)
        self._gym_torque_p_gains = torch.full(
            (1, self._robot.num_joints), self.cfg.joint_stiffness, dtype=torch.float, device=self.device
        )
        self._gym_torque_d_gains = torch.full(
            (1, self._robot.num_joints), self.cfg.joint_damping, dtype=torch.float, device=self.device
        )
        sim_torque_limits_live = self._robot.root_physx_view.get_dof_max_forces()[0].to(self.device)
        fallback_torque_limits = torch.tensor(
            [
                GO2_URDF_EFFORT_LIMITS["hip"]
                if "hip" in name
                else GO2_URDF_EFFORT_LIMITS["thigh"]
                if "thigh" in name
                else GO2_URDF_EFFORT_LIMITS["calf"]
                for name in self.live_joint_names
            ],
            dtype=torch.float,
            device=self.device,
        )
        valid_torque_limits = sim_torque_limits_live > 0.0
        clipped_torque_limits_live = torch.minimum(sim_torque_limits_live, fallback_torque_limits)
        clipped_torque_limits_live = torch.where(valid_torque_limits, clipped_torque_limits_live, fallback_torque_limits)
        self._gym_torque_limits_low_level = self._live_to_low_level(clipped_torque_limits_live.unsqueeze(0))
        if self.cfg.actuator_mode == "gym_torque":
            print(
                "[INFO] gym_torque config "
                f"joint_action_scale={self.cfg.joint_action_scale} "
                f"joint_stiffness={self.cfg.joint_stiffness} "
                f"joint_damping={self.cfg.joint_damping} "
                f"torque_limit_min={self._gym_torque_limits_low_level.min().item():.3f} "
                f"torque_limit_max={self._gym_torque_limits_low_level.max().item():.3f}"
            )

    def _apply_initial_domain_randomization(self):
        env_ids_cpu = torch.arange(self.num_envs, dtype=torch.long, device="cpu")
        self.friction_coeffs = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self.added_base_mass = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        if self.cfg.randomize_friction:
            materials = self._robot.root_physx_view.get_material_properties()
            num_shapes = self._robot.root_physx_view.max_shapes
            num_buckets = 64
            friction_lo, friction_hi = self.cfg.friction_range
            friction_lo = max(0.0, float(friction_lo))
            friction_hi = max(friction_lo, float(friction_hi))
            friction_buckets = torch.empty((num_buckets, 1), dtype=torch.float, device="cpu").uniform_(
                friction_lo, friction_hi
            )
            bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1), device="cpu")
            sampled_friction = friction_buckets[bucket_ids].squeeze(-1).expand(-1, num_shapes)
            materials[env_ids_cpu, :, 0] = sampled_friction
            materials[env_ids_cpu, :, 1] = sampled_friction
            self._robot.root_physx_view.set_material_properties(materials, env_ids_cpu)
            self.friction_coeffs = friction_buckets[bucket_ids].view(self.num_envs).to(self.device)

        if self.cfg.randomize_base_mass:
            masses = self._robot.root_physx_view.get_masses()
            inertias = self._robot.root_physx_view.get_inertias()
            default_mass = self._robot.data.default_mass.detach().cpu().clone()
            default_inertia = self._robot.data.default_inertia.detach().cpu().clone()
            added_mass_lo, added_mass_hi = self.cfg.added_mass_range
            added_mass = torch.empty((self.num_envs,), dtype=torch.float, device="cpu").uniform_(
                added_mass_lo, added_mass_hi
            )
            masses[env_ids_cpu, 0] = torch.clamp(default_mass[env_ids_cpu, 0] + added_mass, min=1.0e-6)
            self._robot.root_physx_view.set_masses(masses, env_ids_cpu)
            mass_ratio = masses[env_ids_cpu, 0] / default_mass[env_ids_cpu, 0]
            inertias[env_ids_cpu, 0] = default_inertia[env_ids_cpu, 0] * mass_ratio.unsqueeze(-1)
            self._robot.root_physx_view.set_inertias(inertias, env_ids_cpu)
            self.added_base_mass = added_mass.to(self.device)

        print(
            "[INFO] domain randomization "
            f"randomize_friction={self.cfg.randomize_friction} "
            f"friction_min={self.friction_coeffs.min().item():.3f} "
            f"friction_max={self.friction_coeffs.max().item():.3f} "
            f"randomize_base_mass={self.cfg.randomize_base_mass} "
            f"added_mass_min={self.added_base_mass.min().item():.3f} "
            f"added_mass_max={self.added_base_mass.max().item():.3f}"
        )

    def _update_replay_buffer(self):
        root_state = self._robot.data.root_state_w
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        self.replay_root_states = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([root_state] * self.replay_len, dim=1),
            torch.cat((self.replay_root_states[:, 1:], root_state.unsqueeze(1)), dim=1),
        )
        self.replay_joint_pos = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([joint_pos] * self.replay_len, dim=1),
            torch.cat((self.replay_joint_pos[:, 1:], joint_pos.unsqueeze(1)), dim=1),
        )
        self.replay_joint_vel = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([joint_vel] * self.replay_len, dim=1),
            torch.cat((self.replay_joint_vel[:, 1:], joint_vel.unsqueeze(1)), dim=1),
        )

    def _resolve_contact_bodies(self):
        self._termination_body_ids = self._find_body_ids(list(self.cfg.termination_body_patterns))
        self._penalized_body_ids = self._find_body_ids(list(self.cfg.penalized_body_patterns))
        self._head_body_ids = self._find_body_ids(["base", "Head_upper", "Head_lower"])
        self._leg_body_ids = self._find_body_ids([".*_thigh", ".*_calf"])

    def _find_body_ids(self, patterns: list[str]) -> torch.Tensor:
        if not patterns:
            return torch.empty(0, dtype=torch.long, device=self.device)
        ids = []
        for pattern in patterns:
            found_ids, _ = self._contact_sensor.find_bodies(pattern)
            if isinstance(found_ids, torch.Tensor):
                ids.extend(found_ids.tolist())
            else:
                ids.extend(list(found_ids))
        if not ids:
            raise RuntimeError(f"No bodies matched patterns: {patterns}")
        return torch.tensor(sorted(set(ids)), dtype=torch.long, device=self.device)

    def _live_to_low_level(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, self.live_to_low_level_ids]

    def _low_level_to_live(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, self.low_level_to_live_ids]

    def _current_room_maps(self) -> torch.Tensor:
        return self.room_maps[self._terrain.terrain_levels, self._terrain.terrain_types]

    def _room_local_xy(self) -> tuple[torch.Tensor, torch.Tensor]:
        local_xy = self._robot.data.root_pos_w[:, :2] - self._terrain.env_origins[:, :2] + self.room_size * 0.5
        return local_xy[:, 0], local_xy[:, 1]

    def _pre_physics_step(self, actions: torch.Tensor):
        self.nav_actions_orig = torch.clamp(actions, -3.0, 3.0)
        if self.cfg.turn_yaw_threshold >= 0.0:
            turn_mask = torch.abs(self.nav_actions_orig[:, 2]) >= self.cfg.turn_yaw_threshold
            pos_mask = turn_mask & (self.nav_actions_orig[:, 2] >= 0.0)
            neg_mask = turn_mask & (self.nav_actions_orig[:, 2] < 0.0)
            if self.cfg.turn_forward_floor_pos >= 0.0:
                pos_mask &= self.nav_actions_orig[:, 0] < self.cfg.turn_forward_floor_pos
                self.nav_actions_orig[pos_mask, 0] = self.cfg.turn_forward_floor_pos
            if self.cfg.turn_forward_floor_neg >= 0.0:
                neg_mask &= self.nav_actions_orig[:, 0] < self.cfg.turn_forward_floor_neg
                self.nav_actions_orig[neg_mask, 0] = self.cfg.turn_forward_floor_neg
        if self.cfg.goal_stop_radius >= 0.0:
            stop_mask = self.distance <= self.cfg.goal_stop_radius
            if self.cfg.goal_stop_mode == "zero":
                self.nav_actions_orig[stop_mask] = 0.0
            elif self.cfg.goal_stop_mode == "linear":
                scales = (self.distance[stop_mask] / self.cfg.goal_stop_radius).clamp_(0.0, 1.0).unsqueeze(1)
                self.nav_actions_orig[stop_mask] *= scales
            else:
                raise ValueError(f"Unsupported goal_stop_mode: {self.cfg.goal_stop_mode}")
        self.nav_actions_orig *= self.nav_action_scale
        self.nav_actions_filtered = 0.5 * self.nav_actions_orig + 0.5 * self.nav_actions_filtered
        self.slr_commands = torch.clamp(self.nav_actions_filtered, self.nav_clip_min, self.nav_clip_max)
        low_level_actions = self._compute_low_level_actions(self.slr_commands)
        self.actions_orig = torch.clamp(low_level_actions, -100.0, 100.0)
        scaled_low_level_actions = self.actions_orig * self._joint_action_scales_low_level
        self._joint_targets = self._robot.data.default_joint_pos + self._low_level_to_live(scaled_low_level_actions)

    def _apply_action(self):
        if self.cfg.actuator_mode == "gym_torque":
            self._joint_efforts = self._compute_gym_torques()
            self._robot.set_joint_effort_target(self._joint_efforts)
        else:
            self._robot.set_joint_position_target(self._joint_targets)

    def _compute_gym_torques(self) -> torch.Tensor:
        actions_scaled = self.actions_orig * self._joint_action_scales_low_level
        joint_pos_target = actions_scaled + self._default_joint_pos_low_level
        joint_pos = self._live_to_low_level(self._robot.data.joint_pos)
        joint_vel = self._live_to_low_level(self._robot.data.joint_vel)
        torques = self._gym_torque_p_gains * (joint_pos_target - joint_pos) - self._gym_torque_d_gains * joint_vel
        torques = torch.clamp(torques, -self._gym_torque_limits_low_level, self._gym_torque_limits_low_level)
        return self._low_level_to_live(torques)

    def _compute_low_level_actions(self, nav_actions: torch.Tensor) -> torch.Tensor:
        if self.cfg.low_level_controller == "robotlab":
            return self._compute_robotlab_low_level_actions(nav_actions)
        return self._compute_sea_nav_jit_low_level_actions(nav_actions)

    def _compute_robotlab_low_level_actions(self, nav_actions: torch.Tensor) -> torch.Tensor:
        scale_ang_vel = 0.25
        scale_dof_vel = 0.05
        command_clip = float(self.cfg.robotlab_command_clip)
        commands = torch.clamp(nav_actions, -command_clip, command_clip)
        self.slr_obs_buf = torch.cat(
            (
                self._robot.data.root_ang_vel_b * scale_ang_vel,
                self._robot.data.projected_gravity_b,
                commands,
                self._live_to_low_level(self._robot.data.joint_pos - self._robot.data.default_joint_pos),
                self._live_to_low_level(self._robot.data.joint_vel) * scale_dof_vel,
                self.actions_orig,
            ),
            dim=-1,
        )
        with torch.no_grad():
            return self.robotlab_policy(self.slr_obs_buf)

    def _compute_sea_nav_jit_low_level_actions(self, nav_actions: torch.Tensor) -> torch.Tensor:
        scale_ang_vel = 0.25
        scale_dof_vel = 0.05
        self.slr_obs_buf = torch.cat(
            (
                self._robot.data.root_ang_vel_b * scale_ang_vel,
                self._robot.data.projected_gravity_b,
                nav_actions * self.slr_commands_scale,
                self._live_to_low_level(self._robot.data.joint_pos - self._robot.data.default_joint_pos),
                self._live_to_low_level(self._robot.data.joint_vel * scale_dof_vel),
                self.actions_orig,
            ),
            dim=-1,
        )
        if self.cfg.add_noise:
            noise_vec = torch.cat(
                (
                    torch.ones(3, device=self.device) * self.cfg.slr_noise_ang_vel,
                    torch.ones(3, device=self.device) * self.cfg.slr_noise_gravity,
                    torch.zeros(3, device=self.device),
                    torch.ones(12, device=self.device) * self.cfg.slr_noise_dof_pos,
                    torch.ones(12, device=self.device) * self.cfg.slr_noise_dof_vel * scale_dof_vel,
                    torch.zeros(12, device=self.device),
                ),
                dim=0,
            )
            self.slr_obs_buf += (2.0 * torch.rand_like(self.slr_obs_buf) - 1.0) * 0.5 * noise_vec
        self.slr_obs_hist = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([self.slr_obs_buf] * self.history_length, dim=1),
            torch.cat((self.slr_obs_hist[:, 1:], self.slr_obs_buf.unsqueeze(1)), dim=1),
        )
        base_lin_vel_pred = self.slr_encoder_vel(self.slr_obs_hist.reshape(self.num_envs, -1))
        latent = self.slr_encoder_latent(self.slr_obs_hist.reshape(self.num_envs, -1))
        actor_obs = torch.cat(
            (base_lin_vel_pred, self.slr_obs_buf, self._robot.data.root_ang_vel_b[:, 2:] * scale_ang_vel, latent),
            dim=-1,
        )
        return self.slr_body(actor_obs)

    def _sample_measured_heights(self):
        room_maps = self._current_room_maps()
        root_x, root_y = self._room_local_xy()
        quat = self._robot.data.root_quat_w
        yaw = torch.atan2(
            2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1.0 - 2.0 * (quat[:, 2] * quat[:, 2] + quat[:, 3] * quat[:, 3]),
        )
        cos_yaw = torch.cos(yaw).unsqueeze(1)
        sin_yaw = torch.sin(yaw).unsqueeze(1)
        px = self.height_points[:, 0].unsqueeze(0)
        py = self.height_points[:, 1].unsqueeze(0)
        local_x = root_x.unsqueeze(1) + cos_yaw * px - sin_yaw * py
        local_y = root_y.unsqueeze(1) + sin_yaw * px + cos_yaw * py
        gx = torch.round(local_x / self.room_resolution).long()
        gy = torch.round(local_y / self.room_resolution).long()
        valid = (gx >= 0) & (gx < self.room_cells) & (gy >= 0) & (gy < self.room_cells)
        gx_clip = gx.clamp(0, self.room_cells - 1)
        gy_clip = gy.clamp(0, self.room_cells - 1)
        env_ids = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, self.num_height_points)
        heights = torch.zeros(self.num_envs, self.num_height_points, device=self.device)
        heights[valid] = room_maps[env_ids[valid], gx_clip[valid], gy_clip[valid]]
        self.measured_heights = heights.view(self.num_envs, 65, 65)

    def _update_navigation_state(self):
        self.distance = torch.norm(self.position_targets[:, :2] - self._robot.data.root_pos_w[:, :2], dim=1)
        self.reach_goal = self.distance < 0.5
        self._sample_measured_heights()
        center_height = self.measured_heights[:, self.c_x, self.c_y].unsqueeze(1).unsqueeze(2)
        raw_heights = (self.measured_heights > (center_height + 0.1)).to(dtype=torch.int32)
        self.rays = batch_ray_cast_torch(
            raw_heights,
            base_row=self.c_x,
            base_col=self.c_y,
            angles=self.ray_angles,
            max_radius=3.0 / self.room_resolution,
            step_r=0.1,
        ) * self.room_resolution
        self.rays_hist = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([self.rays] * self.history_length, dim=1),
            torch.cat((self.rays_hist[:, 1:], self.rays.unsqueeze(1)), dim=1),
        )
        delta_xy = self.position_targets[:, :2] - self._robot.data.root_pos_w[:, :2]
        quat = self._robot.data.root_quat_w
        yaw = torch.atan2(
            2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1.0 - 2.0 * (quat[:, 2] * quat[:, 2] + quat[:, 3] * quat[:, 3]),
        )
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        self.goal_local_pos[:, 0] = cos_yaw * delta_xy[:, 0] + sin_yaw * delta_xy[:, 1]
        self.goal_local_pos[:, 1] = -sin_yaw * delta_xy[:, 0] + cos_yaw * delta_xy[:, 1]
        self.goal_hist = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([self.goal_local_pos] * self.history_length, dim=1),
            torch.cat((self.goal_hist[:, 1:], self.goal_local_pos.unsqueeze(1)), dim=1),
        )
        delay_steps = max(1, int(self.cfg.command_delay_s / self.step_dt))
        env_ids = (self.episode_length_buf % delay_steps == 0).nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            time_idx = -torch.randint(2, 4, (len(env_ids),), device=self.device) - 1
            self.delay_rays[env_ids] = self.rays_hist[env_ids, time_idx, :]
            self.delay_goal[env_ids] = self.goal_hist[env_ids, time_idx, :]
        env_ids = (self.episode_length_buf % 10 == 0).nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            root_xy = self._robot.data.root_pos_w[env_ids, :2]
            self.pos_hist[env_ids] = torch.where(
                (self.episode_length_buf[env_ids] <= 1)[:, None, None],
                torch.stack([root_xy] * self.history_length, dim=1),
                torch.cat((self.pos_hist[env_ids, 1:], root_xy.unsqueeze(1)), dim=1),
            )

    def _get_observations(self) -> dict:
        self._update_replay_buffer()
        self._update_navigation_state()
        self.prop_buf = torch.cat(
            (
                self._robot.data.projected_gravity_b,
                self.slr_commands * self.commands_scale,
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
            ),
            dim=-1,
        )
        if self.cfg.add_noise:
            prop_noise = torch.cat(
                (
                    torch.ones(3, device=self.device) * self.cfg.nav_noise_gravity,
                    torch.zeros(3, device=self.device),
                    torch.ones(3, device=self.device) * self.cfg.nav_noise_lin_vel,
                    torch.ones(3, device=self.device) * self.cfg.nav_noise_ang_vel,
                ),
                dim=0,
            )
            self.prop_buf += (2.0 * torch.rand_like(self.prop_buf) - 1.0) * prop_noise
        obs_buf = torch.cat(
            (
                self.prop_buf,
                torch.log2(self.delay_rays.clamp(min=0.1, max=5.0)),
                self.delay_goal,
            ),
            dim=-1,
        )
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([obs_buf] * self.history_length, dim=1),
            torch.cat((self.obs_history_buf[:, 1:], obs_buf.unsqueeze(1)), dim=1),
        )
        obs = self.obs_history_buf.reshape(self.num_envs, -1)
        return {"policy": obs}

    def _get_clearance(self, fov_deg: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if fov_deg is None:
            subset = self.rays
        else:
            threshold = math.radians(fov_deg / 2.0)
            subset = self.rays[:, torch.abs(self.ray_angles) <= threshold]
        return torch.min(subset, dim=-1).values, torch.max(subset, dim=-1).values

    def _get_guidance_nav_alignment(self, fov_deg: float | None = None) -> torch.Tensor:
        rays_clipped = torch.clamp(self.rays, max=2.0)
        rays_padded = F.pad(rays_clipped.unsqueeze(1), (2, 2), mode="replicate")
        smoothed_rays = F.avg_pool1d(rays_padded, 5, stride=1).squeeze(1)
        if fov_deg is not None:
            threshold = math.radians(fov_deg / 2.0)
            mask = torch.abs(self.ray_angles) <= threshold
            smoothed_rays = torch.where(mask, smoothed_rays, torch.full_like(smoothed_rays, -1.0))
        scores = smoothed_rays - torch.abs(self.ray_angles) * 0.001
        self.guide_ray_idx = torch.argmax(scores, dim=-1)
        return torch.cos(self.ray_angles[self.guide_ray_idx]).clamp(min=0.0)

    def _reward_reach_pos_target_tight(self) -> torch.Tensor:
        return (1.0 / (1.0 + 2.0 * torch.square(self.distance))) * (self.distance < 0.5)

    def _reward_velo_dir(self) -> torch.Tensor:
        goal_dir = self.goal_local_pos / (self.distance.unsqueeze(1) + 1.0e-4)
        alignment = goal_dir[:, 0].clamp(min=0.0)
        target_speed = torch.clamp(self.distance, max=0.5)
        forward_vel = torch.clamp(self._robot.data.root_lin_vel_b[:, 0], min=0.0)
        vel_reward = alignment * forward_vel
        vel_reward = torch.clamp(vel_reward, max=target_speed)
        reach_bonus = 1.0 / (1.0 + 2.0 * torch.square(self.distance))
        return vel_reward + reach_bonus

    def _reward_close_obst_vel(self) -> torch.Tensor:
        front_clearance, _ = self._get_clearance()
        dir_alignment = self._get_guidance_nav_alignment(fov_deg=150.0)
        x_vel = self._robot.data.root_lin_vel_b[:, 0].clamp(min=0.0)
        safe_vel_limit = torch.clamp(front_clearance, max=0.5)
        reward_vel = torch.min(x_vel, safe_vel_limit)
        reward_base = dir_alignment * reward_vel
        overspeed = (x_vel - safe_vel_limit).clamp(min=0.0)
        reward = (reward_base - 0.2 * overspeed).clamp(min=0.0)
        reach_bonus = 1.0 / (1.0 + 2.0 * torch.square(self.distance))
        far_goal = self.distance > 0.5
        return reward * far_goal + reach_bonus * (~far_goal)

    def _reward_stuck(self) -> torch.Tensor:
        distances_hist = torch.norm(self.pos_hist - self._robot.data.root_pos_w[:, None, :2], dim=-1)
        move_dist_max = torch.max(distances_hist, dim=-1).values
        stuck = move_dist_max < 0.1
        stand_velo = (
            torch.abs(self._robot.data.root_ang_vel_b[:, 2])
            + torch.abs(self._robot.data.root_lin_vel_b[:, 1].clamp(max=0.5))
            + torch.abs(self._robot.data.root_lin_vel_b[:, 0].clamp(max=0.5))
        )
        _, max_front_space = self._get_clearance(fov_deg=120.0)
        is_dead_end = max_front_space < 1.0
        no_backward = self._robot.data.root_lin_vel_b[:, 0] > 0.0
        no_turn_back = torch.abs(self._robot.data.root_ang_vel_b[:, 2]) < 1.0
        far_goal = self.distance > 0.5
        not_just_reset = (self.episode_length_buf / self.max_episode_length) > 0.1
        no_escape = is_dead_end & (no_backward | no_turn_back)
        return not_just_reset * far_goal * (no_escape | is_dead_end) * stuck + stand_velo * (~far_goal)

    def _reward_progress(self) -> torch.Tensor:
        valid_prev = self.prev_distance > 0.0
        far_goal = self.distance > 0.5
        progress = (self.prev_distance - self.distance).clamp(min=-0.05, max=0.05)
        return progress * valid_prev * far_goal

    def _reward_far_goal_stand(self) -> torch.Tensor:
        far_goal = self.distance > 0.5
        not_just_reset = (self.episode_length_buf / self.max_episode_length) > 0.1
        lin_speed = torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=-1)
        yaw_speed = torch.abs(self._robot.data.root_ang_vel_b[:, 2])
        stand_deficit = (0.12 - lin_speed).clamp(min=0.0) + 0.5 * (0.12 - yaw_speed).clamp(min=0.0)
        return far_goal * not_just_reset * stand_deficit

    def _reward_collision(self) -> torch.Tensor:
        contact_forces = self._contact_sensor.data.net_forces_w_history[:, :, self._penalized_body_ids]
        mask = ~(self.episode_length_buf <= 1)
        rew_coll = torch.sum(torch.norm(contact_forces[:, :, :, :2], dim=-1) > 0.1, dim=(1, 2)).float() * mask
        head_forces = self._contact_sensor.data.net_forces_w_history[:, :, self._head_body_ids]
        leg_forces = self._contact_sensor.data.net_forces_w_history[:, :, self._leg_body_ids]
        rew_head = torch.sum(torch.norm(head_forces[:, :, :, :2], dim=-1) > 0.1, dim=(1, 2)).float() * mask
        rew_leg = torch.sum(torch.norm(leg_forces[:, :, :, :2], dim=-1) > 0.1, dim=(1, 2)).float() * mask
        vel_square = torch.square(self._robot.data.root_lin_vel_b[:, :2]).sum(dim=-1) + torch.square(
            self._robot.data.root_ang_vel_b[:, 2]
        )
        total_coll = rew_coll + 10.0 * rew_head + 10.0 * rew_leg
        return (1.0 + 4.0 * vel_square) * total_coll

    def _reward_ang_vel_xy(self) -> torch.Tensor:
        return torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)

    def _get_rewards(self) -> torch.Tensor:
        self.last_rays.copy_(self.rays)
        self.last_distance.copy_(self.distance)
        self.last_goal_local_pos.copy_(self.goal_local_pos)
        self.last_root_pos_w.copy_(self._robot.data.root_pos_w)
        self.last_root_quat_w.copy_(self._robot.data.root_quat_w)
        termination = self._terminated.float()
        rewards = {
            "termination": self.cfg.reward_scale_termination * termination,
            "collision": self.cfg.reward_scale_collision * self._reward_collision(),
            "close_obst_vel": self.cfg.reward_scale_close_obst_vel * self._reward_close_obst_vel(),
            "stuck": self.cfg.reward_scale_stuck * self._reward_stuck(),
            "progress": self.cfg.reward_scale_progress * self._reward_progress(),
            "far_goal_stand": self.cfg.reward_scale_far_goal_stand * self._reward_far_goal_stand(),
            "velo_dir": self.cfg.reward_scale_velo_dir * self._reward_velo_dir(),
            "reach_pos_target_tight": self.cfg.reward_scale_reach_pos_target_tight
            * self._reward_reach_pos_target_tight(),
            "ang_vel_xy": self.cfg.reward_scale_ang_vel_xy * self._reward_ang_vel_xy(),
        }
        reward = torch.zeros(self.num_envs, device=self.device)
        for key, value in rewards.items():
            reward += value
            self.episode_sums[key] += value
            self.last_reward_terms[key].copy_(value)
        self.prev_distance.copy_(self.distance)
        return reward

    def _compute_early_replay_prob(self) -> torch.Tensor:
        prob_min, prob_max = self.cfg.collision_replay_early_reset_prob_range
        level_scale = (self.goal_levels / 1.5).clamp(0.0, 1.0)
        return prob_min + (prob_max - prob_min) * level_scale.clamp(0.0, 1.0)

    def _update_terrain_curriculum(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        if not hasattr(self._terrain, "terrain_origins") or self._terrain.terrain_origins is None:
            return
        move_up = self.distance[env_ids] < self.cfg.position_target_sigma_tight
        move_down = self.distance[env_ids] > self.cfg.position_target_sigma_soft
        max_level = float(getattr(self._terrain, "max_terrain_level", 0))
        self.goal_levels[env_ids] += move_up.float() - move_down.float()
        self.goal_levels[env_ids] = self.goal_levels[env_ids].clip(min=0.0, max=max_level)
        self._terrain.update_env_origins(env_ids, move_up, move_down)

    def _reset_collision_replay(
        self, env_ids: torch.Tensor, episode_lengths_before_reset: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(env_ids) == 0:
            return env_ids, env_ids

        undo_lo, undo_hi = self.cfg.collision_replay_undo_steps_range
        undo_steps = torch.randint(undo_lo, undo_hi, (len(env_ids),), device=self.device)
        undo_steps = torch.minimum(undo_steps.long(), episode_lengths_before_reset[env_ids].long())
        undo_steps = torch.minimum(undo_steps, torch.full_like(undo_steps, self.replay_len - 1))

        valid_replay = undo_steps > 20
        replay_ids = env_ids[valid_replay]
        fallback_ids = env_ids[~valid_replay]
        if len(replay_ids) == 0:
            return replay_ids, fallback_ids

        self.is_replay[replay_ids] = True
        history_idx = (self.replay_len - undo_steps[valid_replay]).long()
        root_state = self.replay_root_states[replay_ids, history_idx]
        joint_pos = self.replay_joint_pos[replay_ids, history_idx]
        joint_vel = self.replay_joint_vel[replay_ids, history_idx]
        self._robot.write_root_pose_to_sim(root_state[:, :7], replay_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], replay_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, replay_ids)
        return replay_ids, fallback_ids

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        forces = self._contact_sensor.data.net_forces_w_history
        if self.cfg.enable_contact_termination and self._termination_body_ids.numel() > 0:
            term_contacts = torch.any(
                torch.max(torch.norm(forces[:, :, self._termination_body_ids, :2], dim=-1), dim=1).values > 1.0, dim=1
            )
        else:
            term_contacts = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.bad_masks = self.episode_length_buf <= 1
        # Match the Isaac Gym path: expose the pre-reset initialization mask for this step.
        self.extras["bad_masks"] = self.bad_masks.clone()
        terminal_contacts = term_contacts & (~self.bad_masks)
        if self._penalized_body_ids.numel() > 0:
            new_collisions = torch.any(
                torch.max(torch.norm(forces[:, :, self._penalized_body_ids, :2], dim=-1), dim=1).values > 1.0, dim=1
            )
        else:
            new_collisions = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        new_collisions &= ~self.bad_masks
        is_new_collision = new_collisions & (~self.last_collision_active)
        if self.cfg.enable_collision_replay:
            early_prob = self._compute_early_replay_prob()
            early_replay_reset = is_new_collision & (torch.rand(self.num_envs, device=self.device) < early_prob)
        else:
            early_replay_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.collision_occurred |= new_collisions
        self.last_collision_active.copy_(new_collisions)
        self._terminated = terminal_contacts | early_replay_reset
        v_low = (torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=-1) < 0.1) & (
            torch.abs(self._robot.data.root_ang_vel_b[:, 2]) < 0.1
        )
        d_low = torch.norm(self._robot.data.root_pos_w[:, :2] - self.pos_hist[:, 0, :2], dim=-1) < 0.2
        static = (v_low | d_low) & ((self.episode_length_buf / self.max_episode_length) > 0.1)
        self.goal_hold_timer += self.reach_goal.int()
        self.stay_timer += static.int()
        self.goal_reached_flag = self.goal_hold_timer >= self.cfg.goal_reached_steps
        self.stand_still_flag = self.stay_timer >= self.cfg.stay_steps
        fall_down = self._robot.data.projected_gravity_b[:, 2] > -0.8
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self.last_done_contact.copy_(terminal_contacts)
        self.last_done_goal_hold.copy_(self.goal_reached_flag)
        self.last_done_stand.copy_(self.stand_still_flag)
        self.last_done_fall.copy_(fall_down)
        self.last_done_timeout.copy_(time_out)
        self.last_static.copy_(static)
        self.last_v_low.copy_(v_low)
        self.last_d_low.copy_(d_low)
        terminated = self._terminated | self.goal_reached_flag | self.stand_still_flag | fall_down
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        episode_lengths_before_reset = self.episode_length_buf.clone()
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        wants_replay = (
            self.cfg.enable_collision_replay
            & self.collision_occurred[env_ids]
            & (~self.goal_reached_flag[env_ids])
            & (~self.last_done_timeout[env_ids])
            & (torch.rand(len(env_ids), device=self.device) < self.cfg.collision_replay_prob)
        )
        replay_candidate_ids = env_ids[wants_replay]
        replay_ids, replay_fallback_ids = self._reset_collision_replay(replay_candidate_ids, episode_lengths_before_reset)
        normal_ids = env_ids[~wants_replay]
        if len(replay_fallback_ids) > 0:
            normal_ids = torch.cat((normal_ids, replay_fallback_ids))
        self._update_terrain_curriculum(normal_ids)

        self.actions_orig[env_ids] = 0.0
        self.nav_actions_orig[env_ids] = 0.0
        self.nav_actions_filtered[env_ids] = 0.0
        self.slr_commands[env_ids] = 0.0
        self.slr_obs_buf[env_ids] = 0.0
        self.slr_obs_hist[env_ids] = 0.0
        self.rays[env_ids] = 5.0
        self.last_rays[env_ids] = 5.0
        self.delay_rays[env_ids] = 5.0
        self.rays_hist[env_ids] = 5.0
        self.goal_hist[env_ids] = 0.0
        self.delay_goal[env_ids] = 0.0
        self.goal_local_pos[env_ids] = 0.0
        self.distance[env_ids] = 0.0
        self.prev_distance[env_ids] = 0.0
        self.reach_goal[env_ids] = False
        self.goal_hold_timer[env_ids] = 0
        self.stay_timer[env_ids] = 0
        self.goal_reached_flag[env_ids] = False
        self.stand_still_flag[env_ids] = False
        self.last_static[env_ids] = False
        self.last_v_low[env_ids] = False
        self.last_d_low[env_ids] = False
        self.obs_history_buf[env_ids] = 0.0
        self.pos_hist[env_ids] = 0.0
        self.prop_buf[env_ids] = 0.0
        self._terminated[env_ids] = False
        self.is_replay[normal_ids] = False
        self.collision_occurred[env_ids] = False
        self.last_collision_active[env_ids] = False
        self._joint_targets[env_ids] = self._robot.data.default_joint_pos[env_ids]
        if len(normal_ids) > 0:
            room_maps = self._current_room_maps()[normal_ids].cpu().numpy()
            joint_pos = self._robot.data.default_joint_pos[normal_ids]
            joint_vel = self._robot.data.default_joint_vel[normal_ids]
            root_state = self._robot.data.default_root_state[normal_ids].clone()
            env_origins = self._terrain.env_origins[normal_ids]
            self.position_targets[normal_ids] = 0.0
            preset_cases = self.cfg.preset_start_goal_cases
            case_indices = None
            if preset_cases:
                case_indices = torch.randint(len(preset_cases), (len(normal_ids),), device=self.device)
                use_case_mask = torch.rand(len(normal_ids), device=self.device) < float(self.cfg.preset_start_goal_case_prob)
            else:
                use_case_mask = None
            for i, env_id in enumerate(normal_ids.tolist()):
                if preset_cases and bool(use_case_mask[i].item()):
                    case = preset_cases[int(case_indices[i].item())]
                    robot_pos = case["start_cell"]
                    goal_pos = case["goal_cell"]
                    yaw = float(case["start_yaw"])
                else:
                    robot_pos, goal_pos = place_robot_and_goal(room_maps[i])
                    yaw = torch.empty(1, device=self.device).uniform_(-math.pi, math.pi).item()
                robot_xy = torch.tensor(
                    [
                        robot_pos[0] * self.room_resolution - self.room_size * 0.5,
                        robot_pos[1] * self.room_resolution - self.room_size * 0.5,
                    ],
                    device=self.device,
                )
                goal_xy = torch.tensor(
                    [
                        goal_pos[0] * self.room_resolution - self.room_size * 0.5,
                        goal_pos[1] * self.room_resolution - self.room_size * 0.5,
                    ],
                    device=self.device,
                )
                root_state[i, 0] = env_origins[i, 0] + robot_xy[0]
                root_state[i, 1] = env_origins[i, 1] + robot_xy[1]
                root_state[i, 2] = 0.42 + env_origins[i, 2]
                cy = math.cos(yaw * 0.5)
                sy = math.sin(yaw * 0.5)
                root_state[i, 3:7] = torch.tensor([cy, 0.0, 0.0, sy], device=self.device)
                root_state[i, 7:13] = torch.empty(6, device=self.device).uniform_(-0.5, 0.5)
                self.position_targets[env_id, 0] = env_origins[i, 0] + goal_xy[0]
                self.position_targets[env_id, 1] = env_origins[i, 1] + goal_xy[1]
                self.position_targets[env_id, 2] = env_origins[i, 2]
            self._robot.write_root_pose_to_sim(root_state[:, :7], normal_ids)
            self._robot.write_root_velocity_to_sim(root_state[:, 7:], normal_ids)
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, normal_ids)
        self.extras["episode"] = {}
        for key in self.episode_sums:
            self.extras["episode"][f"rew_{key}"] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.0
        self.extras["episode"]["goal_hold_success"] = torch.mean(self.last_done_goal_hold[env_ids].float())
        if hasattr(self._terrain, "terrain_levels"):
            self.extras["episode"]["terrain_level"] = torch.mean(self._terrain.terrain_levels.float())
            self.extras["episode"]["goal_level"] = torch.mean(self.goal_levels[env_ids])

    def set_manual_start_and_goal(
        self,
        env_ids: torch.Tensor,
        robot_xy_local: torch.Tensor,
        goal_xy_local: torch.Tensor,
        yaw: float = 0.0,
        root_height: float = 0.42,
    ):
        if env_ids.ndim == 0:
            env_ids = env_ids.unsqueeze(0)

        self.actions_orig[env_ids] = 0.0
        self.nav_actions_orig[env_ids] = 0.0
        self.nav_actions_filtered[env_ids] = 0.0
        self.slr_commands[env_ids] = 0.0
        self.slr_obs_buf[env_ids] = 0.0
        self.slr_obs_hist[env_ids] = 0.0
        self.is_replay[env_ids] = False
        self.collision_occurred[env_ids] = False
        self.last_collision_active[env_ids] = False
        self.rays[env_ids] = 5.0
        self.last_rays[env_ids] = 5.0
        self.delay_rays[env_ids] = 5.0
        self.rays_hist[env_ids] = 5.0
        self.goal_hist[env_ids] = 0.0
        self.delay_goal[env_ids] = 0.0
        self.goal_local_pos[env_ids] = 0.0
        self.distance[env_ids] = 0.0
        self.prev_distance[env_ids] = 0.0
        self.reach_goal[env_ids] = False
        self.goal_hold_timer[env_ids] = 0
        self.stay_timer[env_ids] = 0
        self.goal_reached_flag[env_ids] = False
        self.stand_still_flag[env_ids] = False
        self.last_static[env_ids] = False
        self.last_v_low[env_ids] = False
        self.last_d_low[env_ids] = False
        self.obs_history_buf[env_ids] = 0.0
        self.pos_hist[env_ids] = 0.0
        self.prop_buf[env_ids] = 0.0
        self._terminated[env_ids] = False
        self.episode_length_buf[env_ids] = 0
        for key in self.episode_sums:
            self.episode_sums[key][env_ids] = 0.0

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(self._robot.data.default_joint_vel[env_ids])
        root_state = self._robot.data.default_root_state[env_ids].clone()
        env_origins = self._terrain.env_origins[env_ids]

        root_state[:, 0:2] = env_origins[:, 0:2] + robot_xy_local
        root_state[:, 2] = env_origins[:, 2] + root_height
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        root_state[:, 3:7] = torch.tensor([cy, 0.0, 0.0, sy], device=self.device)
        root_state[:, 7:13] = 0.0

        self.position_targets[env_ids, 0:2] = env_origins[:, 0:2] + goal_xy_local
        self.position_targets[env_ids, 2] = env_origins[:, 2]
        self._joint_targets[env_ids] = self._robot.data.default_joint_pos[env_ids]

        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
