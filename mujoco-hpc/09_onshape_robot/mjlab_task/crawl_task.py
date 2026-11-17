"""mjlab task: teach the LEGO crawler (a motor box with two spinning "hand" arms) to crawl in a straight line down +x.

The robot starts at the origin pointing a little off to one side, with its arms at random angles. It must crawl
forward while staying on the line y = 0 and keeping its heading along +x. It outputs one spin speed per arm.

The policy (the robot's "brain") only sees what the REAL robot can measure, so it can run on it (../real_robot/):
its heading and turning speed (the Double Motor's motion sensor), where each arm is in its circle and how fast it
spins (the motor encoders), and its own last command. Those readings get a little noise and arrive a little late,
and the motor commands arrive late too, like over Bluetooth. The critic (the part that only helps during training)
also sees things only a simulation knows, such as the robot's speed and its distance from the line.

So that one brain works on any floor and for the real robot, every copy of the robot gets a slightly different
world: floor grip, robot weight and motor strength are random. The motor numbers come from probe.py on the real robot.

Importing this file registers the task "Crawl-Straight" (see train.py / play.py / make_video.py next to it).
Everything you will want to tune is in the constants block and the reward weights.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator import BuiltinPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  action_rate_l2,
  bad_orientation,
  base_ang_vel,
  base_lin_vel,
  dr,
  flat_orientation_l2,
  is_terminated,
  joint_vel_rel,
  last_action,
  projected_gravity,
  quat_apply,
  reset_joints_by_offset,
  reset_root_state_uniform,
  time_out,
)
from mjlab.envs.mdp.actions import JointVelocityActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.utils.spec_config import TextureCfg
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# ---- constants you may want to change ----------------------------------------------------------------
# the Onshape export: next to this file when run by runIt.py, one folder up in the repo
_HERE = Path(__file__).resolve().parent
ROBOT_XML = next(p for p in (_HERE / "onshape" / "robot.xml", _HERE.parent / "onshape" / "robot.xml") if p.exists())
LEFT, RIGHT = "left_wheel", "right_wheel"   # arm joint names in robot.xml

# The motors, matched to the real robot (../real_robot/probe.py and ../crawl.py).
MAX_ARM_SPEED = math.radians(1030)  # rad/s: the real motor at 100% (about 2.9 turns per second)
TOP_PERCENT = 50                    # the brain may use up to this % of the motor's speed (slower = calmer, no flips)
TOP_SPEED = MAX_ARM_SPEED * TOP_PERCENT / 100
# The brain's output a (-1..1) per arm means a FORWARD speed: -1 = stopped, +1 = TOP_SPEED. Arms never spin backwards
# (spinning them backwards is what flips the robot). It steers by turning one arm slower than the other.
# In the simulation "forward" is a negative arm speed, so: arm speed = ACTION_OFFSET + ACTION_SCALE * a.
ACTION_SCALE = -TOP_SPEED / 2
ACTION_OFFSET = -TOP_SPEED / 2
MOTOR_KD = 0.2                      # how hard a motor pushes toward its speed (N*m for each rad/s it is off)
MOTOR_TORQUE_LIMIT = 0.3            # N*m: the most a motor can push. With these the arms keep ~94% of their speed on
                                    # the floor, like the real robot (93%).

# Bluetooth: the real robot's readings come about every 20 ms (up to 60 ms); the laptop managed ~31 decisions a
# second, so the brain decides every 40 ms (25 times a second), in the simulation and on the real robot.
DECISION_S = 0.04
SENSOR_DELAY_STEPS = (0, 1)         # readings reach the brain 0 to 1 decisions (0-40 ms) late
COMMAND_DELAY_STEPS = (0, 20)       # commands reach the motors 0 to 20 physics steps (0-40 ms) late

# What we reward.
CRAWL_SPEED = 0.05      # m/s; crawling at this speed earns 1 point per step, faster up to twice this earns up to 2
LINE_SCALE = 0.05       # m; drifting this far sideways costs 1 point per step
HEADING_SCALE = 0.2     # rad (about 11 degrees); facing this far off +x costs 1 point per step

# The random world each robot copy gets (so the brain works on any floor and for the real robot).
FLOOR_GRIP = (0.3, 1.0)       # friction: 0.3 = slippery tile ... 1.0 = grippy carpet
ROBOT_WEIGHT_G = 102          # the real robot on a scale (the CAD says about 147 g, so every part is made lighter)
WEIGHT_SCALE = (0.9, 1.1)     # each copy weighs 90% to 110% of ROBOT_WEIGHT_G
MOTOR_SCALE = (0.7, 1.3)      # each motor is 70% to 130% as strong as MOTOR_TORQUE_LIMIT (the two can differ)

# Starts and ends of each try.
START_YAW = 0.3         # rad (about 17 degrees); the robot starts pointing up to this far off +x
MAX_OFF_LINE = 0.3      # m; the try ends if the robot wanders this far from the line
EPISODE_S = 10.0
# -------------------------------------------------------------------------------------------------------

# The arms' axle runs along the motor box's x, so the box's -y is "forward". Turning the robot a quarter turn
# about the vertical makes that forward point down the world's +x.
START_ROT = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
FORWARD_B = (0.0, -1.0, 0.0)


def _set_weight(spec: mujoco.MjSpec, kg: float) -> None:
  """Scale every part's mass (and how hard it is to spin, its inertia) by the same amount, so the robot weighs kg."""
  scale = kg / sum(b.mass for b in spec.bodies)
  for b in spec.bodies:
    b.mass *= scale
    b.inertia = b.inertia * scale
    b.fullinertia = b.fullinertia * scale


def _get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
  for actuator in list(spec.actuators):     # mjlab adds its own arm motors (below)
    spec.delete(actuator)
  _set_weight(spec, ROBOT_WEIGHT_G / 1000)
  spec.body("chassis").add_freejoint()      # let the robot move around (same as crawl.py)
  for body in spec.bodies:
    for g in body.geoms:
      if g.contype or g.conaffinity:        # the collision shapes (the visual copies do not touch anything)
        g.name = f"{body.name}_collision"
        g.priority = 1                      # the robot's grip, not the floor's, decides the friction
        g.friction[0] = sum(FLOOR_GRIP) / 2
  return spec


def _start_height() -> float:
  """How high to put the chassis so the robot's lowest point starts 5 mm above the floor."""
  model = _get_spec().compile()
  data = mujoco.MjData(model)
  data.qpos[3:7] = START_ROT
  mujoco.mj_forward(model, data)
  lowest = math.inf
  for g in range(model.ngeom):
    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and (model.geom_contype[g] or model.geom_conaffinity[g]):
      m = model.geom_dataid[g]
      verts = model.mesh_vert[model.mesh_vertadr[m]: model.mesh_vertadr[m] + model.mesh_vertnum[m]]
      z = data.geom_xpos[g, 2] + verts @ data.geom_xmat[g].reshape(3, 3)[2]
      lowest = min(lowest, float(z.min()))
  return float(data.qpos[2]) - lowest + 0.005


def _robot_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, _start_height()), rot=START_ROT, joint_vel={".*": 0.0}),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        # stiffness 0 = no position target, so the motor only chases the speed we ask for (same as crawl.py)
        BuiltinPdActuatorCfg(
          target_names_expr=(LEFT, RIGHT), stiffness=0.0, damping=MOTOR_KD, effort_limit=MOTOR_TORQUE_LIMIT,
          delay_min_lag=COMMAND_DELAY_STEPS[0], delay_max_lag=COMMAND_DELAY_STEPS[1],
        ),
      ),
    ),
  )


# ---- what the robot sees, and the rewards: how it is doing relative to "straight down +x" ----------------


def heading(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Which way the robot faces, in radians: 0 = straight down +x."""
  q = env.scene["robot"].data.root_link_quat_w
  f = quat_apply(q, torch.tensor(FORWARD_B, device=q.device).expand(q.shape[0], 3))
  return torch.atan2(f[:, 1], f[:, 0])


def _off_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Sideways distance from this copy's start line (its y axis), in metres."""
  return env.scene["robot"].data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]


def heading_and_offset(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[sin(heading), cos(heading), off-line distance]. Shape [num_envs, 3]."""
  h = heading(env)
  return torch.stack([torch.sin(h), torch.cos(h), _off_line(env)], dim=-1)


def heading_sin_cos(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[sin(heading), cos(heading)]: what the real robot's motion sensor gives (its yaw, set to 0 at the start)."""
  h = heading(env)
  return torch.stack([torch.sin(h), torch.cos(h)], dim=-1)


def yaw_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  """How fast the robot turns, rad/s, counter-clockwise seen from above = +: the real robot's gyroscope (z)."""
  return env.scene["robot"].data.root_link_ang_vel_w[:, 2:3]


def arm_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Where each arm is in its circle, as [sin, cos] per arm (an angle that keeps growing would confuse the brain)."""
  angle = env.scene["robot"].data.joint_pos
  return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)


def forward_progress(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Points for moving down +x: 1 at CRAWL_SPEED, up to 2 for faster, negative for going backwards.

  Summed over a try this is (about) the distance crawled. It is capped so that flinging the body forward in one big
  lurch does not pay more than a steady crawl.
  """
  vx = env.scene["robot"].data.root_link_lin_vel_w[:, 0]
  return torch.clamp(vx / CRAWL_SPEED, min=-1.0, max=2.0)


def stay_on_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  return -torch.clamp(torch.square(_off_line(env) / LINE_SCALE), max=4.0)


def face_forward(env: ManagerBasedRlEnv) -> torch.Tensor:
  return -torch.clamp(torch.square(heading(env) / HEADING_SCALE), max=4.0)


def motor_effort(env: ManagerBasedRlEnv) -> torch.Tensor:
  """How hard the two motors really push (squared, added up), after the strength limit.

  (mjlab's own joint_torques_l2 reads the push the motor is ASKED for, which is far above the limit whenever an arm
  presses into the floor, so it punished pushes that never happen.)
  """
  return torch.sum(torch.square(env.scene["robot"].data.qfrc_actuator), dim=1)


def too_far_off_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _off_line(env).abs() > MAX_OFF_LINE


# ---- environment ----------------------------------------------------------------------------------------


_SKY = TextureCfg(
  name="sky", type="skybox", builtin="gradient", rgb1=(0.55, 0.7, 0.85), rgb2=(0.15, 0.25, 0.35), width=512, height=3072
)


def crawl_straight_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  # What the brain sees: only what the real robot measures, IN THIS ORDER (../real_robot/run_brain.py builds the
  # same list). Arms are in the robot's joint order (export_brain.py writes it down). Noise = how far off a reading
  # may be; delay = it may be up to 2 decisions old.
  def sensor(func, noise):
    return ObservationTermCfg(
      func=func, noise=UniformNoiseCfg(n_min=-noise, n_max=noise),
      delay_min_lag=SENSOR_DELAY_STEPS[0], delay_max_lag=SENSOR_DELAY_STEPS[1],
    )

  actor_terms = {
    "heading": sensor(heading_sin_cos, 0.02),     # [sin, cos]
    "yaw_rate": sensor(yaw_rate, 0.05),           # rad/s
    "arm_phase": sensor(arm_phase, 0.03),         # [sin arm1, sin arm2, cos arm1, cos arm2]
    "arm_vel": sensor(joint_vel_rel, 0.5),        # rad/s, [arm1, arm2]
    "last_action": ObservationTermCfg(func=last_action),   # [arm1, arm2], -1..1
  }
  # The critic (training only) also gets what only the simulation knows.
  critic_terms = {
    "lin_vel": ObservationTermCfg(func=base_lin_vel),
    "ang_vel": ObservationTermCfg(func=base_ang_vel),
    "tilt": ObservationTermCfg(func=projected_gravity),
    "heading_offset": ObservationTermCfg(func=heading_and_offset),
    "arm_phase": ObservationTermCfg(func=arm_phase),
    "arm_vel": ObservationTermCfg(func=joint_vel_rel),
    "last_action": ObservationTermCfg(func=last_action),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
    "critic": ObservationGroupCfg(critic_terms),
  }

  actions: dict[str, ActionTermCfg] = {
    "arms": JointVelocityActionCfg(
      entity_name="robot", actuator_names=(LEFT, RIGHT), scale=ACTION_SCALE, offset=ACTION_OFFSET,
      use_default_offset=False,
    ),
  }

  events = {
    # every try: start pointing a little off +x, arms at random angles
    "reset_pose": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {"yaw": (-START_YAW, START_YAW)}, "velocity_range": {}},
    ),
    "reset_arms": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={"position_range": (-math.pi, math.pi), "velocity_range": (0.0, 0.0)},
    ),
    # once, when the copies are made: each copy gets its own floor grip, weight and motor strength
    "floor_grip": EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_collision",)),
        "operation": "abs",
        "axes": [0],
        "ranges": FLOOR_GRIP,
        "shared_random": True,       # the whole robot touches the same floor
      },
    ),
    "weight": EventTermCfg(
      func=dr.pseudo_inertia,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
        # mass scales by e^(2*alpha), so alpha = ln(scale) / 2
        "alpha_range": (math.log(WEIGHT_SCALE[0]) / 2, math.log(WEIGHT_SCALE[1]) / 2),
      },
    ),
    "motor_strength": EventTermCfg(
      func=dr.effort_limits,
      mode="startup",
      params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)), "effort_limit_range": MOTOR_SCALE},
    ),
  }

  rewards = {
    "forward_progress": RewardTermCfg(func=forward_progress, weight=1.0),
    "stay_on_line": RewardTermCfg(func=stay_on_line, weight=0.5),
    "face_forward": RewardTermCfg(func=face_forward, weight=1.0),
    "dont_tilt": RewardTermCfg(func=flat_orientation_l2, weight=-0.5),     # some rocking is part of crawling
    "dont_flip": RewardTermCfg(func=is_terminated, weight=-20.0),          # tipping over / leaving the line
    "smooth_arms": RewardTermCfg(func=action_rate_l2, weight=-0.5),   # real robot zig-zagged with -0.05
    "save_energy": RewardTermCfg(func=motor_effort, weight=-2.0),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "tipped_over": TerminationTermCfg(func=bad_orientation, params={"limit_angle": math.radians(70.0)}),
    "off_line": TerminationTermCfg(func=too_far_off_line),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      # the default checkered floor, plus a sky (without it, the videos show black above and at the horizon)
      terrain=TerrainEntityCfg(terrain_type="plane", textures=TerrainEntityCfg().textures + (_SKY,)),
      entities={"robot": _robot_cfg()},
      num_envs=1,
      env_spacing=1.5,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="chassis",
      distance=0.4,
      elevation=-20.0,
      azimuth=120.0,
    ),
    # small robot, hard plastic contacts: a finer physics step than the car (0.002 s); the brain decides every DECISION_S
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.002)),
    decimation=round(DECISION_S / 0.002),
    episode_length_s=1e10 if play else EPISODE_S,
  )


def crawl_straight_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(64, 64), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    clip_actions=1.0,                # the brain cannot ask for more than the motor's top speed
    experiment_name="crawl_straight",
    logger="tensorboard",            # the default, wandb, needs an account and a login
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=500,
  )


register_mjlab_task(
  task_id="Crawl-Straight",
  env_cfg=crawl_straight_env_cfg(),
  play_env_cfg=crawl_straight_env_cfg(play=True),
  rl_cfg=crawl_straight_ppo_runner_cfg(),
)
