"""Record a video of a trained Drive-Straight or Crawl-Straight policy. Made for the cluster, which has no screen.

    python3 runIt.py --gpu --time 00:10:00 --mem 16G \
      --with 09_onshape_robot/mjlab_task/drive_task.py --with 09_onshape_robot/onshape \
      --with runs/<your run>/logs/rsl_rl/drive_straight/<date folder>/model_499.pt \
      09_onshape_robot/mjlab_task/make_video.py model_499.pt

For the crawler, send crawl_task.py instead of drive_task.py and add the task name at the end:
      ... 09_onshape_robot/mjlab_task/make_video.py model_499.pt Crawl-Straight

runIt.py brings the .mp4 home to ./runs/ and opens it. (mjlab's own play.py also records a video, but then opens a
web viewer that never exits on a machine without a screen.)
"""
import importlib
import math
import sys
from dataclasses import asdict
from pathlib import Path

import torch

# Register whichever of our tasks runIt.py sent along (--with): drive_task.py = "Drive-Straight", crawl_task.py = "Crawl-Straight".
for _name in ("drive_task", "crawl_task"):
  try:
    importlib.import_module(_name)
  except ModuleNotFoundError as e:
    if e.name != _name:   # the file is there but something it needs is missing: show that error
      raise

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


if len(sys.argv) not in (2, 3):
  sys.exit("usage: make_video.py model_499.pt [Drive-Straight | Crawl-Straight]")
checkpoint = Path(sys.argv[1])
TASK = sys.argv[2] if len(sys.argv) == 3 else "Drive-Straight"

configure_torch_backends()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = 1
env_cfg.viewer.width, env_cfg.viewer.height = 640, 480
agent_cfg = load_rl_cfg(TASK)

env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
STEPS_PER_S = round(1 / (env_cfg.sim.mujoco.timestep * env_cfg.decimation))   # how often the policy acts
STEPS = 10 * STEPS_PER_S                                                          # 10 seconds
env = VideoRecorder(env, video_folder=".", step_trigger=lambda step: step == 0, video_length=STEPS, disable_logger=True)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)
runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

robot = env.unwrapped.scene["robot"]


def yaw(q):
  w, qx, qy, qz = q.tolist()
  return torch.atan2(torch.tensor(2 * (w * qz + qx * qy)), torch.tensor(1 - 2 * (qy * qy + qz * qz))).item()


start_yaw = yaw(robot.data.default_root_state[0, 3:7])   # the crawler starts turned a quarter turn; 0 = facing +x
print("  second   x (m)   y (m)   heading (deg)   speed (m/s)   arm speeds asked -> actual (rad/s)")
obs = env.get_observations()
with torch.no_grad():
  for step in range(STEPS):
    action = policy(obs)
    obs = env.step(action)[0]
    if step % STEPS_PER_S == STEPS_PER_S - 1:   # once a second
      x, y = robot.data.root_link_pos_w[0, :2].tolist()
      heading = math.remainder(yaw(robot.data.root_link_quat_w[0]) - start_yaw, 2 * math.pi)
      vx = robot.data.root_link_lin_vel_w[0, 0].item()
      actions = env.unwrapped.action_manager
      asked = actions.get_term(actions.active_terms[0])._processed_actions[0]   # the speeds the policy asked for
      actual = robot.data.joint_vel[0]
      arms = "  ".join(f"{a:+5.1f}->{b:+5.1f}" for a, b in zip(asked.tolist(), actual.tolist()))
      print(f"  {(step + 1) / STEPS_PER_S:5.0f}  {x:7.2f} {y:7.2f}  {math.degrees(heading):11.1f}  {vx:12.2f}   {arms}")
env.close()
print("wrote the video (an .mp4 next to this script)")
