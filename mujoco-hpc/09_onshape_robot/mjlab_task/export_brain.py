"""Save a trained Crawl-Straight brain as crawl_brain.onnx (a small file the laptop runs to drive the real robot),
plus crawl_brain.json: what goes in, in which order, and what comes out. Runs on the cluster:

    python3 runIt.py --gpu --time 00:10:00 --mem 16G \\
      --with 09_onshape_robot/mjlab_task/crawl_task.py --with 09_onshape_robot/onshape \\
      --with runs/<your run>/logs/rsl_rl/crawl_straight/<date folder>/model_999.pt \\
      09_onshape_robot/mjlab_task/export_brain.py model_999.pt

runIt.py brings both files home to runs/export_brain-<time>/. Copy them to 09_onshape_robot/real_robot/.
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from tensordict import TensorDict

import crawl_task  # registers "Crawl-Straight"
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

TASK = "Crawl-Straight"

if len(sys.argv) != 2:
  sys.exit("usage: export_brain.py model_999.pt")
checkpoint = Path(sys.argv[1])

device = "cuda:0" if torch.cuda.is_available() else "cpu"
env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = 1
agent_cfg = load_rl_cfg(TASK)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=agent_cfg.clip_actions)
runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
runner.export_policy_to_onnx(".", "crawl_brain.onnx")

# What goes in: the actor's observation terms, in order, with their sizes.
obs_manager = env.unwrapped.observation_manager
inputs = [
  {"name": name, "size": int(shape[0])}
  for name, shape in zip(obs_manager.active_terms["actor"], obs_manager.group_obs_term_dim["actor"])
]
arm_order = list(env.unwrapped.scene["robot"].joint_names)   # the order of the two arms, in and out

# A few example inputs and the brain's answers, so the laptop can check the .onnx file gives the same.
policy = runner.get_inference_policy(device=device)
size = sum(i["size"] for i in inputs)
torch.manual_seed(0)
examples = torch.randn(5, size, device=device)
with torch.no_grad():
  answers = torch.stack([policy(TensorDict({"actor": e[None]}, batch_size=[1]))[0] for e in examples]).cpu()
examples = examples.cpu()

info = {
  "task": TASK,
  "checkpoint": checkpoint.name,
  "inputs": inputs,
  "arm_order": arm_order,
  "output": "one value a per arm in arm_order; clip to -1..1; sim arm speed (rad/s) = action_offset + action_scale * a",
  "action_scale": crawl_task.ACTION_SCALE,
  "action_offset": crawl_task.ACTION_OFFSET,
  "max_arm_speed_rad_s": crawl_task.MAX_ARM_SPEED,   # the real motor at 100%
  "decision_period_s": env_cfg.sim.mujoco.timestep * env_cfg.decimation,
  "examples": {"inputs": examples.tolist(), "outputs": answers.tolist()},
}
Path("crawl_brain.json").write_text(json.dumps(info, indent=2))
print(json.dumps({k: v for k, v in info.items() if k != "examples"}, indent=2))
print("wrote crawl_brain.onnx and crawl_brain.json")
