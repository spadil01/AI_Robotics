"""Train Drive-Straight or Crawl-Straight. Same as mjlab's own `train`, but it registers our task first.

    python train.py Drive-Straight --env.scene.num-envs 4096
"""
import importlib
import sys

# Register whichever of our tasks runIt.py sent along (--with): drive_task.py = "Drive-Straight", crawl_task.py = "Crawl-Straight".
for _name in ("drive_task", "crawl_task"):
  try:
    importlib.import_module(_name)
  except ModuleNotFoundError as e:
    if e.name != _name:   # the file is there but something it needs is missing: show that error
      raise

from mjlab.scripts.train import main

if __name__ == "__main__":
  sys.exit(main())
