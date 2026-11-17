"""Watch a trained Drive-Straight policy (or a dummy one). Same as mjlab's `play`, with our tasks registered.

    python play.py Drive-Straight --agent zero          # wheels do nothing: checks the robot loads
    python play.py Drive-Straight --agent random        # random wheel speeds
    python play.py Drive-Straight --checkpoint-file logs/rsl_rl/drive_straight/<run>/model_499.pt
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

from mjlab.scripts.play import main

if __name__ == "__main__":
  sys.exit(main())
