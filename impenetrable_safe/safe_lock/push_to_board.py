"""Put the neural network on the UNO Q and start the "board decides" safe.

Puts these into ArduinoApps/impenetrable-safe-onboard/python/:
    model.npz      the trained network's numbers (made here from model.pt)
    numpy_net.py   runs the network with NumPy (the board doesn't need PyTorch)
    pictures.py    how camera pictures are prepared
    safe_rules.py  the game rules (CODE, timings...)
then stops the laptop-decides app (only one app can run at a time) and starts
the board-decides app with ArduinoApps/tools/deploy.py.

Run it again whenever you retrain the network or change safe_rules.py.

Run:  python push_to_board.py
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from numpy_net import export

HERE = Path(__file__).resolve().parent
FILES = ["numpy_net.py", "pictures.py", "safe_rules.py"]
APP = "impenetrable-safe-onboard"
OTHER_APP = "impenetrable-safe"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apps", default=str(HERE.parents[2] / "ArduinoApps"),
                    help="where the ArduinoApps folder is")
    args = ap.parse_args()

    apps = Path(args.apps)
    target = apps / APP / "python"
    if not target.is_dir():
        raise SystemExit(f"Can't find {target}")
    export(HERE / "model.pt", target / "model.npz")      # the network's numbers
    print("saved model.npz")
    for name in FILES:
        shutil.copy2(HERE / name, target / name)
        print(f"copied {name}")

    deploy = [sys.executable, str(apps / "tools" / "deploy.py")]
    subprocess.run(deploy + ["--cmd", f"arduino-app-cli app stop /home/arduino/ArduinoApps/{OTHER_APP}"],
                   cwd=apps)   # fine if it wasn't running
    sys.exit(subprocess.run(deploy + [APP], cwd=apps).returncode)


if __name__ == "__main__":
    main()
