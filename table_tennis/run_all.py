"""
Starts everything with one command: the MQTT broker (if it isn't running),
the paddle node, the vision node and the game. Closing the game (or pressing
Ctrl+C here) shuts the others down cleanly.

Run from the repo root:
    .venv/bin/python table_tennis/run_all.py               # everything
    .venv/bin/python table_tennis/run_all.py --no-paddle   # keyboard instead of the paddle
    .venv/bin/python table_tennis/run_all.py --no-camera   # T key instead of raising your hand

Output from each program is shown with a [paddle] / [camera] / [game] prefix.
"""

import argparse
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable
BROKER_PORT = 1883
MOSQUITTO_PATHS = ["/opt/homebrew/opt/mosquitto/sbin/mosquitto", "/usr/local/opt/mosquitto/sbin/mosquitto"]

# Startup chatter from MediaPipe / Panda3D that isn't useful here.
NOISE = ("I0000", "W0000", "INFO: Created TensorFlow", ":display", "Known pipe types", "  CocoaGraphicsPipe",
         "(all display", "(3 aux", "  requested:", "  got:", ":pnmimage")


def broker_running():
    try:
        with socket.create_connection(("localhost", BROKER_PORT), timeout=0.5):
            return True
    except OSError:
        return False


def start_broker():
    """Start Mosquitto if needed. Returns the process if we started it, else None."""
    if broker_running():
        return None
    path = shutil.which("mosquitto") or next((p for p in MOSQUITTO_PATHS if Path(p).exists()), None)
    if path is None:
        sys.exit("Mosquitto isn't installed. Install it with:  brew install mosquitto")
    print("[launcher] starting the MQTT broker (Mosquitto)...")
    proc = subprocess.Popen([path, "-p", str(BROKER_PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        if broker_running():
            return proc
        time.sleep(0.1)
    sys.exit("The MQTT broker didn't start.")


def launch(name, script, *script_args):
    """Start a program from this folder, printing its output with a [name] prefix."""
    proc = subprocess.Popen([PYTHON, "-u", str(HERE / script), *script_args], cwd=HERE.parent,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            # make sure Ctrl+C works in the child even if this process was started
                            # somewhere that ignores it
                            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))

    def relay():
        for line in proc.stdout:
            if line.strip() and not line.startswith(NOISE):
                print(f"[{name}] {line}", end="", flush=True)

    threading.Thread(target=relay, daemon=True).start()
    return proc


def stop(proc, name):
    """Ask nicely first (Ctrl+C, so the paddle disconnects properly), then firmly, then force it."""
    for send, wait in ((lambda: proc.send_signal(signal.SIGINT), 4), (proc.terminate, 2), (proc.kill, 2)):
        if proc is None or proc.poll() is not None:
            return
        send()
        try:
            proc.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            print(f"[launcher] {name} is slow to stop, pushing harder")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-paddle", action="store_true", help="don't start the paddle node")
    parser.add_argument("--no-camera", action="store_true", help="don't start the vision node")
    args = parser.parse_args()
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)   # treat "terminate" like Ctrl+C

    broker = start_broker()
    helpers = []
    if not args.no_paddle:
        helpers.append(("paddle", launch("paddle", "paddle_node.py")))
    if not args.no_camera:
        helpers.append(("camera", launch("camera", "vision_node.py", "--headless")))
    game = launch("game", "play.py")
    print("[launcher] everything started. Close the game window (or press Ctrl+C here) to quit.")

    try:
        game.wait()
    except KeyboardInterrupt:
        stop(game, "game")
    finally:
        print("[launcher] shutting down...")
        for name, proc in helpers:
            stop(proc, name)
        if broker is not None:
            broker.terminate()
        print("[launcher] done.")


if __name__ == "__main__":
    main()
