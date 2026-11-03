"""Copies the step counter's trials from the board to data/ on the laptop.

    python step_counter/get_data.py [--board 10.247.137.83]

Downloads trials that are new, or whose info changed on the board (like a
step count typed in later). Never deletes anything on the laptop; trials
deleted on the board are just listed.
"""
import argparse
import json
from pathlib import Path

from websockets.sync.client import connect

DATA_DIR = Path(__file__).resolve().parent / "data"


def local_meta(path):
    try:
        with open(path) as f:
            return json.loads(f.readline()[2:])
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def wait_for(ws, kind):
    """Skip the live sensor readings until a message of this kind arrives."""
    while True:
        msg = json.loads(ws.recv(timeout=30))
        if msg["type"] == kind:
            return msg
        if msg["type"] == "error":
            raise RuntimeError(msg["text"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", default="10.247.137.83", help="the UNO Q's address")
    args = ap.parse_args()
    DATA_DIR.mkdir(exist_ok=True)

    with connect(f"ws://{args.board}:8765", max_size=None, open_timeout=5) as ws:
        items = wait_for(ws, "recordings")["items"]
        got = same = 0
        for item in items:
            path = DATA_DIR / item["file"]
            meta = {k: v for k, v in item.items() if k != "seconds"}
            if local_meta(path) == meta:
                same += 1
                continue
            ws.send(json.dumps({"type": "get", "file": item["file"]}))
            path.write_text(wait_for(ws, "file")["text"])
            got += 1
            print(f"got {item['file']}" + ("  (no step count yet!)" if item["steps"] is None else ""))

    on_board = {item["file"] for item in items}
    only_here = sorted(p.name for p in DATA_DIR.glob("*.csv") if p.name not in on_board)
    print(f"\n{got} downloaded, {same} already up to date.")
    if only_here:
        print(f"{len(only_here)} trial(s) are only on the laptop (deleted on the board?): " + ", ".join(only_here))
    missing = [i["file"] for i in items if i["steps"] is None]
    if missing:
        print(f"{len(missing)} trial(s) still need a step count typed in on the page.")


if __name__ == "__main__":
    main()
