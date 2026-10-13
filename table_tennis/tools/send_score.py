"""
Resends a score to the class broker, for when it was down at the end of a match.

    .venv/bin/python table_tennis/tools/send_score.py          # resend the last match's score
    .venv/bin/python table_tennis/tools/send_score.py 23       # send "Hits: 23.0" instead
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_config  # noqa: E402
from game.result import LOG_PATH, send  # noqa: E402


def main():
    rcfg = load_config()["result"]
    topic = rcfg["topic_prefix"] + rcfg["player_name"]
    if len(sys.argv) > 1:
        payload = f"Hits: {float(sys.argv[1])}"
    else:
        lines = [l for l in LOG_PATH.read_text().splitlines() if "Hits:" in l] if LOG_PATH.exists() else []
        if not lines:
            print("No score saved yet. Give one, e.g.: send_score.py 23")
            return
        payload = "Hits: " + lines[-1].split("Hits:")[1].split()[0]
    print(f"Sending {payload} -> {topic} ...")
    for attempt in range(3):
        if send(rcfg["broker"], topic, payload):
            print("Sent!")
            return
    print("The score server still isn't answering. Try again later.")


if __name__ == "__main__":
    main()
