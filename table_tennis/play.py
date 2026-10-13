"""
Starts the table tennis game.

Run from the repo root:  .venv/bin/python table_tennis/play.py
"""

import os
import subprocess
import sys
from pathlib import Path

from panda3d.core import loadPrcFileData

# macOS: ask for a modern OpenGL context so Panda3D's per-pixel lighting and
# shadow shaders work (the default legacy context draws everything black).
loadPrcFileData("", """
window-title Table Tennis
win-size 1280 720
gl-version 3 2
sync-video true
""")

# With headphones plugged in, OpenAL turns on its headphone "3D sound" (HRTF),
# which crashes on this Mac. alsoft.conf switches it off (must be set before
# any sound starts; the sound check below inherits it too).
os.environ["ALSOFT_CONF"] = str(Path(__file__).resolve().parent / "alsoft.conf")

# Starting the sound can still crash the whole program when the Mac's audio is in a
# bad state (seen with headphones plugged in). Try it in a throwaway program
# first; if that one crashes, play without sound instead of not starting at all.
SOUND_CHECK = ("from panda3d.core import loadPrcFileData; loadPrcFileData('', 'window-type none'); "
               "from direct.showbase.ShowBase import ShowBase; ShowBase()")
try:
    sound_ok = subprocess.run([sys.executable, "-c", SOUND_CHECK], capture_output=True, timeout=20).returncode == 0
except subprocess.TimeoutExpired:
    sound_ok = False
if not sound_ok:
    print("WARNING: the Mac's sound isn't working, so the game will be silent.\n"
          "  Try: unplug/replug your headphones, or pick another output in System Settings > Sound,\n"
          "  then restart the game.")
    loadPrcFileData("", "audio-library-name null")

from direct.showbase.ShowBase import ShowBase  # noqa: E402

from common.config import load_config  # noqa: E402
from game.main import Game  # noqa: E402
from game.scene import build_world  # noqa: E402

base = ShowBase()
build_world(base)
Game(base, load_config())
base.run()
