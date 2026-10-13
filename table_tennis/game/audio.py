"""
Plays the sound effects in assets/sounds/ (made by tools/make_sounds.py).

Each sound is loaded a few times so quick repeats (bounce, hit, bounce) can
overlap instead of cutting each other off. Sounds can be panned left/right
to follow the ball.
"""

from panda3d.core import Filename

from common.config import TABLE_TENNIS_DIR

SOUND_DIR = TABLE_TENNIS_DIR / "assets" / "sounds"
COPIES = {"hit": 3, "bounce": 3, "floor": 2}
CROWD = {"cheer", "aww", "clap"}


class Sounds:
    def __init__(self, base, cfg):
        acfg = cfg["audio"]
        self.effects_volume = acfg["volume"] * acfg["effects"]
        self.crowd_volume = acfg["volume"] * acfg["crowd"]
        self.pools = {}
        self.next = {}
        for path in sorted(SOUND_DIR.glob("*.wav")):
            name = path.stem
            self.pools[name] = [base.loader.loadSfx(Filename.fromOsSpecific(str(path)))
                                for _ in range(COPIES.get(name, 1))]
            self.next[name] = 0
        if not self.pools:
            print("No sounds found. Make them with: .venv/bin/python table_tennis/tools/make_sounds.py")

    def play(self, name, volume=1.0, rate=1.0, pan=0.0):
        """pan: -1 left .. 1 right."""
        pool = self.pools.get(name)
        if not pool:
            return
        sound = pool[self.next[name]]
        self.next[name] = (self.next[name] + 1) % len(pool)
        base_volume = self.crowd_volume if name in CROWD else self.effects_volume
        sound.setVolume(min(volume * base_volume, 1.0))
        sound.setPlayRate(rate)
        sound.setBalance(max(-1.0, min(1.0, pan)))
        sound.play()
