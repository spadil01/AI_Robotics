"""
The game controller: serving, hitting, the CPU opponent, scoring and the HUD.

Input arrives as Swing objects (from the keyboard for now; the real paddle
will feed the same Swing objects through MQTT later). Anything coming from
another thread must go through `self.inbox`, never touch the scene directly.
"""

import colorsys
import queue
import random
import sys
import time as clock

from direct.gui.DirectGui import DirectFrame
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import KeyboardButton, TextNode

from .audio import Sounds
from .avatar import CONTACT, SCALE, Avatar
from .calibration import Calibration
from .camera_view import CameraView
from .crowd import Crowd
from .effects import Sparkles
from .menu import ControlsMenu, MainMenu, MatchOverMenu, PracticeMenu
from .sensors import SensorLink, paddle_face
from .physics import CPU, DT, HALF_LENGTH, HALF_WIDTH, PLAYER, TABLE_TOP, Ball, predict_contact
from .result import publish_returns
from .rules import Rally, Score
from .scene import rgb, to_panda
from .scoreboard import Scoreboard
from .shapes import part
from .shots import Swing, opponent_shot, player_shot, serve_shot

TOSS_SPEED = 2.6          # m/s upward when the ball is tossed for a serve
CPU_SERVE_DELAY = 1.2     # s before the CPU tosses
CPU_TOSS_TO_HIT = 0.42    # s between the CPU's toss and its serve
POINT_PAUSE = 1.8         # s to show the point result before the next serve
PADDLE_REACH = CONTACT.x * SCALE   # sideways distance from an avatar's body to its paddle
BALL_DRAW_SIZE = 0.06     # drawn a bit bigger than a real ball so it's easy to see

WINDUP_LEAD = 0.5         # s before contact when an avatar starts winding up
SOFT, NORMAL, HARD = 0.2, 0.6, 1.0

PLAYER_SHIRT = rgb(235, 95, 60)
CPU_SHIRT = rgb(60, 170, 100)
# If the player's shirt looks too much like the opponent's green, the opponent
# switches to whichever of these is most different, so the teams stay easy to tell apart.
CPU_ALTERNATIVES = [CPU_SHIRT, rgb(60, 110, 220), rgb(150, 80, 200), rgb(235, 190, 50)]
SIMILAR_COLORS = 0.35        # RGB distance below which two shirts look alike
SHIRT_MAX_AGE = 2.0          # s: ignore shirt colors older than this (player left the camera)


def color_distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a[:3], b[:3])) ** 0.5


def display_color(rgb_camera):
    """Camera colors come out darker and duller than they look; brighten them a bit
    so the character's shirt reads as the same color under the game's lights."""
    h, s, v = colorsys.rgb_to_hsv(*rgb_camera)
    r, g, b = colorsys.hsv_to_rgb(h, min(1.0, s * 1.2), min(1.0, max(v, 0.3) * 1.15))
    return (r, g, b, 1)

def hud_text(base, pos, scale, align=TextNode.ACenter, parent=None, fg=(1, 1, 1, 1)):
    return OnscreenText(text="", pos=pos, scale=scale, fg=fg, shadow=(0, 0, 0, 0.6), align=align,
                        mayChange=True, parent=parent or base.aspect2d)


class Game:
    def __init__(self, base, cfg):
        self.base = base
        self.cfg = cfg
        g = cfg["game"]
        self.handed = 1 if g["handedness"] == "right" else -1
        self.diff = cfg["difficulty"][g["difficulty"]]
        self.window = self.diff["timing_window_ms"] / 1000
        self.grace = g["grace_ms"] / 1000
        self.attempt_lead = g["attempt_lead_ms"] / 1000
        self.player_plane = g["player_hit_plane"]
        self.cpu_plane = g["cpu_hit_plane"]
        self.inbox = queue.Queue()   # (kind, data) from other threads
        self.rng = random.Random()
        self.sensors = SensorLink(cfg, self.inbox)
        self.paddle_connected = False
        self.paddle_seen = False       # only pause for a paddle that was actually in use
        self.paused = False
        self.pause_started = 0.0
        self.resume_at = None
        self.calibration = None
        self.menu = None
        self.practice = False          # practice mode: endless rallies, no score, nothing published
        self.practice_menu = None      # its "End practice" button (only between rallies)
        self.best_rally = 0
        self.settings = {"difficulty": g["difficulty"], "match_points": g["match_points"]}

        render = base.render
        self.crowd = Crowd(render, team_shirts={PLAYER: PLAYER_SHIRT, CPU: CPU_SHIRT})
        self.player_color = PLAYER_SHIRT     # becomes the player's real shirt color (see update_shirt_color)
        self.cpu_color = CPU_SHIRT
        self.shirt_checked_at = 0.0
        self.sounds = Sounds(base, cfg)
        self.player = Avatar(render, shirt=PLAYER_SHIRT, pants=rgb(50, 60, 95), hair=rgb(70, 45, 30),
                             pos=(0, self.player_plane - 0.3, 0), facing=1, handed=self.handed)
        self.cpu = None
        self.make_opponent(dog=False)
        self.ball_np = part(render, "sphere", (1, 1, 1, 1), scale=BALL_DRAW_SIZE)
        self.shadow = part(render, "cylinder", (0, 0, 0, 0.35), scale=(0.05, 0.05, 0.002))
        self.shadow.setLightOff()
        self.sparkles = Sparkles(render, self.rng)

        self.scoreboard = Scoreboard(render, base.loader, base.camera.getPos(), PLAYER_SHIRT, CPU_SHIRT)
        self.message = hud_text(base, (0, 0.45), 0.12)
        self.detail = hud_text(base, (0, 0.3), 0.07)
        self.debug = hud_text(base, (0.05, -0.12), 0.05, align=TextNode.ALeft, parent=base.a2dTopLeft)
        self.debug.hide()
        self.debug_on = False
        self.paddle_text = hud_text(base, (-0.05, -0.08), 0.05, align=TextNode.ARight, parent=base.a2dTopRight)
        if self.sensors.start():
            self.sensors.send_config(handedness=g["handedness"])
        else:
            self.paddle_text.setText("Sensors off (no MQTT broker)")
        self.camera_view = CameraView(base)

        self.pause_overlay = DirectFrame(parent=base.aspect2d, frameColor=(0.05, 0.08, 0.15, 0.75),
                                         frameSize=(-3, 3, -1, 1))
        self.pause_text = OnscreenText(text="", pos=(0, 0.05), scale=0.1, fg=(1, 1, 1, 1), mayChange=True,
                                       parent=self.pause_overlay)
        self.pause_overlay.hide()
        for key, strength in (("z", SOFT), ("space", NORMAL), ("x", HARD)):
            base.accept(key, self.on_swing_key, [strength])
        base.accept("t", self.on_toss_key)
        base.accept("enter", self.on_enter)
        base.accept("l", self.on_calibration_key, ["l"])
        base.accept("r", self.on_calibration_key, ["r"])
        base.accept("c", self.start_calibration)
        base.accept("m", self.end_practice)
        base.accept("tab", self.toggle_debug)
        base.accept("escape", sys.exit)
        base.taskMgr.add(self.update, "game-update")

        self.new_match()
        if g["calibrate"]:
            self.start_calibration()
        else:
            self.show_menu()

    # --- calibration ------------------------------------------------------------

    def start_calibration(self):
        """Show the calibration screen (at startup, or press C between points)."""
        if self.calibration or self.state not in ("serve", "match_over", "menu"):
            return
        self.close_menu()
        self.end_practice_button()
        self.sensors.send_config(show_shirt_box=True)     # the color box shows on the camera picture
        self.state = "calibrate"
        self.held = True
        self.message.setText("")
        self.detail.setText("")
        self.calibration = Calibration(self)
        self.show_match_hud(False)

    def show_match_hud(self, shown):
        self.scoreboard.set_visible(shown)

    def recalibrate_from_menu(self):
        self.start_calibration()

    # --- menus --------------------------------------------------------------------

    def show_menu(self):
        self.close_menu()
        self.state = "menu"
        self.held = True
        self.show_match_hud(False)
        self.message.setText("")
        self.detail.setText("")
        self.end_practice_button()
        self.sensors.send_config(show_shirt_box=True)
        self.practice = False
        self.menu = MainMenu(self, on_start=self.start_from_menu, on_practice=self.start_practice,
                             on_controls=self.show_controls)
        aspect = self.base.getAspectRatio()
        self.camera_view.place(aspect - 0.36, -0.78, 0.36)

    def show_controls(self):
        self.close_menu()
        self.menu = ControlsMenu(self, on_back=self.show_menu)

    def close_menu(self):
        if self.menu:
            self.menu.destroy()
            self.menu = None
        self.camera_view.hide()

    def start_from_menu(self, practice=False):
        self.close_menu()
        self.apply_settings()
        self.practice = practice
        self.sensors.send_config(show_shirt_box=False)    # it would cover the small in-game camera view
        self.show_match_hud(True)
        self.show_game_camera()
        self.new_match()

    # --- practice mode ----------------------------------------------------------

    def start_practice(self):
        self.best_rally = 0
        self.start_from_menu(practice=True)

    def end_practice(self):
        if self.practice_menu:          # same rule as the button: only before your serve
            self.show_menu()

    def end_practice_button(self):
        if self.practice_menu:
            self.practice_menu.destroy()
            self.practice_menu = None

    def update_practice_button(self, dt):
        """Show "End practice" only while you're about to serve (ball still in your hand),
        so there's time to choose it and the cursor isn't in the way while playing."""
        between = (self.practice and self.state == "serve" and self.rally.hitter == PLAYER
                   and self.held and not self.paused)
        if between and not self.practice_menu:
            self.practice_menu = PracticeMenu(self, on_end=self.end_practice)
        elif not between:
            self.end_practice_button()
        if self.practice_menu:
            self.practice_menu.update(dt)

    def show_game_camera(self):
        """Small camera view in the bottom-right corner during matches."""
        aspect = self.base.getAspectRatio()
        self.camera_view.place(aspect - 0.3, -0.78, 0.27, border_color=(1, 1, 1, 0.8))

    def make_opponent(self, dog):
        """The opponent: a person, or the dog on the hardest difficulty."""
        if self.cpu and self.cpu.dog == dog:
            return
        if self.cpu:
            self.cpu.root.removeNode()
        self.cpu = Avatar(self.base.render, shirt=self.cpu_color, pants=rgb(40, 40, 50), hair=rgb(30, 25, 25),
                          pos=(0, self.cpu_plane + 0.3, 0), facing=-1, handed=1, dog=dog)

    # --- the player's shirt color -------------------------------------------------

    def update_shirt_color(self, now):
        """While calibrating or on the menu, the player's character (and their fans,
        and their scoreboard strip) take on the color of the shirt the camera sees.
        It glides toward each new measurement and locks once a match starts."""
        if now - self.shirt_checked_at < 0.25:
            return
        self.shirt_checked_at = now
        shirt = self.sensors.shirt
        if not shirt or now - self.sensors.shirt_at > SHIRT_MAX_AGE:
            return
        target = display_color(shirt["rgb"])
        color = tuple(c + (t - c) * 0.5 for c, t in zip(self.player_color, target))
        if color_distance(color, self.player_color) > 0.01:
            self.set_player_color(color)

    def set_player_color(self, color):
        self.player_color = color
        self.player.set_team_color(color)
        self.crowd.set_team_color(PLAYER, color)
        cpu_color = CPU_SHIRT
        if color_distance(color, CPU_SHIRT) < SIMILAR_COLORS:
            cpu_color = max(CPU_ALTERNATIVES, key=lambda c: color_distance(c, color))
        if cpu_color != self.cpu_color:
            self.cpu_color = cpu_color
            self.cpu.set_team_color(cpu_color)
            self.crowd.set_team_color(CPU, cpu_color)
        self.scoreboard.set_colors(color, self.cpu_color)

    def apply_settings(self):
        g = self.cfg["game"]
        g["difficulty"] = self.settings["difficulty"]
        self.make_opponent(dog=g["difficulty"] == "pro")
        g["match_points"] = self.settings["match_points"]
        self.diff = self.cfg["difficulty"][g["difficulty"]]
        self.window = self.diff["timing_window_ms"] / 1000

    def set_handedness(self, handedness):
        self.cfg["game"]["handedness"] = handedness
        self.handed = 1 if handedness == "right" else -1
        self.player.handed = self.handed
        self.sensors.send_config(handedness=handedness)

    def on_calibration_key(self, key):
        if self.calibration:
            self.calibration.on_key(key)

    # --- match and point flow ---------------------------------------------

    def new_match(self):
        g = self.cfg["game"]
        if self.practice:
            # No winner ever, and the serve swaps every rally so you practice both.
            self.score = Score(10 ** 9, 1, first_server=PLAYER)
        else:
            self.score = Score(g["match_points"], g["serves_per_turn"], first_server=PLAYER)
        names = g["dog_names"] if self.cpu.dog else g["opponent_names"]
        self.opponent_name = self.rng.choice(names)
        self.returns = 0
        self.last_player_strength = 0.5
        self.last_delta = None
        self.start_point()

    def start_point(self):
        server = self.score.server
        self.rally = Rally(server)
        self.rally_count = 0         # your good returns in this rally
        self.state = "serve"
        self.ball = Ball()
        self.held = True
        self.sim_t = clock.time()
        self.player_contact = None   # (time, position) where the player should hit
        self.cpu_contact = None
        self.pending_swing = None
        self.windup_contact = None   # the contact a swing used up, in case it was only a wind-up
        self.player_ready = True     # False from the player's hit until that ball crosses the net
        self.cpu_serve_at = self.sim_t + CPU_SERVE_DELAY if server == CPU else None
        self.cpu_hit_at = None
        self.player.serving = server == PLAYER
        self.cpu.serving = server == CPU
        # Stand with the paddle near the middle, body off to the side, so the
        # player's own avatar doesn't hide the ball.
        self.player.target_x = -PADDLE_REACH * self.handed
        self.cpu.target_x = 0
        self.detail.setText("")
        self.message.setText("Your serve: raise your free hand to toss" if server == PLAYER else "")

    def end_point(self, winner):
        self.state = "point_over"
        self.score.add(winner)
        self.point_over_at = clock.time() + POINT_PAUSE
        self.player_contact = self.cpu_contact = None
        self.sensors.send_event("point_won" if winner == PLAYER else "point_lost")
        self.crowd.react(winner, 0.4)
        self.sounds.play("point_won" if winner == PLAYER else "point_lost", volume=0.6)
        self.sounds.play("cheer", volume=0.75 if winner == PLAYER else 0.55)
        self.sounds.play("aww", volume=0.3 if winner == PLAYER else 0.5)
        self.sounds.play("clap", volume=0.4)
        if winner == PLAYER:
            self.message.setText("Point!")
            self.player.celebrate()
            self.cpu.slump()
        else:
            self.message.setText(f"{self.opponent_name}'s point")
            self.cpu.celebrate()
            self.player.slump()
        if self.practice:
            n = self.rally_count
            if n > self.best_rally:
                self.best_rally = n
                self.message.setText(f"New best: {n} in a row!")
            else:
                self.message.setText(f"{n} in a row" if n != 1 else "1 return")

    def end_match(self):
        self.state = "match_over"
        self.message.setText("You win!" if self.score.winner == PLAYER else f"{self.opponent_name} wins")
        self.crowd.react(self.score.winner, 1.0)
        won = self.score.winner == PLAYER
        self.sounds.play("match_won" if won else "match_lost", volume=0.8)
        self.sounds.play("cheer", volume=1.0)
        self.sounds.play("aww", volume=0.4 if won else 0.7)
        if self.score.winner == PLAYER:
            self.sensors.send_event("match_won")
        self.show_final("Sending score...")
        publish_returns(self.cfg["result"], self.returns,
                        on_done=lambda ok, msg: self.inbox.put(("result", msg)))
        self.menu = MatchOverMenu(self, on_continue=self.show_menu)

    def show_final(self, status):
        p, c = self.score.points[PLAYER], self.score.points[CPU]
        self.detail.setText(f"Final score {p} - {c}\nTotal returns: {self.returns}\n\n{status}")

    # --- hitting ------------------------------------------------------------

    @staticmethod
    def hand_ball_pos(avatar):
        """The ball sits in the server's free hand (Panda3D -> physics axes)."""
        p = avatar.free_hand_world()
        return [p.x, p.z + 0.07, p.y]

    def toss(self):
        self.held = False
        (self.player if self.rally.hitter == PLAYER else self.cpu).toss()
        self.ball.vel = [0.0, TOSS_SPEED, 0.0]
        self.sim_t = clock.time()
        self.message.setText("")

    def hit(self, who, pos, vel, after_bounce=None, power=0.6):
        self.sounds.play("hit", volume=0.55 + 0.45 * power, rate=0.92 + 0.22 * power, pan=pos[0] / 2)
        self.ball.pos = list(pos)
        self.ball.vel = list(vel)
        self.ball.bounce_override = after_bounce
        self.pending_swing = None
        if who == PLAYER:
            self.player_ready = False    # ignore the follow-through until this shot crosses the net
        if after_bounce is None:
            self.rally.on_hit(who)   # serves keep the rally in "serving" mode
        if who == PLAYER:
            self.player_contact = None
            self.cpu_contact = self.predict(CPU, self.cpu_plane)
            if self.cpu_contact:
                self.cpu.target_x = self.cpu_contact[1][0] + PADDLE_REACH
        else:
            self.sparkles.burst(to_panda(pos), 0.25 * power)    # smaller burst for the opponent
            self.cpu_contact = None
            self.player_contact = self.predict(PLAYER, self.player_plane)
            if self.player_contact:
                self.player.target_x = self.player_contact[1][0] - PADDLE_REACH * self.handed

    def predict(self, side, plane):
        found = predict_contact(self.ball, side, plane)
        return None if found is None else (self.sim_t + found[0], found[1])

    def on_swing(self, swing, replaces=False):
        """replaces: this is the real stroke right after a wind-up the paddle
        reported as a swing, so undo whatever that wind-up did."""
        if self.paused:
            return
        if replaces:
            # Only counts if the swing before it was too early (a wind-up);
            # otherwise it's just the paddle being brought back.
            if not (self.state == "rally" and self.windup_contact and self.player_ready):
                return
            self.player_contact, self.windup_contact = self.windup_contact, None
            self.pending_swing = None
            self.message.setText("")
        self.windup_contact = None
        if self.calibration:
            self.calibration.on_swing()
            return
        height = self.player_contact[1][1] if self.player_contact else self.ball.pos[1]
        self.player.swing(swing.strength, swing.face, contact_height=height)
        if self.state == "serve":
            if self.rally.hitter == PLAYER and not self.held:
                target_x = self.rng.uniform(-0.5, 0.5) * HALF_WIDTH
                vel, after = serve_shot(self.ball.pos, PLAYER, target_x, swing.strength, self.cfg["shots"])
                self.last_player_strength = swing.strength
                self.state = "rally"
                self.player.serving = False
                self.hit(PLAYER, self.ball.pos, vel, after_bounce=after, power=swing.strength)
            return
        if self.state != "rally" or self.player_contact is None or not self.player_ready:
            return
        contact_t, contact_pos = self.player_contact
        delta = swing.t - contact_t
        if delta < -self.attempt_lead or self.pending_swing:
            return                          # ball still far away, or already committed to a swing
        self.last_delta = delta
        self.windup_contact = self.player_contact if delta < 0 else None
        if abs(delta) > self.window:
            # One attempt per ball: a mistimed swing is a whiff, no second chances.
            self.player_contact = None
            return
        if self.sim_t < contact_t:
            self.pending_swing = swing      # early but OK: hit when the ball arrives
        else:
            self.player_hit(swing, delta, contact_pos)

    def player_hit(self, swing, delta, contact_pos):
        vel = player_shot(contact_pos, swing, delta / self.window, self.handed, self.cfg["shots"])
        self.last_player_strength = swing.strength
        self.sensors.send_event("hit", swing.strength)
        self.hit(PLAYER, contact_pos, vel, power=swing.strength)
        # Sparkles: bigger and more of them the harder you hit; smashes add orange.
        smash = contact_pos[1] > TABLE_TOP + self.cfg["shots"]["smash_height"] and swing.strength > 0.8
        self.sparkles.burst(to_panda(contact_pos), swing.strength, smash)

    def check_contacts(self):
        """Called every physics step, before the ball moves."""
        if self.player_contact:
            contact_t, contact_pos = self.player_contact
            if self.pending_swing and self.sim_t >= contact_t:
                self.player_hit(self.pending_swing, self.pending_swing.t - contact_t, contact_pos)
            elif self.sim_t > contact_t + self.window + self.grace:
                self.player_contact = None  # missed: let the ball fly on
        if self.cpu_contact and self.sim_t >= self.cpu_contact[0]:
            contact_pos = self.cpu_contact[1]
            self.cpu.swing(0.6, self.rng.uniform(-0.3, 0.3), contact_height=contact_pos[1])
            miss = self.diff["miss_chance"] + self.diff["hard_shot_miss"] * self.last_player_strength
            if self.rng.random() < miss:
                self.cpu_contact = None
            else:
                self.hit(CPU, contact_pos, opponent_shot(contact_pos, self.diff, contact_pos[1], self.rng))

    def ball_sound(self, event):
        x, speed = self.ball.pos[0], abs(self.ball.vel[1])
        if event == "bounce":
            self.sounds.play("bounce", volume=min(0.4 + speed / 6, 1.0), pan=x / 2)
        elif event == "net":
            self.sounds.play("net", volume=0.8, pan=x / 2)
        elif event == "floor":
            self.sounds.play("floor", volume=0.6, pan=x / 2)

    def on_ball_event(self, event, side):
        if self.state != "rally":
            return
        winner = None
        if event == "bounce":
            winner = self.rally.on_bounce(side)
            if self.rally.landed_in() and self.rally.hitter == PLAYER:
                self.returns += 1
                self.rally_count += 1
        elif event == "floor":
            winner = self.rally.on_floor()
        if winner:
            self.end_point(winner)

    # --- per frame ------------------------------------------------------------

    def update(self, task):
        now = clock.time()
        dt = self.base.clock.getDt()
        self.crowd.update(dt)
        self.sparkles.update(dt)
        while not self.inbox.empty():
            kind, data = self.inbox.get()
            if kind == "result" and self.state == "match_over":
                self.show_final(data)
            elif kind == "swing":
                # Holding an arrow key overrides the paddle's tilt.
                face = self.current_face() or paddle_face(data, self.handed, self.cfg["swing"])
                self.on_swing(Swing(t=data["t_peak"], strength=data["strength"], face=face, side=data["side"]),
                              replaces=data.get("replaces", False))
            elif kind == "paddle_status":
                self.paddle_connected = data
                self.paddle_seen = self.paddle_seen or data
            elif kind == "toss":
                self.on_toss_key()

        if self.state in ("calibrate", "menu"):
            self.update_shirt_color(now)
        if self.state == "calibrate":
            if self.calibration.update():
                self.calibration = None
                self.show_menu()
            self.player.update(dt)
            self.cpu.update(dt)
            self.draw_ball()
            return task.cont

        if self.state == "menu":
            self.menu.update(dt)
            self.camera_view.update(self.sensors.preview_jpeg)
            self.player.update(dt)
            self.cpu.update(dt)
            self.draw_ball()
            return task.cont
        if self.update_pause(now):
            self.player.update(dt)
            self.cpu.update(dt)
            self.draw_hud()
            return task.cont
        if self.menu:                       # the Continue button after a match
            self.menu.update(dt)
        self.update_practice_button(dt)
        self.camera_view.update(self.sensors.preview_jpeg)

        if self.state == "serve":
            self.update_serve(now)
        elif self.state == "point_over" and now >= self.point_over_at:
            if self.score.winner:
                self.end_match()
            else:
                self.start_point()

        if self.held:
            self.sim_t = now
        else:
            while self.sim_t + DT <= now:
                self.sim_t += DT
                if self.state == "rally":
                    self.check_contacts()
                for event, side in self.ball.step():
                    self.ball_sound(event)
                    self.on_ball_event(event, side)
                if not self.player_ready and self.ball.pos[2] > 0:
                    self.player_ready = True    # your shot crossed the net: you get a fresh attempt

        self.update_windups(now)
        ball = to_panda(self.ball.pos)
        self.player.update(dt, look_at=ball)
        self.cpu.update(dt, look_at=ball)
        self.draw_ball()
        self.draw_hud()
        return task.cont

    # --- pausing when the paddle disconnects -------------------------------------

    def update_pause(self, now):
        """Freeze the match while the paddle is disconnected. Returns True while paused."""
        in_match = self.state in ("serve", "rally", "point_over")
        if not self.paused:
            if in_match and self.paddle_seen and not self.paddle_connected:
                self.paused = True
                self.pause_started = now
                self.resume_at = None
                self.pause_overlay.show()
            else:
                return False
        if not self.paddle_connected:
            self.resume_at = None
            self.pause_text.setText("Paddle disconnected\n\nTurn the paddle back on.\nReconnecting...")
            return True
        if self.resume_at is None:
            self.resume_at = now + 3
        left = self.resume_at - now
        if left > 0:
            self.pause_text.setText(f"Paddle reconnected!\n\nResuming in {int(left) + 1}...")
            return True
        self.resume(now)
        return False

    def resume(self, now):
        """Unfreeze: shift every scheduled time by how long the pause lasted."""
        self.shift_times(now - self.pause_started)
        self.paused = False
        self.pause_overlay.hide()

    def shift_times(self, shift):
        """Move the game clock and everything scheduled on it forward by `shift` seconds."""
        self.sim_t += shift
        if self.player_contact:
            self.player_contact = (self.player_contact[0] + shift, self.player_contact[1])
        if self.cpu_contact:
            self.cpu_contact = (self.cpu_contact[0] + shift, self.cpu_contact[1])
        for name in ("cpu_serve_at", "cpu_hit_at", "point_over_at"):
            if getattr(self, name, None) is not None:
                setattr(self, name, getattr(self, name) + shift)
        if self.pending_swing:
            self.pending_swing.t += shift

    def update_windups(self, now):
        """Avatars draw the paddle back as the ball comes to them (like Wii Sports)."""
        for avatar, contact in ((self.player, self.player_contact), (self.cpu, self.cpu_contact)):
            ready = contact is not None and contact[0] - now < WINDUP_LEAD
            avatar.set_windup(1.0 if ready else 0.0)

    def update_serve(self, now):
        server = self.rally.hitter
        if server == PLAYER:
            if self.held:
                self.ball.pos = self.hand_ball_pos(self.player)
            elif self.ball.vel[1] < 0 and self.ball.pos[1] <= self.hand_ball_pos(self.player)[1]:
                self.held = True            # no swing: it falls back into the same hand, try again
                self.message.setText("Raise your free hand to toss, then swing")
            return
        if self.held:
            self.ball.pos = self.hand_ball_pos(self.cpu)
            if now >= self.cpu_serve_at:
                self.toss()
                self.cpu_hit_at = now + CPU_TOSS_TO_HIT
        elif now >= self.cpu_hit_at:
            target_x = self.rng.uniform(-1, 1) * self.diff["target_spread"] * HALF_WIDTH
            strength = self.rng.uniform(0.2, 0.4) + 0.4 * self.diff["target_spread"]
            vel, after = serve_shot(self.ball.pos, CPU, target_x, strength, self.cfg["shots"])
            self.cpu.swing(strength, contact_height=self.ball.pos[1])
            self.cpu.serving = False
            self.state = "rally"
            self.hit(CPU, self.ball.pos, vel, after_bounce=after, power=strength)

    def draw_ball(self):
        in_play = self.state not in ("menu", "calibrate")
        for np in (self.ball_np, self.shadow):
            np.show() if in_play else np.hide()
        x, y, z = self.ball.pos
        self.ball_np.setPos(to_panda(self.ball.pos))
        over_table = abs(x) <= HALF_WIDTH and abs(z) <= HALF_LENGTH
        ground = TABLE_TOP + 0.003 if over_table and y > TABLE_TOP else 0.004
        self.shadow.setPos(to_panda((x, ground, z)))
        size = max(0.02, 0.06 - 0.02 * (y - ground))
        self.shadow.setScale(size, size, 0.002)

    def draw_hud(self):
        if self.practice:
            self.scoreboard.show("Rally", "Best", self.rally_count, self.best_rally)
        else:
            serving = None if self.state == "match_over" else 0 if self.score.server == PLAYER else 1
            self.scoreboard.show("You", self.opponent_name,
                                 self.score.points[PLAYER], self.score.points[CPU], serving)
        if self.sensors.connected:
            camera = "sees you" if self.sensors.body_seen() else "can't see you"
            paddle = "connected" if self.paddle_connected else "not connected"
            self.paddle_text.setText(f"Paddle: {paddle}\nCamera: {camera}")
        if self.debug_on:
            timing = "-" if self.last_delta is None else f"{self.last_delta * 1000:+.0f} ms"
            self.debug.setText(f"last swing timing: {timing} (window +/-{self.window * 1000:.0f} ms)\n"
                               f"returns: {self.returns}   state: {self.state}")

    # --- keyboard -------------------------------------------------------------

    def key_down(self, button):
        watcher = self.base.mouseWatcherNode   # None in an offscreen (test) window
        return watcher is not None and watcher.isButtonDown(button)

    def current_face(self):
        return 0.5 * self.key_down(KeyboardButton.up()) - 0.5 * self.key_down(KeyboardButton.down())

    def on_swing_key(self, strength):
        self.on_swing(Swing(t=clock.time(), strength=strength, face=self.current_face()))

    def on_toss_key(self):
        if self.paused:
            return
        if self.calibration:
            self.calibration.on_toss()
            return
        if self.state == "serve" and self.rally.hitter == PLAYER and self.held:
            self.toss()

    def on_enter(self):
        if self.calibration:
            self.calibration.on_key("enter")
        elif self.state == "menu":
            self.start_from_menu()
        elif self.state == "match_over":
            self.show_menu()

    def toggle_debug(self):
        self.debug_on = not self.debug_on
        self.debug.show() if self.debug_on else self.debug.hide()
