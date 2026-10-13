# Table Tennis: Detailed Implementation Plan

Builds on [wii-table-tennis-plan.md](wii-table-tennis-plan.md). Stack: Python + Ursina, LEGO Double Motor paddle, AprilTag pointer, MediaPipe pose (required), local MQTT.

---

## 0. What the existing repo already tells us

- **One Python environment probably covers everything.** The repo `.venv` is Python 3.13.5 with `legoeducation 1.1.1`, `mediapipe 0.10.35`, `paho-mqtt 2.1.0` and `opencv-contrib-python 5.0` installed together, and `hand_control/hand_control_robot.py` already uses LEGO + MediaPipe in one process. The "3.14 minimum" worry from the first notes doesn't apply to this setup. We still split into processes (see §2), but not for version reasons.
- **AprilTags:** `apriltag_centering/` detects tags with OpenCV's `cv2.aruco` AprilTag dictionaries, not `pupil-apriltags` (which isn't installed in this venv). Reuse whichever detector you trust more; `cv2.aruco` is already proven here.
- **MediaPipe:** past work uses the **Tasks API** (`HandLandmarker`). Pose will use the matching `PoseLandmarker` with a `pose_landmarker_lite.task` / `_full.task` model file.
- **MQTT:** `mqttlib.py` defaults to the public broker `test.mosquitto.org`. For a game, use a **local Mosquitto** (`brew install mosquitto`) and pass `broker="localhost"`. A public internet broker adds tens to hundreds of ms of jitter, and anyone can publish to your topics.
- **New:** Ursina / Panda3D. Install it into the same venv on day 1 and check that it opens a window on macOS.

---

## 1. Gameplay design (what the player actually does)

| Action | Input | Source |
|---|---|---|
| Navigate menus | Move the paddle; tag center = cursor | AprilTag |
| Select menu item | Hover ~1 s (dwell), or a quick flick | AprilTag + IMU |
| Serve | Raise the free hand (toss), then swing | **MediaPipe** + IMU |
| Hit the ball | Swing during the timing window | IMU |
| Direction (left/right) | Swing early vs. late | IMU timestamp |
| Height / arc | Paddle face angle at contact | IMU pitch |
| Pace | Swing speed | IMU gyro peak |
| Spin | Wrist roll during the swing | IMU gyro roll rate |
| Forehand vs. backhand | Paddle wrist left/right of body midline | **MediaPipe** (gyro sign as cross-check) |
| Avatar position / lean | Shoulders and hips | **MediaPipe** |

### MediaPipe's role (required, so it does real work)

Ranked by value vs. effort:

1. **Serve toss (core):** a serve starts only when the free hand's wrist rises above that shoulder. It's very visible in a demo and easy to detect.
2. **Paddle-hand / handedness detection (core):** during calibration, the wrist closest to the tag center is the paddle hand. This sets which hand is the toss hand and mirrors forehand/backhand for left-handers.
3. **Forehand vs. backhand (core):** at the swing moment, compare the paddle wrist's x position with the shoulder midpoint. This picks the avatar's animation and which side of the body the ball is played on. The gyro rotation sign confirms it, so a bad pose frame doesn't ruin a shot.
4. **Avatar mirroring (polish):** shoulder midpoint drives the avatar's small left/right shift and lean. The character still auto-moves to the ball, as in Resort.
5. **Tag-loss fallback (polish):** if the tag isn't visible in menus, the paddle wrist becomes the cursor.
6. **Skeleton overlay (polish):** small picture-in-picture webcam feed with the skeleton on the menu and calibration screens.

1–3 are the minimum MediaPipe features; 4–6 are day-5 polish.

### Left- and right-handed support

Handedness is detected during calibration (item 2 above) and shown on screen ("Right-handed — swing to confirm, raise other hand to switch"). It can also be changed in the menu. One `handedness` value then flips everything that depends on it:

- IMU yaw sign for forehand vs. backhand (the same physical swing rotates the opposite way for a lefty)
- which hand is the toss hand
- which side of the body counts as forehand in the pose check
- the avatar's paddle hand and its swing animation (mirrored)

Keep every handedness flip in one place (a small `Handedness` helper) so nothing gets missed.

---

## 2. Architecture

Three processes plus a broker, all on one laptop:

```
 Double Motor ──BLE──> [paddle_node]  ──paddle/*──┐
                        ^   (beep, light)          │
                        └──── game/event ──────┐   v
 Webcam ─────────────> [vision_node]  ──vision/*──> [Mosquitto localhost] <──> [game (Ursina)]
```

**Why separate processes, even with one venv:** the BLE event loop, the camera/MediaPipe loop and the render loop each want to own their thread. Crashes stay contained (a BLE disconnect doesn't kill the game), and each node can be tested alone with recorded data.

- **paddle_node:** connects to the Double Motor, remaps IMU axes to paddle axes, runs swing detection, publishes IMU + swing events, and listens for game events to beep and flash the light.
- **vision_node:** one webcam, one frame loop. It runs AprilTag detection on every frame and MediaPipe pose on every frame (or every second frame if it's slow). It also detects the toss gesture and handedness.
- **game:** Ursina. Subscribes to everything and runs the state machine, ball simulation, hit resolution, opponent, scoring, UI and audio.

### Timestamps

Every message carries `t = time.time()` taken when the sample was **captured** (IMU callback or camera frame grab), not when it was sent. All processes share the same clock, so the game can compare a swing's timestamp with the ball's trajectory directly (see §4).

### Topics and payloads (JSON)

| Topic | From → To | Rate | QoS | Payload |
|---|---|---|---|---|
| `paddle/imu` | paddle → game | IMU rate | 0 | `{t, yaw, pitch, roll, gx, gy, gz}` (already remapped to paddle axes) |
| `paddle/swing` | paddle → game | per swing | 1 | `{t_peak, peak_rate, pitch, roll_rate, side}` |
| `paddle/status` | paddle → game | on change, retained | 1 | `{connected, battery?}` |
| `vision/tag` | vision → game | camera fps | 0 | `{t, visible, id, cx, cy}` (cx, cy normalized 0–1) |
| `vision/pose` | vision → game | camera fps | 0 | `{t, visible, paddle_hand, lw, rw, ls, rs, lh, rh, nose}` (normalized x, y) |
| `vision/toss` | vision → game | per event | 1 | `{t, hand}` |
| `game/state` | game → nodes | on change, retained | 1 | `{state}` (menu / calibrate / play …) |
| `game/event` | game → paddle | per event | 1 | `{type: hit/miss/point_won/point_lost/match_won, strength}` |
| **`ME193/Rogers/<name>`** | game → class (**test.mosquitto.org**) | once per match | 1 | Plain float string only, e.g. `23.0` (see below) |

`game/state` lets the nodes change behavior, e.g. vision_node skips toss detection while in menus.

### Required: match score broadcast

**This is a hard requirement**; the other topics are just how the processes talk to each other.

- **What:** the total number of **returns by the player during the whole match**, as a floating point number. The payload is only that number as text, e.g. `23.0`: no JSON, no winner, no other fields.
- **Where:** topic `ME193/Rogers/<name>`, where `<name>` is a fixed `player_name` in `config.toml`. Broker: **`test.mosquitto.org`**, port 1883, the class broker already used by `mqttlib.py`.
- **When:** once, when the match ends, **before** the win screen appears. QoS 1, **not retained** (a retained message would be re-sent to every new subscriber, who might count an old match again).
- **What counts as a return:** a player hit that clears the net and lands on the opponent's side. Missed swings, balls into the net or off the table, and **serves don't count**: a point only starts once the serve is done correctly, so counting begins with the first return after it. The count keeps running across points (it never resets mid-match).
- The count is **not shown during play**. It appears only on the end-of-match screen ("Total returns: 23"), next to the final score.
- **Two MQTT connections in the game:** the local broker for sensor data (low latency), and a separate client to `test.mosquitto.org` that only publishes the score. The public broker's delay doesn't matter for a single message at the end of a match.
- Practice mode doesn't publish (not a match).
- If the publish fails (no internet / broker down), the win screen shows a warning and retries in the background, so the demo doesn't freeze. Log every published score to a local file as a backup.
- The winner is shown on screen only; it is **not** published.

### Is MQTT needed for the sensors?

No. The sensor nodes could instead be threads inside the game process, sending data through in-memory queues. The multi-process MQTT design is still recommended, because crashes stay contained and recording/replay for tuning is free. Since the game needs an MQTT client for the result anyway, it costs nothing extra.

### Threading rule in the game

paho callbacks run on paho's background thread. **Never touch Ursina entities from a callback.** Callbacks only push messages into a `queue.Queue`; Ursina's `update()` drains the queue every frame. This is the most common bug in this kind of setup.

---

## 3. Sensor layer details

### 3.1 IMU axis remap
- One function maps the motor's native axes to paddle axes: **roll** = rotation around the handle, **pitch** = face tilt (open/closed), **yaw** = left/right swing.
- Find the mapping on day 1 by rotating the paddle around each axis and watching which gyro channel spikes. Store it in config, not code.

### 3.2 Swing detection (in paddle_node)
- Signal: the gyro magnitude around the swing axes, |ω|.
- Two thresholds (hysteresis): a swing **starts** when |ω| > `SWING_START`, **ends** when it drops below `SWING_END`. Record the peak and its timestamp in between.
- **Cooldown** of about 300 ms after each swing, so the follow-through doesn't count as a second swing.
- Ignore swings shorter than a minimum duration (filters out bumps and wobble).
- Output on swing end: `t_peak`, `peak_rate` (pace), `pitch` at peak (face angle), `roll_rate` at peak (spin), `side` from the yaw-rate sign (fore/back hint).
- Beep + light flash **immediately** on detection (local, no round trip). The game can still send `game/event` for a hit/miss color afterwards.
- Beep pitch scales with `peak_rate`.

### 3.3 AprilTag pointer (in vision_node, mapped in game)
- Publish the raw normalized tag center `(cx, cy)`. Do the mapping in the game, so it can be tuned without restarting the camera.
- Calibration records the tag's comfortable range of motion (min/max x and y). The cursor maps that box to the full screen, mirrored horizontally so it feels like a mirror.
- Smooth with a **One Euro filter**: steady when still, responsive when moving.
- Selection: a dwell of about 1 s with a visible radial fill. A flick (short gyro spike below swing strength) is optional.
- Tag size: **6–7 cm black square** (36h11) with a white margin of at least one cell, camera at **1280×720**. Matte paper on a stiff backing. The player stands about 1–1.5 m from the Mac; that distance is set by pose framing (§3.4), not by the tag.
- Good lighting, and a short exposure if the webcam allows it.

### 3.4 Pose (in vision_node)
- `PoseLandmarker`, `VIDEO` or `LIVE_STREAM` running mode, **lite** model first; switch to full only if accuracy is poor and fps holds.
- Publish only the landmarks used (wrists, shoulders, hips, nose), not all 33.
- **Framing:** a laptop camera sees only about 0.8 m of height at 1 m away. Don't rely on hips; use shoulders for the body midline. The toss check needs the raised wrist in frame, so the calibration screen shows the webcam view with a "step back until your raised hand fits" guide.
- **Toss detector:** free-hand wrist y above that shoulder's y for N consecutive frames, then a cooldown.
- **Paddle hand:** during calibration, the wrist nearest the tag center over a few seconds wins.
- If camera fps drops, run pose every other frame; tags every frame.

---

## 4. Game layer details

### 4.1 State machine
```
CONNECTING → CALIBRATE → MENU ─┬→ MATCH_SETUP (difficulty, 6 or 11 points) → SERVE → RALLY → POINT_OVER → (SERVE | MATCH_OVER) → MENU
                               │                                              ↑______________________________|
                               └→ PRACTICE (stretch goal)
Any state → PAUSED_DISCONNECTED when paddle/status says disconnected
MATCH_OVER: publish the return count to ME193/Rogers/<name> first, then show the win screen
```

- **MATCH_SETUP:** pick Beginner / Intermediate / Pro, then 6- or 11-point match, all with the pointer.

- **CONNECTING:** wait for `paddle/status` connected and for tag/pose messages to arrive. Show a checklist (paddle ✓, camera ✓, pose ✓).
- **CALIBRATE:** "Hold the paddle facing the screen" → zero yaw. "Move the paddle around the screen edges" → pointer box. Handedness is detected along the way.
- **SERVE:** the player's serve waits for a toss, then a swing. The opponent serves after a short delay.

### 4.2 Ball simulation
- A fixed timestep (e.g. 120 Hz sub-steps inside Ursina's frame), not tied to frame rate.
- Gravity, a small drag term, table bounce with restitution, net collision as a simple height check at the net plane, and a small Magnus term from spin.
- Use real-ish table dimensions (2.74 m × 1.525 m, net 15.25 cm), then tune speed for fun over realism.

### 4.3 Hit resolution (the core of "feel")
1. When the ball is coming toward the player, the game predicts **t_contact**: when the ball crosses the player's hitting plane.
2. It opens a window `[t_contact − W, t_contact + W]`, with W tuned from the measured latency (start around 150–200 ms total).
3. When `paddle/swing` arrives, compute `Δ = t_peak − t_contact`, using the swing's **own timestamp**.
4. If |Δ| ≤ W, it's a hit. Because the message arrives late by BLE latency, the game **waits a short grace period** (latency + margin) after t_contact before declaring a miss. If the swing arrives during the grace period, the hit is applied from the contact point. A few frames of hit-stop hide the small jump.
5. **Aim by target, not raw physics.** Turn the inputs into a **target landing point** on the opponent's side, then solve for the launch velocity that reaches it:
   - Δ (early/late) → target x (cross-court vs. down the line)
   - pitch → arc height / depth; extreme face angles push the target into the net or off the end
   - `peak_rate` → speed (flight time)
   - `roll_rate` → spin, which bends the arc a little through the Magnus term
   - Good swings land on the table; bad ones miss in understandable ways. This is how the Wii game feels fair.

### 4.4 Opponent and difficulty
- Uses the same target-point solver: when the ball reaches its side, pick a target on the player's side.
- The opponent avatar moves to the ball's predicted position (animation only).
- The player chooses the level before each match. All numbers live in `config.toml` under `[difficulty.beginner]` etc.

| Parameter | Beginner | Intermediate | Pro |
|---|---|---|---|
| Return speed | slow | medium | fast |
| Targets | near the middle | some corners | corners and lines |
| Spin on returns | none | light | heavy |
| Miss chance on a hard shot from the player | high | medium | low |
| Smashes a high ball | never | sometimes | usually |
| Player's timing window W | widest | normal | narrowest |
| Serve variety | one simple serve | two | fast and spin serves |

Widening the player's timing window on Beginner matters as much as making the opponent weaker: it makes the game feel easier to control.

**Practice mode (stretch goal):** the opponent never misses and feeds balls at an adjustable pace. No score and no broadcast. Since it's the same game with different settings, it costs very little once the match works, and it's also the best way to tune swings.

### 4.5 Rules (as in Wii Sports Resort)
- **Match length:** choose a **6-point or 11-point** match, as Resort offers. You must win by 2 (deuce).
- **Serve rotation:** standard table tennis: 2 serves each, then alternate every point at deuce (5–5 or 10–10). I couldn't confirm that Resort rotates exactly this way, so it's a config setting.
- **Serve:** real rule, where the ball bounces once on the server's side, then once on the receiver's side. In the game this is automatic: the target-point solver aims the serve at the server's side first. The player only needs the toss (MediaPipe) + swing.
- **Point:** a point ends when the ball bounces twice, misses the table, or hits the net and doesn't go over.

### 4.6 Visuals and audio (Resort-like style, original assets)
- Bright sky, warm directional light + ambient light, a rounded blue table with a white line.
- Camera behind and above the player, looking down the table.
- On hit: "tock" sound, 3–5 frames of hit-stop, a little camera shake, a particle burst, and paddle beep + light flash.
- Webcam picture-in-picture with the skeleton in menus/calibration (MediaPipe visibility for the demo).
- **One window only on demo day:** vision_node runs headless (no OpenCV window) when started by `run_all.py`; the game shows the camera view itself. Build this together with the calibration screen.

### 4.7 Avatars (people, not floating paddles)

Two options:

| | **A. Built from shapes (recommended)** | B. Imported rigged models |
|---|---|---|
| What | Original stylized person made of Ursina primitives: sphere head with simple face, rounded body, two-part arms, hands, legs | Free low-poly human models with skeletons (e.g. CC0 packs like Quaternius), loaded through Panda3D's `Actor` |
| Animation | Code-driven: rotate arm/body parts each frame | Uses the model's own animations |
| Risk | Low. Only basic Ursina features | Medium–high. glTF loading, rig and animation import into Panda3D/Ursina is the most likely place to lose a day |
| Looks | Friendly and clean; fits the bright Resort style | More detailed, but mixing styles can look odd |

**Recommendation: A**, because code-driven animation enables something imported animations can't do well:

- **The player avatar's paddle arm follows the real paddle.** Paddle orientation from the IMU sets the avatar's forearm/hand rotation every frame, so the on-screen paddle tilts and swings when yours does.
- **The body follows MediaPipe:** shoulder midpoint → small step/lean, forehand/backhand → body turn.
- The opponent uses the same body, with scripted swing animations and a different color scheme.
- Small extras: idle bob, a celebration and a disappointed pose after points, eyes that follow the ball.
- Design original characters; don't copy Miis.

If A is done early, B can be tried as a day-5/6 experiment on a branch. Don't switch to it unless it's clearly better.

---

## 5. Proposed file layout

```
table_tennis/
  config.toml               # all tunable numbers: thresholds, window W, ball speed, gravity, axis map, tag ID…
  run_all.sh                # starts mosquitto (if needed), paddle_node, vision_node, game
  common/
    topics.py               # topic name constants + message helpers
    filters.py              # One Euro filter
    (reuse ../mqttlib.py with broker="localhost")
  paddle_node.py
  vision_node.py
  game/
    main.py                 # Ursina app + state machine
    inbox.py                # thread-safe queue from MQTT callbacks
    ball.py                 # ball simulation
    shot.py                 # target-point solver (used by player and opponent)
    hit.py                  # timing window + hit resolution
    opponent.py
    ui.py                   # menus, cursor, scoreboard, calibration screens
    effects.py              # hit-stop, shake, particles, sounds
  tools/
    record.py               # save the MQTT stream to a file
    replay.py               # play a recording back into MQTT at real speed
    latency_test.py         # IMU rate, beep delay, end-to-end swing delay
    fake_sensors.py         # keyboard → fake paddle/vision messages
  assets/sounds/
```

---

## 6. Six-day schedule (with MediaPipe as required)

| Day | Goal | Done when |
|---|---|---|
| **1** | Risks + measurements | Ursina opens a window in the venv. Mosquitto runs on localhost. `latency_test.py` reports the IMU rate and beep delay. Paddle built, motor mounted rigidly, tag on each face. Axis map found. |
| **2** | Sensor nodes | paddle_node prints/publishes clean swings (no doubles, no false triggers when walking around). vision_node publishes tag + pose at a stable fps. Toss detector works. `record.py` captures a session of real swings. |
| **3** | Core game | Table, net, ball physics, opponent returns, timing-window hits from replayed or keyboard swings. Target-point solver working. Basic shape-built avatars (body + paddle arm following the IMU). |
| **4** | Full loop (**the real deadline**) | Calibration with handedness, pointer menu, difficulty + match-length select, toss-to-serve, Resort scoring (6/11, deuce), win screen, **return count published to `ME193/Rogers/<name>`**, disconnect pause. Fully playable with the real paddle, left- and right-handed. |
| **5** | Polish + MediaPipe extras | Tune the 3 difficulty levels, forehand/backhand via pose, avatar lean/celebrations, skeleton picture-in-picture, sounds, hit-stop, lighting, camera. Practice mode if time allows. |
| **6** | Buffer | Tuning with other people, bug fixes, demo script. No new features. |

### Revised cut list

**Never cut:** MediaPipe core (toss, handedness, forehand/backhand), left/right-handed support, human avatars, 3 difficulty levels, return-count broadcast to `ME193/Rogers/<name>`.

Cut in this order if behind:
1. Practice mode
2. Spin physics
3. Avatar extras (celebrations, eyes following the ball); keep the body and paddle arm
4. Skeleton picture-in-picture / avatar lean
5. Dwell selection → "swing to select"
6. Difficulty differences beyond speed, accuracy and timing window (spin, smashes, serve variety)

---

## 7. Day-1 measurements to record

| Measurement | How | Why it matters |
|---|---|---|
| IMU update rate (Hz) | Count callbacks per second | Sets swing peak precision and window W |
| IMU latency | Tap the paddle on the table next to the mic/camera and compare | Sets the grace period |
| Beep delay | Time from command to audible beep | Decides whether the paddle beep feels like part of the hit |
| Camera fps with tag + pose | Log in vision_node | Pointer smoothness, toss responsiveness |
| MQTT localhost round trip | Publish/echo test | Should be ~1 ms; confirms the broker isn't the bottleneck |

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| Swings register inconsistently | Hysteresis + cooldown + min duration; tune with recorded swings; generous W first |
| BLE latency too high | Use swing timestamps + grace period; keep feedback in-game, with the beep as a bonus |
| Tag lost during motion | Only used in menus, where the paddle moves slowly; pose wrist fallback |
| MediaPipe fps too low alongside tags | Lite model, pose every other frame, lower camera resolution for pose |
| Ursina learning curve | Stick to basic Entities, `update()`, Audio and simple lights; no custom shaders |
| Threading bugs between MQTT and Ursina | Queue-only callbacks (§2) |
| Paddle disconnect mid-demo | Pause state + auto-reconnect; charge every night |

---

## 9. Decisions

### Made
1. **Handedness:** left- and right-handed, detected automatically at calibration and changeable in the menu (§1).
2. **Avatars:** people, built from shapes, with the paddle arm following the real paddle (§4.7).
3. **Difficulty:** Beginner / Intermediate / Pro chosen before each match; practice mode is a stretch goal (§4.4).
4. **Rules:** as in Wii Sports Resort: 6- or 11-point match, win by 2, real serve bounce (§4.5).
5. **AprilTag detector:** keep `cv2.aruco` (already working in `apriltag_centering/`); try `pupil-apriltags` only if day-1 detection range is poor.
6. **Match score broadcast:** total player returns in the match, posted at match end as a plain float (e.g. `23.0`) to `ME193/Rogers/<name>` on `test.mosquitto.org`. `<name>` is fixed in config. The winner is not published (§2).

7. **Serves don't count as returns**; the return total is shown only on the end-of-match screen.
