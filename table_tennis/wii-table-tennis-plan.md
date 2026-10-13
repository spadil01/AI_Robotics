# Wii Sports Resort Table Tennis Recreation: Planning Notes

A record of the brainstorming conversation for recreating Wii Sports Resort table tennis with a LEGO Education Double Motor, AprilTags, MediaPipe pose detection, and MQTT.

---

## 1. Starting Ideas

1. Use the Double Motor as the "Wii Remote", with an AprilTag on one end so it works as a pointer.
2. Use MediaPipe to find the user's pose and the location of the Double Motor, and apply it to the game.
3. Build the game in either Python or Unity + Python (undecided).
4. Make the Double Motor "vibrate" every time the player hits the ball.

---

## 2. Key Findings from the LEGO Education Python API

Source: https://github.com/LEGO/LEGOEducation

- **The Double Motor has a built-in 6-axis IMU.** It exposes yaw, pitch, roll, accelerometer X/Y/Z, gyroscope X/Y/Z, and an IMU gesture value, readable inline or through a notification callback.
- **Python 3.14 minimum.** MediaPipe may not support 3.14 yet, so check compatibility before committing.
- **Connects over Bluetooth Low Energy.** Measure the latency of IMU data and of commands sent to the motor.
- **Built-in feedback:** button light color (`light_color`) and beeps (`beep`).
- **Motor data** includes position and a per-motor gesture value, so a free motor axle could serve as an input knob or button.

Useful API calls:

```python
import legoeducation as le

doublemotor = le.DoubleMotor()
doublemotor.connect(card_color=le.LEGO_COLOR_AZURE, card_serial='XXXX')

doublemotor.imu_device          # orientation, yaw/pitch/roll, accel, gyro
doublemotor.imu_gesture         # motion gestures
doublemotor.imu_reset_yaw_axis(0)
doublemotor.light_color(le.LEGO_COLOR_BLUE, pattern=le.LIGHT_PATTERN_BREATHE, intensity=100)
doublemotor.beep(pattern=le.SOUND_PATTERN_BEEP_SINGLE, frequency=440)
doublemotor.set_notification_callback(callback)
```

---

## 3. Design Insights

### Resort's design fits this hardware

Table tennis is in **Wii Sports Resort** (not the original Wii Sports) and used the MotionPlus gyroscope. The character moves to the ball automatically; the player controls:

- **Swing timing** (early/late sets left/right direction)
- **Paddle face angle** (sets height and arc of the return)
- **Swing speed** (sets pace)
- **Wrist rotation during the swing** (sets topspin/backspin)

So precise 3D paddle tracking isn't needed. Good orientation and swing data from the IMU is enough.

### AprilTag + IMU: each does what it's good at

- **AprilTag:** menu pointer (like the Wii's IR pointer), rough absolute position, and correcting IMU yaw drift when the paddle is still. Weak during fast swings because of motion blur.
- **IMU:** swing detection (gyro spikes), swing speed, and paddle angle at contact. Fast and unaffected by blur, but yaw drifts (no magnetometer).
- A simple complementary filter is enough to combine them.

### What MediaPipe is for

MediaPipe finds the **person**, not the motor (the AprilTag locates the motor). Possible uses:

- Wrist position as a backup when the tag isn't detected
- Torso position to move the avatar left/right
- Forehand vs. backhand from which side of the body the wrist is on
- Detecting the serve toss with the off hand

### Hit detection

Don't require an exact 3D collision. Predict when the ball reaches the player, open a timing window, and count a hit if a swing spike lands inside it. Compute the return from timing, paddle pitch, swing speed, and gyro roll rate (spin via a simple Magnus-effect term).

---

## 4. Decisions Made

### Feedback: beeps instead of vibration

Since the motor is being built into an actual paddle, spinning the motors for rumble isn't practical. **Decision: use the beep to signal a hit.** To make it effective:

- The beep will be faint, so make the main feedback happen in the game: a solid "tock" sound, a few frames of hit-stop, slight camera shake, and particles.
- Flash the button light on each hit (different color for a lost point).
- Send the beep as soon as Python detects the swing in the timing window, not after a round trip to the game.
- Vary beep pitch with swing strength (higher for a smash).

### AprilTag role

**Confirmed: the AprilTag is the menu pointer.**

### Game engine: Python + Ursina

- Initially recommended **Unity + Python** for the Resort-like visuals. Note: Unity C# scripts can be edited in VS Code with Claude Code, and community MCP bridges (MCP for Unity by CoplayDev, CoderGamester's mcp-unity, IvanMurzak's Unity-MCP) let Claude Code control the Unity Editor.
- **Final decision, given the 6-day timeline and rusty Unity skills: Python + Ursina.** One language, one editor, Claude Code helps with every part, no time lost relearning Unity. The Resort feel comes mostly from bright colors, simple rounded shapes, sound, and snappy hit feedback.
- Visuals should approximate Resort's *style*; don't use Nintendo's actual models, Miis, maps, or music.

---

## 5. Architecture

Separate processes communicating through a local **Mosquitto** MQTT broker. This also lets the LEGO process run on Python 3.14 while MediaPipe and the game run on whatever version they support.

| Topic | Contents | QoS |
|---|---|---|
| `paddle/imu` | High-rate orientation and gyro data | 0 |
| `paddle/tag` | AprilTag pose / pointer position | 0 |
| `player/pose` | Selected MediaPipe landmarks | 0 |
| `game/event/hit` | Hit events; LEGO process beeps and flashes the light | 1 |

Processes:

1. **LEGO process (Python 3.14):** connects to the Double Motor, publishes IMU data, detects swings, subscribes to hit events for beep/light feedback.
2. **Vision process:** webcam, AprilTag detection (`pupil-apriltags`), MediaPipe pose; publishes tag and pose data.
3. **Game process (Ursina):** subscribes to everything, runs the game.

---

## 6. Building the Paddle

- **Mount the motor rigidly.** Wobble shows up as gyro noise and false swing detections.
- **Remap the IMU axes** to match the paddle in one Python function so the game doesn't care how the motor is mounted.
- **Put a tag on each face** with different IDs; this also tells you which side faces the camera.
- **Pointing:** either ask players to aim the paddle face at the screen in menus, or compute the pointing ray from the tag orientation plus a fixed offset. A quick calibration (point at two or three screen corners) improves accuracy.
- **Smooth the cursor** (a One Euro filter works well) and select by hovering for about a second or by a gyro-detected flick.
- **Calibration screen at startup:** "hold the paddle facing the screen" to zero the IMU yaw and record the tag's neutral pose.
- **Battery:** charge every night; handle disconnects gracefully (pause and show "reconnect paddle").

---

## 7. Six-Day Schedule

| Day | Goal | Details |
|---|---|---|
| 1 | Eliminate risks | Install `legoeducation` on 3.14 and connect. Measure IMU update rate and beep delay. Install OpenCV, `pupil-apriltags`, MediaPipe, Ursina and confirm working Python versions. Run Mosquitto. Build the physical paddle. |
| 2 | Sensor layer | IMU publisher with axis remap and swing detection (gyro spike + cooldown). AprilTag pointer publisher with smoothing. Hit-event subscriber that beeps and flashes. Goal: swinging prints "SWING" reliably and nothing else. |
| 3 | Core game | Table, net, ball physics (gravity, bounce). Auto-positioned player paddle, timing-window hit check, return direction from timing and paddle angle. Simple opponent that returns the ball with randomness. Keyboard fallback. |
| 4 | Full game loop | Serving, scoring to 11, win screen, start menu with the AprilTag pointer. **Fully playable by end of day: this is the real deadline.** |
| 5 | Polish | Colors, lighting, sounds, hit-stop, camera framing, background. Spin and MediaPipe forehand/backhand only if everything else works. |
| 6 | Buffer | Tuning, playtesting with others, bug fixes, demo prep. No new features. |

### Cut list (in order, if behind)

1. **MediaPipe.** If required, use it for one simple thing (forehand/backhand, or a skeleton overlay on the menu).
2. **Spin physics.** Speed and direction are enough.
3. **Pointer menus.** Fall back to "swing to start".
4. **Opponent difficulty levels and animations.**

### Time savers

- **Keyboard fallback** (spacebar = swing, arrows = paddle angle) to test without the paddle.
- **Record swings** from the MQTT stream to a file on day 2 and replay them while tuning.
- **One config file** for all tunable numbers: swing threshold, timing window width, ball speed, gravity.
- **Charge the motor every night.**

---

## 8. Biggest Risk

**Swing detection feeling right.** A game with basic graphics and great hit timing beats a beautiful game where swings register inconsistently. Measure Bluetooth latency and IMU update rate first: those numbers decide how responsive everything else can be.
