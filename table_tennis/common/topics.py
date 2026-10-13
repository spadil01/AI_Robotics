"""MQTT topic names shared by every process."""

PADDLE_IMU = "paddle/imu"
PADDLE_SWING = "paddle/swing"
PADDLE_STATUS = "paddle/status"

VISION_TAG = "vision/tag"
VISION_POSE = "vision/pose"
VISION_TOSS = "vision/toss"
VISION_PREVIEW = "vision/preview"   # small JPEG of the camera view (binary)
VISION_SHIRT = "vision/shirt"       # the player's main shirt color, found with k-means

GAME_STATE = "game/state"
GAME_EVENT = "game/event"
GAME_CONFIG = "game/config"         # settings chosen in the game, e.g. handedness (retained)
