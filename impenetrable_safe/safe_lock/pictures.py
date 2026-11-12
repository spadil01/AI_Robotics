"""How a camera picture is prepared for the network. No PyTorch here, so the
UNO Q can use this file too (finger_net.py and the board both import it).

Keeping it in one file means every program prepares pictures the exact same
way. If they didn't match, the network would see "strange" pictures and guess badly.
"""

import cv2

LABELS = ["0", "1", "2", "3", "4", "5", "none"]
SAVE_WIDTH = 256          # collect_data.py saves pictures this wide (256x144)
IMG_W, IMG_H = 128, 72    # the network looks at a smaller 128x72 color picture


def prepare(frame):
    """Camera picture (OpenCV BGR, any size) -> small uint8 picture the network eats.

    Shrinks in the same two steps collect_data.py + training use, so live
    pictures look just like the training pictures.
    """
    h, w = frame.shape[:2]
    if w != SAVE_WIDTH:
        frame = cv2.resize(frame, (SAVE_WIDTH, round(SAVE_WIDTH * h / w)), interpolation=cv2.INTER_AREA)
    return cv2.resize(frame, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
