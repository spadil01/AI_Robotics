"""Talk to the safe's UNO Q (the impenetrable-safe app must be running on it).

The board does no thinking: we tell it which lights to turn on and where to
move the servo, and it tells us when the button is pressed.
"""

import queue
import shutil
import socket
import subprocess
import threading

from safe_rules import GREEN, INSIDE, YELLOW  # noqa: F401  (pins, kept here for old imports)

PORT = 7100
BOARD_IP = "10.247.137.83"   # used only when the board isn't plugged in over USB


def connect(board_ip=BOARD_IP, port=PORT):
    """Open a connection to the board: over USB if it's plugged in, else over WiFi.
    Returns (socket, "USB" or the IP)."""
    # Over USB, ask adb to pass this laptop port through the cable to the board.
    if board_ip not in ("localhost", "127.0.0.1") and shutil.which("adb"):
        devices = subprocess.run(["adb", "devices"], capture_output=True, text=True).stdout
        if any(line.strip().endswith("device") for line in devices.splitlines()[1:]):
            subprocess.run(["adb", "forward", f"tcp:{port}", f"tcp:{port}"],
                           check=True, capture_output=True)
            return socket.create_connection(("localhost", port), timeout=3), "USB"
    return socket.create_connection((board_ip, port), timeout=3), board_ip


class Board:
    def __init__(self, board_ip=BOARD_IP):
        self.sock, self.how = connect(board_ip, PORT)
        self.presses = queue.Queue()
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        self.sock.settimeout(None)
        for line in self.sock.makefile():
            if line.strip() == "button":
                self.presses.put(True)

    def _send(self, text):
        self.sock.sendall((text + "\n").encode())

    def light(self, pin, on):
        self._send(f"light {pin} {1 if on else 0}")

    def servo(self, angle):
        self._send(f"servo {angle}")

    def button_pressed(self):
        """True once for every button press since the last time you asked."""
        try:
            self.presses.get_nowait()
            return True
        except queue.Empty:
            return False

    def close(self):
        self.sock.close()
