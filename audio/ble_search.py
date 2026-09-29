"""Fast Bluetooth lookup for LEGO devices.

legoeducation's `search(timeout=N)` always scans for the full N seconds,
even if the device shows up in the first half-second. `search(timeout=0)`
is its "first match" mode instead: it returns as soon as a matching device
is seen, but gives up after a fixed 10s. `find_first` repeats first-match
scans until `timeout_s` runs out, so connecting takes about as long as the
device takes to advertise while still honoring BLE_SCAN_TIMEOUT_S.
"""

import time


def find_first(device, timeout_s, card_color, card_serial):
    """Return the first matching BLE device for `device` (a legoeducation
    device object), or None if nothing matched within `timeout_s` seconds."""
    deadline = time.monotonic() + timeout_s
    while True:
        found = device.search(timeout=0, card_color=card_color, card_serial=card_serial)
        if found:
            return found[0]
        if time.monotonic() >= deadline:
            return None
