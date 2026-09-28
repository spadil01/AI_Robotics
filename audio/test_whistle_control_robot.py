from whistle_control_robot import _message_matches, prompt_role, subscribe_for_start


class FakeMQTTClient:
    """Records subscribe()/publish() calls instead of talking to a real broker."""

    def __init__(self):
        self.subscriptions = {}
        self.published = []

    def subscribe(self, topic, callback, qos=0):
        self.subscriptions[topic] = callback

    def publish(self, topic, message, qos=0, retain=False):
        self.published.append((topic, message))


# ---- _message_matches ----


def test_message_matches_exact():
    assert _message_matches("start", "start") is True


def test_message_matches_is_case_insensitive():
    assert _message_matches("START", "start") is True


def test_message_matches_trims_whitespace():
    assert _message_matches("  start  ", "start") is True


def test_message_matches_rejects_unrelated_text():
    assert _message_matches("not start", "start") is False


# ---- prompt_role ----


def test_prompt_role_accepts_ball(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "ball")
    assert prompt_role() == "ball"


def test_prompt_role_accepts_goalie_case_insensitive_and_trims(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "  Goalie  ")
    assert prompt_role() == "goalie"


def test_prompt_role_reprompts_on_invalid_input(monkeypatch):
    answers = iter(["nonsense", "ball"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert prompt_role() == "ball"


# ---- subscribe_for_start ----


def test_subscribe_for_start_sets_event_on_start_message():
    client = FakeMQTTClient()
    started = subscribe_for_start(client)
    on_message = next(iter(client.subscriptions.values()))

    assert not started.is_set()
    on_message("ignored/topic", "start")
    assert started.is_set()


def test_subscribe_for_start_is_case_insensitive_and_trims_whitespace():
    client = FakeMQTTClient()
    started = subscribe_for_start(client)
    on_message = next(iter(client.subscriptions.values()))

    on_message("ignored/topic", "  START  ")
    assert started.is_set()


def test_subscribe_for_start_ignores_unrelated_messages():
    client = FakeMQTTClient()
    started = subscribe_for_start(client)
    on_message = next(iter(client.subscriptions.values()))

    on_message("ignored/topic", "not start")
    assert not started.is_set()
