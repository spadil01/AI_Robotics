import whistle_control_robot
from config import (
    GOALIE_LOSS_MESSAGE,
    GOALIE_LOSS_SONG_URL,
    GOALIE_LOSS_TOPIC,
    GOALIE_SONG_URL,
    GOALIE_TOPIC,
    GOALIE_TRIGGER_MESSAGE,
)
from whistle_control_robot import _message_matches, goalie_outcome, prompt_role, run_goalie, subscribe_for_start


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


# ---- goalie_outcome ----


def test_goalie_outcome_save():
    assert goalie_outcome("t/save", " STOP ", "t/save", "stop", "t/ball", "Goal!") == "save"


def test_goalie_outcome_goal():
    assert goalie_outcome("t/ball", "goal!", "t/save", "stop", "t/ball", "Goal!") == "goal"


def test_goalie_outcome_needs_matching_topic():
    # Right message on the wrong topic doesn't count.
    assert goalie_outcome("t/ball", "stop", "t/save", "stop", "t/ball", "Goal!") is None
    assert goalie_outcome("t/save", "Goal!", "t/save", "stop", "t/ball", "Goal!") is None


def test_goalie_outcome_shared_topic():
    # Both messages on one topic still map to the right outcome.
    assert goalie_outcome("t", "stop", "t", "stop", "t", "Goal!") == "save"
    assert goalie_outcome("t", "Goal!", "t", "stop", "t", "Goal!") == "goal"
    assert goalie_outcome("t", "hello", "t", "stop", "t", "Goal!") is None


# ---- run_goalie ----


def _run_goalie_with(monkeypatch, messages):
    """Run run_goalie with a fake MQTT client and a fake live plot that
    delivers `messages` ((topic, payload) pairs) and then processes one
    chunk. Returns (subscribed topics, URLs opened, on_chunk's result)."""
    client = FakeMQTTClient()
    opened = []
    chunk_result = []

    def fake_live_plot_main(motor, on_chunk, gate):
        for topic, payload in messages:
            client.subscriptions[topic](topic, payload)
        chunk_result.append(on_chunk("stop", 0.0))

    monkeypatch.setattr(whistle_control_robot.live_plot, "main", fake_live_plot_main)
    monkeypatch.setattr(whistle_control_robot.webbrowser, "open", opened.append)
    run_goalie(motor=None, mqtt_client=client, started=None)
    return set(client.subscriptions), opened, chunk_result[0]


def test_run_goalie_subscribes_to_save_and_loss_topics(monkeypatch):
    topics, _, _ = _run_goalie_with(monkeypatch, [])
    assert topics == {GOALIE_TOPIC, GOALIE_LOSS_TOPIC}


def test_run_goalie_save_plays_success_song(monkeypatch):
    _, opened, ended = _run_goalie_with(monkeypatch, [(GOALIE_TOPIC, GOALIE_TRIGGER_MESSAGE)])
    assert ended is True
    assert opened == [GOALIE_SONG_URL]


def test_run_goalie_goal_plays_loss_song(monkeypatch):
    _, opened, ended = _run_goalie_with(monkeypatch, [(GOALIE_LOSS_TOPIC, GOALIE_LOSS_MESSAGE)])
    assert ended is True
    assert opened == [GOALIE_LOSS_SONG_URL]


def test_run_goalie_first_outcome_wins(monkeypatch):
    _, opened, _ = _run_goalie_with(
        monkeypatch,
        [(GOALIE_LOSS_TOPIC, GOALIE_LOSS_MESSAGE), (GOALIE_TOPIC, GOALIE_TRIGGER_MESSAGE)],
    )
    assert opened == [GOALIE_LOSS_SONG_URL]


def test_run_goalie_no_trigger_plays_nothing(monkeypatch):
    _, opened, ended = _run_goalie_with(monkeypatch, [])
    assert ended is False
    assert opened == []
