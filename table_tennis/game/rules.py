"""
Table tennis rules: who wins a rally, the score, and whose serve it is.
"""

from .physics import CPU, PLAYER, other


class Rally:
    """Referee for one point. Feed it hits, bounces and floor hits; each
    returns the winner ("player" / "cpu") once the point is decided, else None."""

    def __init__(self, server):
        self.hitter = server
        self.serving = True
        self.bounces = []   # sides the ball bounced on since the last hit

    def on_hit(self, who):
        self.hitter = who
        self.serving = False
        self.bounces = []

    def on_bounce(self, side):
        self.bounces.append(side)
        h, receiver = self.hitter, other(self.hitter)
        if self.serving:
            # Serve: server's half, then receiver's half, then receiver must hit it.
            if self.bounces[0] != h:
                return receiver
            if len(self.bounces) >= 2 and self.bounces[1] != receiver:
                return receiver
            if len(self.bounces) >= 3:
                return h
            return None
        # Rally: must land on the receiver's half, and the receiver must hit it
        # before it bounces again.
        if self.bounces[0] != receiver:
            return receiver
        if len(self.bounces) >= 2:
            return h
        return None

    def on_floor(self):
        h, receiver = self.hitter, other(self.hitter)
        good_bounces = 2 if self.serving else 1
        if len(self.bounces) >= good_bounces:
            return h          # landed in, receiver didn't return it
        return receiver       # never landed in

    def landed_in(self):
        """True right after a legal (non-serve) bounce on the receiver's half."""
        return not self.serving and self.bounces == [other(self.hitter)]


class Score:
    def __init__(self, match_points, serves_per_turn, first_server=PLAYER):
        self.match_points = match_points
        self.serves_per_turn = serves_per_turn
        self.first_server = first_server
        self.points = {PLAYER: 0, CPU: 0}

    def add(self, winner):
        self.points[winner] += 1

    @property
    def deuce(self):
        return min(self.points.values()) >= self.match_points - 1

    @property
    def server(self):
        played = sum(self.points.values())
        turns = played if self.deuce else played // self.serves_per_turn
        return self.first_server if turns % 2 == 0 else other(self.first_server)

    @property
    def winner(self):
        p, c = self.points[PLAYER], self.points[CPU]
        if max(p, c) >= self.match_points and abs(p - c) >= 2:
            return PLAYER if p > c else CPU
        return None
