// Animated stick figure for the live gait (like the LED panel, but smooth).
// drawFigure(canvas, gait, seconds, color): gait is "stand", "walk", "run", "bike";
// anything else ("" or "other") draws a standing figure with a "?".

const FIG = { thigh: 34, shin: 34, torso: 46, head: 11, upper: 24, fore: 22 };
const deg = (d) => (d * Math.PI) / 180;

// A point `len` away from p at `angle` (0 = straight down, positive = forward).
const limb = (p, angle, len) => ({ x: p.x + Math.sin(angle) * len, y: p.y + Math.cos(angle) * len });

// Knee position for a leg from hip to foot (knee bends forward).
function knee(hip, foot, a, b) {
  const dx = foot.x - hip.x, dy = foot.y - hip.y;
  const d = Math.min(Math.hypot(dx, dy), a + b - 0.01);
  const toFoot = Math.atan2(dx, dy);
  const bend = Math.acos((a * a + d * d - b * b) / (2 * a * d));
  return limb(hip, toFoot + bend, a);
}

function drawFigure(canvas, gait, t, color) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.lineCap = ctx.lineJoin = "round";

  const groundY = h - 16;
  const cx = w / 2;
  const speed = { walk: 60, run: 150, bike: 110 }[gait] || 0;  // ground slides this many px/s

  // Ground: dashes sliding left when moving.
  ctx.strokeStyle = "#3a4350";
  ctx.lineWidth = 2;
  ctx.setLineDash([14, 10]);
  ctx.lineDashOffset = (t * speed) % 24;
  ctx.beginPath();
  ctx.moveTo(0, groundY);
  ctx.lineTo(w, groundY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = ctx.fillStyle = color;
  ctx.lineWidth = 5;
  const line = (...pts) => {
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (const p of pts.slice(1)) ctx.lineTo(p.x, p.y);
    ctx.stroke();
  };
  const head = (p) => { ctx.beginPath(); ctx.arc(p.x, p.y, FIG.head, 0, 2 * Math.PI); ctx.fill(); };

  if (gait === "bike") return drawBike(ctx, cx, groundY, t, color, line, head);

  // Walking and running: thighs swing, knees bend, arms swing the other way.
  const style = {
    stand: { hz: 0, swing: 0, bend: 0, lean: 0, arm: 0, elbow: 10, bounce: 0, spread: 7 },
    walk: { hz: 0.9, swing: 26, bend: 25, lean: 3, arm: 22, elbow: 15, bounce: 3 },
    run: { hz: 1.5, swing: 42, bend: 80, lean: 14, arm: 40, elbow: 85, bounce: 8 },
  }[gait] || { hz: 0, swing: 0, bend: 0, lean: 0, arm: 0, elbow: 10, bounce: 0, spread: 7 };
  const phase = 2 * Math.PI * style.hz * t;
  const bob = style.bounce * Math.abs(Math.sin(phase));
  const hip = { x: cx, y: groundY - FIG.thigh - FIG.shin + 2 - bob };
  const lean = deg(style.lean);
  const shoulder = limb(hip, Math.PI + lean, FIG.torso);

  for (const side of [1, -1]) {  // the far leg/arm first, drawn dimmer
    ctx.globalAlpha = side === 1 ? 0.45 : 1;
    const p = phase + (side === 1 ? Math.PI : 0);
    const spread = deg(style.spread || 0) * side;  // standing: legs/arms a little apart so both show
    const thighA = deg(style.swing) * Math.sin(p) + spread;
    // The knee bends most while the leg swings forward (foot off the ground).
    const kneeA = deg(style.bend) * Math.max(0, Math.cos(p));
    const k = limb(hip, thighA, FIG.thigh);
    line(hip, k, limb(k, thighA - kneeA - deg(gait === "stand" ? 0 : 4), FIG.shin));
    const armA = -deg(style.arm) * Math.sin(p) - spread * 1.6;
    const elbow = limb(shoulder, armA, FIG.upper);
    line(shoulder, elbow, limb(elbow, armA + deg(style.elbow), FIG.fore));
  }
  ctx.globalAlpha = 1;
  line(hip, shoulder);
  head(limb(shoulder, Math.PI + lean, FIG.head + 6));

  if (!["stand", "walk", "run", "bike"].includes(gait)) {  // nothing yet, or "other"
    ctx.font = "bold 28px system-ui";
    ctx.textAlign = "center";
    ctx.fillText("?", cx + 46, groundY - 90);
  }
}

function drawBike(ctx, cx, groundY, t, color, line, head) {
  const r = 26;                                   // wheel radius
  const back = { x: cx - 44, y: groundY - r }, front = { x: cx + 44, y: groundY - r };
  const crank = { x: cx - 4, y: groundY - r + 2 };
  const seat = { x: cx - 16, y: groundY - r - 40 };
  const bars = { x: cx + 30, y: groundY - r - 46 };
  const turn = 2 * Math.PI * 1.3 * t;             // pedal turns per second

  // Bike in a dimmer shade.
  ctx.globalAlpha = 0.55;
  ctx.lineWidth = 3;
  for (const wheel of [back, front]) {
    ctx.beginPath();
    ctx.arc(wheel.x, wheel.y, r, 0, 2 * Math.PI);
    ctx.stroke();
    for (let i = 0; i < 3; i++) {  // spokes, turning with the wheel
      const a = turn * 1.6 + (i * Math.PI) / 3;
      line({ x: wheel.x - Math.cos(a) * r, y: wheel.y - Math.sin(a) * r },
           { x: wheel.x + Math.cos(a) * r, y: wheel.y + Math.sin(a) * r });
    }
  }
  line(back, crank, seat, back);
  line(crank, { x: bars.x - 4, y: bars.y + 10 }, front);
  line(seat, { x: bars.x - 4, y: bars.y + 10 }, bars);
  ctx.globalAlpha = 1;

  // Rider: hips on the seat, hands on the bars, feet on pedals going round.
  ctx.lineWidth = 5;
  const hip = { x: seat.x, y: seat.y - 4 };
  const shoulder = limb(hip, Math.PI - deg(32), FIG.torso);
  for (const side of [1, -1]) {
    ctx.globalAlpha = side === 1 ? 0.45 : 1;
    const a = turn + (side === 1 ? Math.PI : 0);
    const foot = { x: crank.x + Math.cos(a) * 13, y: crank.y + Math.sin(a) * 13 };
    line(hip, knee(hip, foot, FIG.thigh, FIG.shin), foot);
    line(shoulder, knee(shoulder, bars, FIG.upper, FIG.fore), bars);
  }
  ctx.globalAlpha = 1;
  line(hip, shoulder);
  head(limb(shoulder, Math.PI - deg(32), FIG.head + 6));
}
