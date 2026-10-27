// Gait identifier page. Talks straight to the UNO Q over a WebSocket
// (ArduinoApps/gait-identifier/python/main.py, port 8765) -- no MQTT broker.
//
// Board -> page:  {"type": "imu", "t": ms, "a": [x, y, z] in g, "g": [x, y, z] in deg/s}
//                 {"type": "status", "text": "..."}
//                 {"type": "pong", "t": ...}   (answer to our ping)
//                 {"type": "recording", "phase": "idle"|"countdown"|"recording", "left": s, ...}
//                 {"type": "recordings", "items": [{file, person, spot, activity, started, note, seconds}]}
//                 {"type": "error", "text": "..."}
// Page -> board:  {"type": "ping", "t": ...}
//                 {"type": "record", person, spot, activity, seconds, countdown, started, note}
//                 {"type": "stop"}   {"type": "delete", "file": ...}

const PORT = 8765;
const DEFAULT_BOARD = "10.247.137.45";
const HISTORY = 500;  // readings kept for the lines: 10 seconds at 50/s
const COUNTDOWN_S = 10;  // time to strap in and get moving before recording starts
const GOAL_MIN = 2;      // minutes per spot+activity before the totals stop showing red
const ACTIVITY_NAMES = { stand: "standing", walk: "walking", run: "running", bike: "biking" };

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const COLORS = [css("--x"), css("--y"), css("--z")];

// ---------------- connection ----------------

const CONNECT_TIMEOUT_MS = 4000;  // an address that doesn't answer by then is given up on

let ws = null;
let host = "";          // the address confirmed with Connect/Enter (not half-typed text)
let retryTimer = null;
let readingsThisSecond = 0;

// Remembered settings (the page still works if the browser blocks storage).
function load(key, fallback) {
  try { return localStorage.getItem(key) || fallback; } catch { return fallback; }
}
function save(key, value) {
  try { localStorage.setItem(key, value); } catch {}
}

// The address to start with: the one used last time, unless DEFAULT_BOARD
// has been changed in this file since then (then the new default wins).
function startingAddress() {
  if (load("boardDefault", "") !== DEFAULT_BOARD) {
    save("boardDefault", DEFAULT_BOARD);
    save("board", DEFAULT_BOARD);  // forget the old remembered address too
    return DEFAULT_BOARD;
  }
  return load("board", DEFAULT_BOARD);
}

// Called by the Connect button / Enter: take the address from the box.
function connectToBox() {
  host = $("board").value.trim();
  save("board", host);
  connect();
}

// (Re)connect to `host`. Retries keep using `host`, so typing in the box
// doesn't disturb anything until Connect/Enter is pressed.
function connect() {
  clearTimeout(retryTimer);
  if (ws) ws.onclose = null, ws.close();
  setConnected(false, `Connecting to ${host}…`);
  try {
    ws = new WebSocket(`ws://${host}:${PORT}`);
  } catch {
    ws = null;
    return setConnected(false, `"${host}" isn't a valid address`);
  }
  const socket = ws;
  const giveUp = setTimeout(() => socket.close(), CONNECT_TIMEOUT_MS);
  ws.onopen = () => { clearTimeout(giveUp); setConnected(true, "Connected"); };
  ws.onclose = () => {
    clearTimeout(giveUp);
    setConnected(false, `Can't reach ${host} — retrying`);
    retryTimer = setTimeout(connect, 1000);
  };
  ws.onmessage = (e) => handle(JSON.parse(e.data));
}

function setConnected(on, text) {
  $("dot").classList.toggle("on", on);
  $("conn-text").textContent = text;
  if (!on) $("rate").textContent = $("ping").textContent = "–";
}

function send(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
}

function handle(msg) {
  if (msg.type === "imu") {
    accel.push(msg.a);
    gyro.push(msg.g);
    readingsThisSecond++;
  } else if (msg.type === "pong") {
    $("ping").textContent = (performance.now() - msg.t).toFixed(0);
  } else if (msg.type === "status") {
    console.log("board:", msg.text);
  } else if (msg.type === "recording") {
    showRecording(msg);
  } else if (msg.type === "recordings") {
    showRecordings(msg.items);
  } else if (msg.type === "models") {
    showModels(msg);
  } else if (msg.type === "live") {
    showLive(msg);
  } else if (msg.type === "guess") {
    showGuess(msg);
  } else if (msg.type === "error") {
    setRecStatus(`Board says: ${msg.text}`, "");
  }
}

// Once a second: show the reading rate and time a round trip to the board.
setInterval(() => {
  if (ws && ws.readyState === WebSocket.OPEN) $("rate").textContent = readingsThisSecond;
  readingsThisSecond = 0;
  send({ type: "ping", t: performance.now() });
}, 1000);

// ---------------- live lines ----------------

class LineChart {
  constructor(canvas, minRange) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.minRange = minRange;  // the y axis always shows at least +-this much
    this.data = [];
  }

  push(values) {
    this.data.push(values);
    if (this.data.length > HISTORY) this.data.shift();
  }

  draw() {
    const { canvas, ctx, data } = this;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    // Grow the y axis to fit the biggest value on screen.
    let range = this.minRange;
    for (const v of data) for (const x of v) range = Math.max(range, Math.abs(x) * 1.1);
    const y = (v) => h / 2 - (v / range) * (h / 2 - 4);

    // Zero line and scale labels.
    ctx.strokeStyle = css("--line");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();
    ctx.fillStyle = css("--muted");
    ctx.font = "11px system-ui";
    ctx.fillText(`+${range.toFixed(range < 10 ? 1 : 0)}`, 4, 12);
    ctx.fillText(`-${range.toFixed(range < 10 ? 1 : 0)}`, 4, h - 4);

    // One line per direction; newest reading at the right edge.
    const dx = w / (HISTORY - 1);
    const x0 = w - (data.length - 1) * dx;
    ctx.lineWidth = 1.5;
    for (let axis = 0; axis < 3; axis++) {
      ctx.strokeStyle = COLORS[axis];
      ctx.beginPath();
      data.forEach((v, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, x0 + i * dx, y(v[axis])));
      ctx.stroke();
    }
  }
}

const accel = new LineChart($("accel"), 2);
const gyro = new LineChart($("gyro"), 100);

function frame() {
  accel.draw();
  gyro.draw();
  requestAnimationFrame(frame);
}

// ---------------- pick-one buttons ----------------

// Turns a .seg row into a remembered choice; returns a function that reads it.
function segmented(id, fallback) {
  const box = $(id);
  let value = load(id, fallback);
  const paint = () => box.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.v === value));
  box.onclick = (e) => {
    if (!e.target.dataset.v) return;
    value = e.target.dataset.v;
    save(id, value);
    paint();
    onSpotOrActivityChange();
    if (id === "spot") onSpotChange();
  };
  paint();
  return () => value;
}

const spot = segmented("spot", "ankle");
const activity = segmented("activity", "walk");

// ---------------- recording ----------------

let recPhase = "idle";
let recordings = [];

function localTimestamp() {
  const d = new Date();
  return new Date(d - d.getTimezoneOffset() * 60000).toISOString().slice(0, 19);
}

$("record").onclick = () => {
  if (recPhase !== "idle") return send({ type: "stop" });
  speechSynthesis.cancel();  // (clicking also lets the page play sounds)
  send({
    type: "record", person: $("person").value.trim() || "me", spot: spot(), activity: activity(),
    seconds: +$("seconds").value, countdown: COUNTDOWN_S, started: localTimestamp(),
    note: $("note").value.trim(),
  });
};
$("person").onchange = () => save("person", $("person").value.trim());

function setRecStatus(text, cls) {
  $("rec-status").textContent = text;
  $("rec-status").className = "rec-status " + cls;
}

function showRecording(r) {
  const what = `${ACTIVITY_NAMES[r.activity] || ""} (${r.spot})`;
  if (r.phase === "countdown") {
    setRecStatus(`Get ready: ${what} in ${r.left}…`, "countdown");
    if (r.left <= 3) beep(660, 120);
  } else if (r.phase === "recording") {
    if (recPhase !== "recording") { beep(990, 300); say("Go!"); }
    setRecStatus(`Recording ${what}: ${r.left} s left`, "recording");
  } else {
    if (r.result === "cancelled") setRecStatus("Cancelled.", "");
    else if (r.saved) { setRecStatus(`Saved ${r.seconds} s of ${what}.`, ""); beep(990, 120); say("Done. Saved."); }
    else if (r.result) { setRecStatus("Too short to keep (the first and last 2 s are cut off).", ""); say("Too short."); }
  }
  recPhase = r.phase;
  $("record").textContent = recPhase === "idle" ? "Record" : "Stop";
  $("record").classList.toggle("stop", recPhase !== "idle");
}

// Minutes recorded for each spot + activity, and the list of recordings.
function showRecordings(items) {
  recordings = items;
  const acts = Object.keys(ACTIVITY_NAMES);
  let html = "<tr><th></th>" + acts.map((a) => `<th>${cap(a)}</th>`).join("") + "</tr>";
  for (const sp of ["ankle", "waist"]) {
    html += `<tr class="${sp === spot() ? "current" : ""}"><th>${cap(sp)}</th>`;
    for (const a of acts) {
      const min = items.filter((r) => r.spot === sp && r.activity === a).reduce((t, r) => t + r.seconds, 0) / 60;
      html += `<td class="${min < GOAL_MIN ? "low" : ""}">${min.toFixed(1)} min</td>`;
    }
    html += "</tr>";
  }
  $("totals").innerHTML = html;

  $("recordings").innerHTML = "<tr><th>When</th><th>Person</th><th>Spot</th><th>Activity</th><th>Length</th><th>Note</th><th></th></tr>" +
    items.slice().reverse().map((r) => `<tr>
      <td>${r.started.replace("T", " ").slice(5, 16)}</td><td>${esc(r.person)}</td><td>${cap(r.spot)}</td>
      <td>${cap(r.activity)}</td><td>${r.seconds} s</td><td>${esc(r.note)}</td>
      <td><button data-file="${esc(r.file)}" title="Delete">×</button></td></tr>`).join("");
}

$("recordings").onclick = (e) => {
  const file = e.target.dataset.file;
  if (file && confirm(`Delete this recording?\n${file}`)) send({ type: "delete", file });
};

function onSpotOrActivityChange() {
  showRecordings(recordings);
}

const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);

// ---------------- sounds (so you know what's happening away from the laptop) ----------------

let audio = null;
function beep(freq, ms) {
  audio = audio || new AudioContext();
  const osc = audio.createOscillator(), gain = audio.createGain();
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(0.3, audio.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.001, audio.currentTime + ms / 1000);
  osc.connect(gain).connect(audio.destination);
  osc.start();
  osc.stop(audio.currentTime + ms / 1000);
}
function say(text) {
  speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}

// ---------------- start ----------------
// (live.js and figure.js load after this file and add training + live guessing.)

host = $("board").value = startingAddress();
$("person").value = load("person", "");
$("connect").onclick = connectToBox;
$("board").onkeydown = (e) => { if (e.key === "Enter") connectToBox(); };  // (returning false would block typing)
connect();
requestAnimationFrame(frame);
