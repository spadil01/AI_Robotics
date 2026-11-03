// Step counter page. Talks straight to the UNO Q over a WebSocket
// (ArduinoApps/step-counter/python/main.py, port 8765) -- no MQTT broker.
//
// Board -> page:  {"type": "imu", "t": ms, "a": [x, y, z] in g, "g": [x, y, z] in deg/s, "d": cm (-1 = no echo, -2 = sensor not answering)}
//                 {"type": "recording", "phase": "idle"|"countdown"|"recording", "left"/"took": s, ...}
//                 {"type": "recordings", "items": [{file, person, height_cm, leg, activity, started, note, steps, meters, seconds}]}
//                 {"type": "settings", "setup": {...}}
//                 {"type": "status"|"error", "text": "..."}   {"type": "pong", "t": ...}
// Page -> board:  {"type": "ping", "t": ...}   {"type": "setup", person, height_cm, leg, activity, note}
//                 {"type": "record"}   {"type": "cancel"}   {"type": "steps", file, steps}   {"type": "meters", file, meters}
//                 {"type": "delete", file}

const PORT = 8765;
const DEFAULT_BOARD = "10.247.137.83";
const HISTORY = 1000;  // readings kept for the lines: 10 seconds at 100/s
const GOAL_STEPS = 500;  // counted steps per activity before the totals stop showing red
const GOAL_FAKE_MIN = 3; // minutes of "no steps" (fakes) wanted
const GOAL_MEASURED = 12; // trials with a measured distance wanted
const ACTIVITY_NAMES = {
  walk: "Walk", slow: "Slow walk", run: "Run", stairs: "Stairs",
  turns: "Turns & shuffles", fake: "No steps",
};
const HOW = {
  walk: "walk normally",
  slow: "walk slowly, like browsing a shop",
  run: "jog or run",
  stairs: "walk up the stairs and back down (as many times as you like)",
  turns: "turn around, side-step, shuffle in place",
  fake: "DON’T step: sit and tap your foot, swing your leg, sway, stamp while seated… (counts as 0 steps)",
};

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
  if (load("stepsBoardDefault", "") !== DEFAULT_BOARD) {
    save("stepsBoardDefault", DEFAULT_BOARD);
    save("stepsBoard", DEFAULT_BOARD);
    return DEFAULT_BOARD;
  }
  return load("stepsBoard", DEFAULT_BOARD);
}

function connectToBox() {
  host = $("board").value.trim();
  save("stepsBoard", host);
  connect();
}

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
  if (!on) $("rate").textContent = $("ping").textContent = $("dist").textContent = "–";
}

function send(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
}

function handle(msg) {
  if (msg.type === "imu") {
    accel.push(msg.a);
    gyro.push(msg.g);
    distance.push([msg.d]);
    readingsThisSecond++;
    lastDist = msg.d;
  } else if (msg.type === "pong") {
    $("ping").textContent = (performance.now() - msg.t).toFixed(0);
  } else if (msg.type === "status") {
    console.log("board:", msg.text);
    if (/check|failed/.test(msg.text)) setRecStatus(`Board says: ${msg.text}`, "");
  } else if (msg.type === "recording") {
    showRecording(msg);
  } else if (msg.type === "recordings") {
    showRecordings(msg.items);
  } else if (msg.type === "settings") {
    showSettings(msg);
  } else if (msg.type === "error") {
    setRecStatus(`Board says: ${msg.text}`, "");
  }
}

let lastDist = -1;

// Once a second: show the reading rate and the distance, and time a round trip.
setInterval(() => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    $("rate").textContent = readingsThisSecond;
    $("dist").textContent = lastDist === -2 ? "no signal" : lastDist < 0 ? "no echo" : lastDist.toFixed(1);
  }
  readingsThisSecond = 0;
  send({ type: "ping", t: performance.now() });
}, 1000);

// ---------------- live lines ----------------

class LineChart {
  // fromZero: the y axis goes 0..max (distances) instead of -range..+range.
  constructor(canvas, minRange, colors, fromZero = false) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.minRange = minRange;
    this.colors = colors;
    this.fromZero = fromZero;
    this.data = [];
  }

  push(values) {
    this.data.push(values);
    if (this.data.length > HISTORY) this.data.shift();
  }

  draw() {
    const { canvas, ctx, data, fromZero } = this;
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
    const y = fromZero ? (v) => h - 4 - (v / range) * (h - 8)
                       : (v) => h / 2 - (v / range) * (h / 2 - 4);

    // Zero line and scale labels.
    ctx.strokeStyle = css("--line");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, y(0));
    ctx.lineTo(w, y(0));
    ctx.stroke();
    ctx.fillStyle = css("--muted");
    ctx.font = "11px system-ui";
    const label = range.toFixed(range < 10 ? 1 : 0);
    ctx.fillText(fromZero ? label : `+${label}`, 4, 12);
    ctx.fillText(fromZero ? "0" : `-${label}`, 4, h - 4);

    // One line per value; newest reading at the right edge. Negative
    // distances mean "no echo", so the distance line has a gap there.
    const dx = w / (HISTORY - 1);
    const x0 = w - (data.length - 1) * dx;
    ctx.lineWidth = 1.5;
    this.colors.forEach((color, k) => {
      ctx.strokeStyle = color;
      ctx.beginPath();
      let pen = false;
      data.forEach((v, i) => {
        if (fromZero && v[k] < 0) return (pen = false);
        (pen ? ctx.lineTo : ctx.moveTo).call(ctx, x0 + i * dx, y(v[k]));
        pen = true;
      });
      ctx.stroke();
    });
  }
}

const accel = new LineChart($("accel"), 2, COLORS);
const gyro = new LineChart($("gyro"), 100, COLORS);
const distance = new LineChart($("distance"), 30, [css("--d")], true);

function frame() {
  accel.draw();
  gyro.draw();
  distance.draw();
  requestAnimationFrame(frame);
}

// ---------------- trial setup ----------------
// The board keeps the setup (so every open page shows the same one);
// the page just shows it and sends changes.

let setup = {};

function segmented(id) {
  const box = $(id);
  box.onclick = (e) => {
    if (!e.target.dataset.v) return;
    sendSetup({ [id]: e.target.dataset.v });
  };
}
["activity", "leg"].forEach(segmented);

function sendSetup(change) {
  Object.assign(setup, change);
  paintSetup();
  send({ type: "setup", ...change });
}

function paintSetup() {
  for (const id of ["activity", "leg"])
    $(id).querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.v === setup[id]));
  for (const [id, key] of [["person", "person"], ["note", "note"], ["height", "height_cm"]])
    if (document.activeElement !== $(id)) $(id).value = setup[key] ?? "";
  $("how").innerHTML = `<b>How:</b> press Record → 10 s to get in place → <b>stand still 2 s</b> → ` +
    `${esc(HOW[setup.activity] || "")}${setup.activity === "fake" ? "" : ", <b>counting every step out loud</b> (both feet)"} → ` +
    `<b>stand still 2 s</b> → Stop.` +
    (setup.activity === "fake" || setup.activity === "turns" ? "" :
      ` <b>For distance:</b> start with your toes on a tape line, walk back and forth between the lines, and count the lengths.`);
  showRecordings(recordings);
}

$("person").onchange = () => sendSetup({ person: $("person").value.trim() || "me" });
$("note").onchange = () => sendSetup({ note: $("note").value.trim() });
$("height").onchange = () => sendSetup({ height_cm: $("height").value.trim() });

function showSettings(msg) {
  setup = msg.setup;
  paintSetup();
}

// ---------------- recording ----------------

let recPhase = "idle";
let recordings = [];
let askFile = null;  // the trial waiting for its step count

$("record").onclick = () => {
  speechSynthesis.cancel();  // (clicking also lets the page play sounds)
  send({ type: "record" });
};
$("cancel").onclick = () => send({ type: "cancel" });

function setRecStatus(text, cls) {
  $("rec-status").textContent = text;
  $("rec-status").className = "rec-status " + cls;
}

function showRecording(r) {
  const what = ACTIVITY_NAMES[r.activity] || "";
  if (r.phase === "countdown") {
    setRecStatus(`Get in place: ${what} in ${r.left}…`, "countdown");
    if (r.left <= 3) beep(660, 120);
    askSteps(null);
  } else if (r.phase === "recording") {
    if (recPhase !== "recording") { beep(990, 300); say("Go!"); }
    setRecStatus(`Recording ${what}: ${r.took} s`, "recording");
  } else if (r.result) {
    if (r.result === "cancelled") { setRecStatus("Cancelled — nothing saved.", ""); say("Cancelled."); }
    else if (!r.saved) { setRecStatus("Too short to keep (under 3 s).", ""); say("Too short."); }
    else {
      setRecStatus(`Saved ${r.seconds} s of ${what}.`, "");
      beep(990, 120);
      if (r.steps == null) { say("Saved. How many steps?"); askSteps(r.saved); }
      else say("Saved.");
    }
  }
  recPhase = r.phase;
  $("record").textContent = recPhase === "idle" ? "Record" : "Stop";
  $("record").classList.toggle("stop", recPhase !== "idle");
  $("cancel").hidden = recPhase === "idle";
}

function askSteps(file) {
  askFile = file;
  $("ask-steps").hidden = !file;
  if (file) { $("steps-input").value = $("lengths-input").value = ""; showMeters(); $("steps-input").focus(); }
}
// Distance = lengths walked x the course length (remembered on this computer).
$("course").value = load("courseMeters", "2");
const meters = () => {
  const lengths = $("lengths-input").value.trim();
  return lengths === "" ? null : Math.round(+lengths * +$("course").value * 100) / 100;  // lengths x course
};
const showMeters = () => {
  const m = meters();
  $("meters-out").textContent = m == null ? "— leave lengths empty if the distance wasn’t measured"
    : `× ${$("course").value} m course = ${m} m`;
};
$("lengths-input").oninput = showMeters;
$("course").oninput = () => { save("courseMeters", $("course").value); showMeters(); };

$("ask-steps").onsubmit = (e) => {
  e.preventDefault();
  send({ type: "steps", file: askFile, steps: +$("steps-input").value });
  if (meters() != null) send({ type: "meters", file: askFile, meters: meters() });
  askSteps(null);
};

// Per activity: trials, minutes and counted steps; then the list of trials.
function showRecordings(items) {
  recordings = items;
  let html = "<tr><th></th><th>Trials</th><th>Minutes</th><th>Steps counted</th><th>Meters measured</th></tr>";
  for (const [a, name] of Object.entries(ACTIVITY_NAMES)) {
    const mine = items.filter((r) => r.activity === a);
    const min = mine.reduce((t, r) => t + r.seconds, 0) / 60;
    const steps = mine.reduce((t, r) => t + (r.steps || 0), 0);
    const missing = mine.filter((r) => r.steps == null).length;
    const low = a === "fake" || a === "turns" ? min < GOAL_FAKE_MIN : steps < GOAL_STEPS;
    html += `<tr class="${a === setup.activity ? "current" : ""}"><th>${name}</th>
      <td>${mine.length}</td><td>${min.toFixed(1)}</td>
      <td class="${low ? "low" : ""}">${a === "fake" ? "–" : steps}${missing ? ` <span class="warn">(${missing} not typed in)</span>` : ""}</td>
      <td>${a === "fake" || a === "turns" ? "–" : sumMeters(mine)}</td></tr>`;
  }
  const measured = items.filter((r) => r.meters > 0);
  html += `<tr><th>Measured walks</th><td>${measured.length}</td><td></td><td></td>
    <td class="${measured.length < GOAL_MEASURED ? "low" : ""}">${sumMeters(measured)}</td></tr>`;
  const people = new Set(items.map((r) => r.person));
  html += `<tr><th>People</th><td colspan="4">${people.size} ${people.size ? `(${[...people].map(esc).join(", ")})` : ""}</td></tr>`;
  $("totals").innerHTML = html;

  $("recordings").innerHTML = "<tr><th>When</th><th>Person</th><th>Activity</th><th>Leg</th><th>Length</th><th>Steps</th><th>Meters</th><th>Note</th><th></th></tr>" +
    items.slice().reverse().map((r) => `<tr>
      <td>${r.started.replace("T", " ").slice(5, 16)}</td><td>${esc(r.person)}</td><td>${ACTIVITY_NAMES[r.activity] || esc(r.activity)}</td>
      <td>${cap(r.leg)}</td><td>${r.seconds} s</td>
      <td><input class="steps ${r.steps == null ? "missing" : ""}" type="number" min="0" data-kind="steps" data-file="${esc(r.file)}" value="${r.steps ?? ""}" placeholder="?"></td>
      <td><input class="steps" type="number" min="0" step="0.1" data-kind="meters" data-file="${esc(r.file)}" value="${r.meters ?? ""}" placeholder="–"></td>
      <td>${esc(r.note)}</td>
      <td><button data-file="${esc(r.file)}" title="Delete">×</button></td></tr>`).join("");
}

$("recordings").onclick = (e) => {
  const file = e.target.dataset.file;
  if (e.target.tagName === "BUTTON" && file && confirm(`Delete this trial?\n${file}`)) send({ type: "delete", file });
};
$("recordings").onchange = (e) => {
  if (!e.target.classList.contains("steps")) return;
  const v = e.target.value.trim();
  const kind = e.target.dataset.kind;
  send({ type: kind, file: e.target.dataset.file, [kind]: v === "" ? null : +v });
};

const sumMeters = (rows) => {
  const m = rows.filter((r) => r.meters > 0);
  return m.length ? `${Math.round(m.reduce((t, r) => t + r.meters, 0))} m (${m.length})` : "0";
};
const cap = (s) => String(s ?? "").charAt(0).toUpperCase() + String(s ?? "").slice(1);
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

host = $("board").value = startingAddress();
$("connect").onclick = connectToBox;
$("board").onkeydown = (e) => { if (e.key === "Enter") connectToBox(); };
connect();
requestAnimationFrame(frame);
