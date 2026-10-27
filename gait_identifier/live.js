// Training, the score card and live guessing (uses $, send, spot, cap... from app.js).
//
// Board -> page:  {"type": "models", "training": bool, "reports": {spot: report}}   (see gait_model.py)
//                 {"type": "live", "on": bool, "spot": ..., "led": model name, agree, far, preview}
//                 {"type": "guess", spot, led, shown, guesses: {name: {gait, best, sure, why, shares}},
//                                   features: {peak: {X, Y, Z}, all: [...], far: {peak, all}}, ms}
//                 (gait may be "other": why = "split" or "unfamiliar")
// Page -> board:  {"type": "train"}
//                 {"type": "live", "on"?, "spot"?, "led"?, "agree"?, "far"?, "preview"?}

const MODEL_LABELS = {
  "peak-tree": "Model 1 · tree",
  "peak-knn": "Model 1 · KNN",
  "all-tree": "Model 2 · forest",
  "all-knn": "Model 2 · KNN",
};
const ACTS = ["stand", "walk", "run", "bike"];
const gaitColor = (g) => (ACTS.includes(g) ? css(`--${g}`) : css("--muted"));

let reports = {};
let liveState = { on: false, spot: "ankle", led: "all-tree", agree: 0.6, far: 1.0 };
let picked = load("picked", "all-tree");  // the score-card row whose mix-ups are shown
let shownGait = "";                        // what the big label / figure / LED show
let lastGuess = null;

// ---------------- training + score card ----------------

$("train").onclick = () => send({ type: "train" });

function showModels(msg) {
  reports = msg.reports;
  $("train").disabled = msg.training;
  $("train").textContent = msg.training ? "Training…" : "Train";
  const r = reports[spot()];
  $("train-note").textContent = msg.training ? "Learning from every recording…"
    : r ? `${cap(spot())} models trained ${r.trained} from ${r.slices} slices (${r.groups.join(", ")})`
    : `No ${spot()} models yet — record some examples, then press Train.`;
  showScores();
  showWhyLayout();
}

function showScores() {
  let html = "<tr><th>Model</th><th>Looks at</th><th>Ankle</th><th>Waist</th></tr>";
  for (const name of Object.keys(MODEL_LABELS)) {
    const looks = name.startsWith("peak")
      ? `peak along ${reports[spot()]?.models[name].axis ?? "?"}` : "all 6 readings";
    html += `<tr data-name="${name}" class="${name === picked ? "picked" : ""}">
      <th>${MODEL_LABELS[name]}</th><td class="muted">${looks}</td>`;
    for (const sp of ["ankle", "waist"]) {
      const m = reports[sp]?.models[name];
      html += `<td class="score">${m ? Math.round(m.score * 100) + "%" : "–"}</td>`;
    }
    html += "</tr>";
  }
  $("scores").innerHTML = html;
  showMixups();
}

$("scores").onclick = (e) => {
  const row = e.target.closest("tr[data-name]");
  if (!row) return;
  picked = row.dataset.name;
  save("picked", picked);
  showScores();
};

// "Of the slices that really were X, how many did it call Y?" as percentages.
function showMixups() {
  const r = reports[spot()], m = r?.models[picked];
  $("mix-title").textContent = `Mix-ups: ${MODEL_LABELS[picked]} on the ${spot()}`;
  if (!m) return ($("mixups").innerHTML = "");
  let html = `<tr><th>Really was ↓ / guessed →</th>${ACTS.map((a) => `<th>${cap(a)}</th>`).join("")}</tr>`;
  m.confusion.forEach((row, i) => {
    const total = row.reduce((a, b) => a + b, 0);
    if (!total) return;
    html += `<tr><th>${cap(ACTS[i])}</th>` + row.map((n, j) => {
      const pct = Math.round((100 * n) / total);
      return `<td class="${i === j ? "right" : n ? "wrong" : ""}">${n ? pct + "%" : "·"}</td>`;
    }).join("") + "</tr>";
  });
  if (m.axis_scores) {
    html += `<tr><td colspan="5" class="muted" style="text-align:left">Direction tried: ` +
      Object.entries(m.axis_scores).map(([a, s]) => `${a} ${Math.round(s * 100)}%`).join(" · ") +
      ` → kept ${m.axis}</td></tr>`;
  }
  $("mixups").innerHTML = html;
}

// ---------------- live ----------------

$("run").onclick = () => send({ type: "live", on: !liveState.on, spot: spot() });
$("led-pick").onclick = (e) => e.target.dataset.v && send({ type: "live", led: e.target.dataset.v });
// Hold one picture on the LED panel (stops Run) -- handy for showing them off.
$("led-show").onclick = (e) => e.target.dataset.v && send({ type: "live", preview: e.target.dataset.v });

function onSpotChange() {
  send({ type: "live", spot: spot() });
  showModels({ reports, training: $("train").disabled });
}

function showLive(msg) {
  liveState = msg;
  $("run").textContent = msg.on ? "Stop" : "Run";
  $("run").classList.toggle("stop", msg.on);
  $("led-name").textContent = MODEL_LABELS[msg.led];
  $("live-note").textContent = msg.on ? `Guessing 4× a second from the last 2 s (${msg.spot})` : "";
  if (!msg.on) {
    setShown("");
    $("guesses").innerHTML = '<p class="muted">Press Run to start guessing.</p>';
  }
  document.querySelectorAll("#led-pick button").forEach((b) => b.classList.toggle("on", b.dataset.v === msg.led));
  document.querySelectorAll("#led-show button").forEach((b) => b.classList.toggle("on", !msg.on && b.dataset.v === msg.preview));
  if (document.activeElement !== $("agree")) $("agree").value = Math.round(msg.agree * 100);
  if (document.activeElement !== $("far")) $("far").value = msg.far;
  paintOtherSettings();
}

// "Other" strictness sliders: labels update while dragging, the board hears on release.
function paintOtherSettings() {
  $("agree-val").textContent = `${$("agree").value}%`;
  $("far-val").textContent = `${(+$("far").value).toFixed(1)}×`;
  document.querySelectorAll("#why .limit").forEach((el) => (el.style.left = `${(+$("far").value / FAM_MAX) * 100}%`));
}
$("agree").value = 60;
$("far").value = 1;
$("agree").oninput = $("far").oninput = paintOtherSettings;
$("agree").onchange = () => send({ type: "live", agree: $("agree").value / 100 });
$("far").onchange = () => send({ type: "live", far: +$("far").value });

function setShown(gait) {
  shownGait = gait || "";
  $("now-label").textContent = ACTIVITY_NAMES[shownGait] || shownGait || "–";
  $("now-label").style.color = gaitColor(shownGait);
}

function showGuess(msg) {
  lastGuess = msg;
  setShown(msg.shown);
  $("live-note").textContent = `Guessing 4× a second from the last 2 s (${msg.spot}) · ${msg.ms} ms to think`;

  $("guesses").innerHTML = Object.keys(MODEL_LABELS).map((name) => {
    const g = msg.guesses[name];
    const bars = ACTS.map((a) => `<i style="width:${(g.shares[a] || 0) * 100}%;background:${gaitColor(a)}"></i>`).join("");
    return `<div class="row"><div class="top">
      <span class="${name === msg.led ? "led-pick" : ""}">${MODEL_LABELS[name]}${name === msg.led ? " (picked)" : ""}</span>
      <span><b style="color:${gaitColor(g.gait)}">${g.gait}</b>${g.why ? "" : ` · ${Math.round(g.sure * 100)}% sure`}</span></div>
      <div class="stack">${bars}</div>${otherReason(g, name, msg.features.far)}</div>`;
  }).join("");
  updateWhy(msg);
}

// Why a model said "other", in plain words.
function otherReason(g, name, far) {
  if (g.why === "unfamiliar") {
    const f = far[name.split("-")[0]];
    return `<div class="reason">Unfamiliar: ${f.toFixed(1)}× farther than usual from anything it learned (best match: ${g.best})</div>`;
  }
  if (g.why === "split") {
    const [a, b] = Object.entries(g.shares).sort((x, y) => y[1] - x[1]);
    return `<div class="reason">Split vote: ${a[0]} ${Math.round(a[1] * 100)}% vs ${b[0]} ${Math.round(b[1] * 100)}%</div>`;
  }
  return "";
}

$("share-legend").innerHTML = "How the vote split: " +
  ACTS.map((a) => `<i style="background:${gaitColor(a)}"></i>${cap(a)}`).join("");

// ---------------- why ----------------

const FAM_MAX = 3;  // the "how familiar" bars run from 0 to 3× the usual distance

// Builds the number lines/bars once per training; updateWhy just moves the markers.
function showWhyLayout() {
  const r = reports[spot()];
  if (!r) {
    $("fam-peak").innerHTML = $("fam-all").innerHTML = "";
    return ($("why-peak").innerHTML = $("why-all").innerHTML = '<p class="muted">Train first.</p>');
  }

  for (const fam of ["peak", "all"]) {
    $(`fam-${fam}`).innerHTML = `<div class="line" id="far-${fam}">
      <div class="top"><span>How unfamiliar <small>(distance from what it learned; red line = "Other")</small></span><span class="val">–</span></div>
      <div class="track"><div class="fill" style="width:0"></div><div class="limit"></div></div></div>`;
  }
  paintOtherSettings();

  $("why-peak").innerHTML = ["peak-tree", "peak-knn"].map((name) => {
    const m = r.models[name], end = m.zones[m.zones.length - 1].to;
    const zones = m.zones.map((z, i) => {
      const to = i + 1 < m.zones.length ? m.zones[i + 1].from : z.to;
      return `<div class="zone" style="left:${(z.from / end) * 100}%;width:${((to - z.from) / end) * 100}%;background:${gaitColor(z.gait)}">
        <span>${(to - z.from) / end > 0.08 ? z.gait : ""}</span></div>`;
    }).join("");
    return `<div class="line" id="why-${name}" data-end="${end}" data-axis="${m.axis}">
      <div class="top"><span>${MODEL_LABELS[name]}: peak along ${m.axis}</span><span class="val">–</span></div>
      <div class="track">${zones}<div class="marker" style="left:0"></div></div></div>`;
  }).join("");

  $("why-all").innerHTML = r.clues.map((c) => {
    const ticks = Object.entries(c.typical).map(([a, v]) =>
      `<div class="tick" title="typical ${a}" style="left:${Math.min(100, (v / c.max) * 100)}%;background:${gaitColor(a)}"></div>`).join("");
    return `<div class="line" data-index="${c.index}" data-max="${c.max}">
      <div class="top"><span>${c.name} <small>(${Math.round(c.importance * 100)}% of the forest's decisions)</small></span><span class="val">–</span></div>
      <div class="track"><div class="fill" style="width:0"></div>${ticks}</div></div>`;
  }).join("");
}

function updateWhy(msg) {
  for (const fam of ["peak", "all"]) {
    const el = document.getElementById(`far-${fam}`), f = msg.features.far[fam];
    if (!el) continue;
    el.querySelector(".fill").style.width = `${Math.min(100, (f / FAM_MAX) * 100)}%`;
    el.querySelector(".fill").style.background = f > liveState.far ? css("--bad") : "#3a4350";
    el.querySelector(".val").textContent = `${f.toFixed(1)}× usual`;
  }
  for (const name of ["peak-tree", "peak-knn"]) {
    const el = document.getElementById(`why-${name}`);
    if (!el) continue;
    const v = msg.features.peak[el.dataset.axis];
    el.querySelector(".marker").style.left = `${Math.min(100, (v / el.dataset.end) * 100)}%`;
    el.querySelector(".val").textContent = `${v.toFixed(2)} g → ${msg.guesses[name].gait}`;
  }
  document.querySelectorAll("#why-all .line").forEach((el) => {
    const v = msg.features.all[el.dataset.index];
    el.querySelector(".fill").style.width = `${Math.min(100, (v / el.dataset.max) * 100)}%`;
    el.querySelector(".val").textContent = v < 10 ? v.toFixed(2) : v.toFixed(0);
  });
}

// Hidden until asked for: most people only want the answer.
let whyOpen = load("whyOpen", "no") === "yes";
function paintWhy() {
  $("why").hidden = !whyOpen;
  $("why-toggle").classList.toggle("on", whyOpen);
}
$("why-toggle").onclick = () => { whyOpen = !whyOpen; save("whyOpen", whyOpen ? "yes" : "no"); paintWhy(); };
paintWhy();

// The same sensor charts as the Training tab, sharing their readings.
const accelLive = new LineChart($("accel-live"), 2);
const gyroLive = new LineChart($("gyro-live"), 100);
accelLive.data = accel.data;
gyroLive.data = gyro.data;
let dataOpen = load("dataOpen", "no") === "yes";
function paintData() {
  $("live-data").hidden = !dataOpen;
  $("data-toggle").classList.toggle("on", dataOpen);
}
$("data-toggle").onclick = () => { dataOpen = !dataOpen; save("dataOpen", dataOpen ? "yes" : "no"); paintData(); };
paintData();

// ---------------- tabs ----------------

function showTab(name) {
  save("tab", name);
  document.querySelectorAll(".tab").forEach((t) => (t.hidden = t.id !== `tab-${name}`));
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
}
$("tabs").onclick = (e) => e.target.dataset.tab && showTab(e.target.dataset.tab);
// index.html#live opens straight to the Live tab.
showTab(["training", "live"].includes(location.hash.slice(1)) ? location.hash.slice(1) : load("tab", "training"));

// ---------------- animation ----------------

function animateFigure(now) {
  drawFigure($("figure"), shownGait, now / 1000, gaitColor(shownGait));
  if (dataOpen && !$("tab-live").hidden) { accelLive.draw(); gyroLive.draw(); }
  requestAnimationFrame(animateFigure);
}
requestAnimationFrame(animateFigure);
showWhyLayout();
showScores();
